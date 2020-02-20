
#logging.debug('message')
#logging.info('message')
#logging.warn('message')
#logging.error('message')
#logging.critical('message')

import re 
import os
import io
from pathlib import Path
from shutil import copyfile
import logging

#import magic
#import urllib.request
from flask import Flask
from flask import render_template
from flask import request
from flask import flash
from flask import redirect
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, PasswordField
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3)
from modular.SklearnHelper import SklearnHelper
from modular.MlUtil import MlUtil 
from modular.transformers import (CategoriesExtractor, CountryTransformer, GoalAdjustor,
                          TimeTransformer)

ALLOWED_EXTENSIONS = set(['csv'])
APP_FOLDER = '/app/'
UPLOAD_FOLDER = APP_FOLDER+'static/input/'
INPUT_FOLDER = APP_FOLDER+'static/input/'
OUTPUT_FOLDER = APP_FOLDER+'static/output/'
LOG_FOLDER = APP_FOLDER+'log/'
DEFAULT_ERRORMESSAGE = "Something wrong, Check error log"
mactive = []
datapack = {}
sessionlist = ["Select a session..."]
test = "Test ...."

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#from flask_ext.navigation import Navigation

DEBUG = True
app = Flask(__name__)
@app.context_processor
def inject_session():
    return dict(listsession=sessionlist, mysession=mactive)

app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# SETTING werkzeug only to ERRORS
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

logging.basicConfig(filename=LOG_FOLDER+'log.txt', 
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                    filemode="w")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from app import views
from app import ensamble
import configparser as cp 

m = MlUtil()

config = cp.ConfigParser()
config.read('MlUtil.ini')

gc = cp.ConfigParser()

  
def Filter(mstring, msubstr, flag): 
    if flag:
        return [str for str in mstring if not any(sub in str for sub in msubstr)]
    else:
        return [str for str in mstring if any(sub in str for sub in msubstr)]

class ReusableForm(Form):
    username = TextField('Name:', validators=[validators.required()])
    email = StringField('Email Address', [validators.Length(min=6, max=35)])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])


@app.route('/')
def home():
    sublist = ['RS_', 'GS_', 'ITRS_']
    exclude = True 
    mylist = config.sections()
    methodlist = Filter(mylist, sublist, exclude)
    sublist = ['RS_']
    exclude = False 
    randomsearchlist = Filter(mylist, sublist, exclude)
    sublist = ['GS_']
    exclude = False 
    gridsearchlist = Filter(mylist, sublist, exclude)
    sublist = ['ITRS_']
    exclude = False 
    iterationrandomsearchlist = Filter(mylist, sublist, exclude)

    mlist1 = Filter(mylist,sublist, exclude)
    return render_template('section_list.html', your_list=methodlist)


@app.route('/startsession')
def start_session():
    
    sessionlist.append(m.startsession())
    datapack["generatedsession"] = str(m.uuid)
    datapack["train_loaded"] = False
    datapack["test_loaded"] = False
    datapack["full_loaded"] = False
    
    #return render_template('session_started.html', version=m.__version__)
    logger.debug(datapack)
    return render_template('session_started.html', version=datapack)

@app.route('/sessionstatus')
def session_status():
    
    return render_template('session_started.html', version=datapack)

@app.route('/setsession', methods=['POST'])
def setsession():
    if request.method == 'POST':
        result = request.form
        # check if session anready exist in list 
        if(result.get('activesession') in mactive ):
            datapack["activesession"] = result.get('activesession')
            m.setsession(datapack["activesession"])
        else:
            # if not append the session to a list 
            mactive.append(result.get('activesession'))
            datapack["activesession"] = mactive[-1]
            m.setsession(datapack["activesession"])

        if("train" in datapack and "test" in datapack):
            if(m.loadSplittedData(UPLOAD_FOLDER+datapack["train"], UPLOAD_FOLDER+datapack["test"])):
                datapack["train_loaded"] = True
                datapack["test_loaded"] = True

                df_train = m.getTrain()
                df_test = m.getTest()

                ## Get info data on train
                buffer = io.StringIO()
                df_train.info(buf=buffer)
                s = buffer.getvalue()
                with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_traininfo.txt", "w", encoding="utf-8") as f:  
                    f.write(s)

                # Get info data on test
                buffer = io.StringIO()
                df_test.info(buf=buffer)
                s = buffer.getvalue()
                with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_testinfo.txt", "w", encoding="utf-8") as f:  
                    f.write(s)

                # Get dtypes data on train
                buffer = io.StringIO()
                df_train.dtypes.to_string(buf=buffer)
                s = buffer.getvalue()
                with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_traindtypes.txt", "w", encoding="utf-8") as f:  
                    f.write(s)

                # Get dtypes data on test
                buffer = io.StringIO()
                df_test.dtypes.to_string(buf=buffer)
                s = buffer.getvalue()
                with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_testdtypes.txt", "w", encoding="utf-8") as f:  
                    f.write(s)



        elif("full" in datapack):        
            if(m.loadSingleData(UPLOAD_FOLDER+datapack["full"])):
                datapack["full_loaded"] = True
                df = m.getCombined()
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                with open(OUTPUT_FOLDER+datapack["activesession"]+"df_fullinfo.txt", "w", encoding="utf-8") as f:  
                    f.write(s)

        else:
            logger.error("Error, u have to select at list test/train csv or full single csv file")
            return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)       

    #return(str(mactive))
    return render_template('session_started.html', version=datapack)
    #return render_template("result.html",result = result)

@app.route('/showdescribe/<key>,<type>')
def show_describe(key, type):
    df = m.getDescribe(type)
    df.round(3)
    cnames = df.columns.values.tolist()
    cnames.insert(0, "value")
    logger.info(df.info)
    dvalues = df.values.tolist()
    #d = np.around(dvalues, 3)
    #dval = d.tolist()
    ivalues = df.index.values.tolist()
    final = []
    for c in dvalues:
        logger.info(c)
        for p in ivalues:
            #logger.info(p)
            c.insert(0, p)
            #logger.info(c)
            final.append(c)
            #logger.info(final)
            ivalues.pop(0)
            break


    rdata = list(final)

    return render_template("show_describe.html", column_names=cnames, link_column="Index", row_data=rdata, zip=zip)


@app.route('/showlog/<key>')
def show_log(key):
    file = LOG_FOLDER+"log.txt"

    try:
        with open(file, "r") as f:
            content = f.read()
    except:
        logger.debug("File read exception", exc_info=True)
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

    return(render_template('show_text.html', content=content))

@app.route('/showtext/<key>,<type>')
def show_text(key, type):
    if(key=="info"):
        filename = OUTPUT_FOLDER+datapack["activesession"]+"_df_"+type+key+".txt"
    elif(key=="dtypes"):
        filename = OUTPUT_FOLDER+datapack["activesession"]+"_df_"+type+key+".txt"
    elif(key=="describe"):
        filename = OUTPUT_FOLDER+datapack["activesession"]+"_df_"+type+key+".txt"
    else:
        if(type=='key'):
            filename = OUTPUT_FOLDER+key
        else:

            logger.debug("Missing key value")
            return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

    try:
        logger.info("File to open %s %s ", filename, key)
        with open(filename, "r") as f:
            content = f.read()
    except:
        logger.debug("File read exception", exc_info=True)
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

    return(render_template('show_text.html', content=content))

@app.route('/upload')
def upload_form():
	return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			return redirect('/')
		else:
			flash('Allowed file types are csv')
			return redirect(request.url)

@app.route('/listinput')
def list_input():
    path = os.getcwd()+"/static/input"
    list_of_files = []
    for filename in os.listdir(path):
        list_of_files.append(filename)

    if(len(list_of_files)==0):
        logger.warn('No files in input')
        return redirect('/')
    else:      
        return render_template('file_list.html', your_list=list_of_files)

@app.route('/listoutput/<jolly>')
def list_output(jolly):
    path = os.getcwd()+"/static/output"
    list_of_files = []
    for filename in os.listdir(path):
        list_of_files.append(filename)

    if(len(list_of_files)==0):
        logger.warn('No files in input')
        return redirect('/')
    else:
        if(jolly=='all'):
            filtered_names = filter(lambda item: (datapack["activesession"] in item) , list_of_files)
        else:
            filtere_by_session = filter(lambda item: (datapack["activesession"] in item) , list_of_files)
            filtered_names = filter(lambda item: (jolly in item) , filtere_by_session)

        return render_template('output_list.html', your_list=filtered_names)

@app.route('/generalconfig',methods = ['POST', 'GET'])
def general_config():

    if not mactive:
        logger.error("Error, u have at list generate and select an active session for setting General Configuration")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)       
    else:
        src = APP_FOLDER+"general_config.ini"
        dst = OUTPUT_FOLDER+datapack["activesession"]+"_"+"general_config.ini"
        section_grid = "GENERAL"
        if(request.method == 'POST'):
            ## Scrive le modifiche sul file di configurazione x sessione
            result = request.form
            #request.form.get('name')
            logger.info(result.get('logname'))

            for key, value in result.items():
                gc.set(section_grid, key, value)
 
            try:
                with open(dst, 'w') as configfile:
                    gc.write(configfile)
                    logger.info("writing config file done")

                general = gc[section_grid]
                return render_template('general_config.html', data=general)

            except:
                logger.debug("Config File write exception", exc_info=True)
                return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
            
        elif(request.method == 'GET'):
            my_file = Path(dst)
            if my_file.is_file():        
                gc.read(dst)
            else:
                copyfile(src, dst)
                gc.read(dst)

            general = gc[section_grid]
            return render_template('general_config.html', data=general)
        
@app.route('/loaddata',methods = ['POST', 'GET'])
def laoddata():
    if request.method == 'POST':
        ## Reset previouse values
        datapack["test"] = ""
        datapack["train"] = ""
        datapack["full"] = ""
        datapack["test_loaded"] = False
        datapack["train_loaded"] = False
        datapack["full_loaded"] = False

        result = request.form
        temp = []
        for r in result.getlist('fileselect'):
            temp.append(r)

        for key, value in result.items():
            if key in temp:
                datapack[value] = key

        return redirect("/startsession")
        #return render_template("result.html",result = result)

@app.route('/detail/<key>')
def detailmethod(key):
    mkeys = []
    mval = []
    for keys in config[key]:  
        mval.append(config[key][keys])
        mkeys.append(keys)
    zipped = zip(mkeys, mval)

    return render_template('section_detail.html', your_list=zipped)

@app.route('/testpandas/<key>')
def test_pandas(key):

    try:
        df = pd.read_csv(INPUT_FOLDER+key)
    except:
        logger.debug("File read exception", exc_info=True)

    # link_column is the column that I want to add a button to
    return render_template("test_pandas.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                           link_column="PassengerId", zip=zip)

@app.route('/showdataframe/<type>,<where>,<num>')
def show_dataframe(type, where, num):
    if(type=="train"):
        df = m.getTrain()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    elif(type=="test"):
        df = m.getTest()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    else:
        logger.debug("Wrong dataframe type %s", type)
        return("")

    # link_column is the column that I want to add a button to
    return render_template("test_pandas.html", column_names=cnames, row_data=rdata,
                           link_column="PassengerId", zip=zip)


@app.route('/columnlist/<type>')
def column_list(type):
    if(type=="train"):
        df = m.getTrain()
        cnames = df.columns.values
    elif(type=="test"):
        df = m.getTest()
        cnames = df.columns.values
    elif(type=="full"):
        df = m.getCombined()
        cnames = df.columns.values

    else:
        logger.debug("Wrong dataframe type %s", type)
        return("")

    return render_template("manage_columns.html", columns=cnames)


@app.route('/listcolumns/<key>')
def list_columns(key):

    try:
        df = pd.read_csv(INPUT_FOLDER+key)
    except:
        logger.debug("File read exception", exc_info=True)

    mlist = list(df.columns.values)
    df1 = pd.DataFrame({'Colums':mlist})
    # link_column is the column that I want to add a button to

    return render_template("test_pandas.html", column_names=df1.columns.values, row_data=list(df1.values.tolist()),
                           link_column="Columns", zip=zip)

@app.route('/template')
def template():
    return render_template('home.html')

@app.route('/ensamble', methods=['GET', 'POST'])
def ensamble():
    form = ReusableForm(request.form)
    if request.method == 'POST' and form.validate():
        flash('Thanks for registering')
        #return redirect(url_for('ensamble'))
    return render_template('ensamble.html', form=form)

@app.route('/about-us/')
def about():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')