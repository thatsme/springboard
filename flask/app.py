
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
import yaml

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

ALLOWED_EXTENSIONS = set(['csv','txt','yaml'])
APP_FOLDER = '/app/'
UPLOAD_FOLDER = APP_FOLDER+'static/input/'
INPUT_FOLDER = APP_FOLDER+'static/input/'
OUTPUT_FOLDER = APP_FOLDER+'static/output/'
LOG_FOLDER = APP_FOLDER+'log/'
DEFAULT_ERRORMESSAGE = "Something wrong, Check error log"
mactive = []
datapack = {}
sessionlist = ["Select a session"]
test = "Test ...."
#v_config = "dropdown-menu"
#v_loaddata = "disabled"
#v_featureengineering = "disabled"
#v_fileoutput = "disabled"
#v_ensamble = "disabled"
context = {}
context["session"] = sessionlist
context["config"] = "dropdown-toggle"
context["config_head"] = "dropdown"
context["loaddata"] = "disabled"
context["loaddata_head"] = "disabled"
context["datawrangling"] = "disabled"
context["datawrangling_head"] = "disabled"
context["dataexploration"] = "disabled"
context["dataexploration_head"] = "disabled"
context["featureengineering"] = "disabled"
context["featureengineering_head"] = "disabled"
context["fileoutput"] = "disabled"
context["fileoutput_head"] = "disabled"
context["ensamble"] = "disabled"
context["ensamble_head"] = "disabled"
context["show_train"] = "hide"
context["show_test"] = "hide"
context["show_full"] = "hide"

DW_pipeline = []
DW_content = {}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#from flask_ext.navigation import Navigation

DEBUG = True
app = Flask(__name__)
@app.context_processor
def inject_session():
    return dict(context=context)

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

def DummyDWContent():
    DW_content["colname"] = "colname"
    DW_content["newname"] = ""
    DW_content["prefix"] = "new_"
    DW_content["action"] = "toint"
    
    DW_pipeline.append(DW_content)
    
def MoveUploadedFiles(src,destpath,dst):
    if(datapack["activesession"]):
        msrc = UPLOAD_FOLDER+src
        mdst = destpath+datapack["activesession"]+"_"+dst
        try:
            copyfile(msrc,mdst)
            return True
        except:
            logger.debug("Copy file error %s, %s", msrc, mdst)
            return False
            
            
  
def Filter(mstring, msubstr, flag): 
    if flag:
        return [str for str in mstring if not any(sub in str for sub in msubstr)]
    else:
        return [str for str in mstring if any(sub in str for sub in msubstr)]

def ResetDatapack():
        ## Reset previouse values
        datapack["test"] = ""
        datapack["train"] = ""
        datapack["full"] = ""
        datapack["test_loaded"] = False
        datapack["train_loaded"] = False
        datapack["full_loaded"] = False
        datapack["column_list"] = []

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

@app.route('/featureengineering')
def feature_engineering():
    return render_template('feature_engineering.html')

@app.route('/datawrangling')
def data_wrangling():
    return render_template('data_wrangling.html')

@app.route('/dataexploration')
def data_exploration():
    return render_template('data_exploration.html')

@app.route('/dummy')
def dummy():
    return render_template('tb_implemented.html', version=datapack)
    
@app.route('/startsession')
def start_session():
    
    sessionlist.append(m.startsession())
    context["session"] = sessionlist
    
    datapack["generatedsession"] = str(m.uuid)
    datapack["train_loaded"] = False
    datapack["test_loaded"] = False
    datapack["full_loaded"] = False
    
    #return render_template('session_started.html', version=m.__version__)
    logger.debug(datapack)
    
    try:
        return render_template('session_started.html', version=datapack)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/sessionstatus')
def session_status():
    
    try:
        return render_template('session_started.html', version=datapack)
    except:
        return render_template('tb_implemented.html', version=datapack)

@app.route('/sessionreset')
def session_reset():
    
    try:
        return render_template('session_reset.html', version=datapack)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/setsession', methods=['POST'])
def setsession():
    if request.method == 'POST':
        result = request.form
        # check if session anready exist in list 
        if(result.get('activesession') in mactive ):
            datapack["activesession"] = result.get('activesession')
            
        else:
            # if not append the session to a list 
            if(len(result.get('activesession'))==36):
                mactive.append(result.get('activesession'))
                datapack["activesession"] = mactive[-1]
            else:
                logger.debug("You must select a valid session")
                return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
                
        m.setsession(datapack["activesession"])
        ## Set plot output as image
        m.setPlotToImage(True)
        ## Set output folder 
        m.setOutputFolder(OUTPUT_FOLDER)
        
        ## After a session is set .. enable the menus 
        context["loaddata"] = "dropdown-toggle"
        context["loaddata_head"] = "dropdown"
            
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

    try:
        logger.debug(datapack)
        return render_template('session_started.html', version=datapack)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)


@app.route('/createpipeline/<type>, <returnpage>', methods=['GET', 'POST'])
def create_pipeline(type, returnpage):
    if(request.method == 'POST'):
        pass
            
    elif(request.method == 'GET'):
        DummyDWContent()
        stuff = {}
        colnames = m.getColumns(type)
        dtypes = m.getDtypes(type)
        dtypes_list = [x.name for x in dtypes]
        #logger.debug(dtypes_list)
        #logger.debug(colnames)
        
        select_actions = []
        select_actions.append({'name':'drop', 'value':'Drop'})
        select_actions.append({'name':'toint', 'value':'To Integer'})
        select_actions.append({'name':'tostring', 'value':'To String'})
        stuff["actions"] = select_actions
        stuff["colnames"] = colnames
        stuff["dtypes"] = dtypes_list
                
        try:
            return render_template('show_pipelines.html',rr=returnpage, pipelines=DW_pipeline, stuff=stuff, zip=zip)
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
            

@app.route('/splitdata/<key>')
def split_data(key):
    return render_template('tb_implemented.html')
    
 
@app.route('/showdescribe/<key>,<type>, <returnpage>')
def show_describe(key, type, returnpage):
    '''
    Fixing DataFrame.describe visualization for dataframe to list 
    plus index column as standard column
    '''
    df = m.getDescribe(type)
    df.round(3)
    cnames = df.columns.values.tolist()
    cnames.insert(0, "value")
    logger.info(df.info)
    dvalues = df.values.tolist()
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

    try:
        return render_template("show_describe.html", column_names=cnames, link_column="Index", row_data=rdata, zip=zip, rr=returnpage)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)


@app.route('/showlog/<key>, <returnpage>')
def show_log(key, returnpage):
    '''
    Show log file
    '''
    file = LOG_FOLDER+"log.txt"

    try:
        with open(file, "r") as f:
            content = f.read()
    except:
        logger.debug("File read exception", exc_info=True)
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
    try:
        return(render_template('show_text.html', content=content, rr=returnpage))
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/showtext/<key>, <type>, <returnpage>')
def show_text(key, type, returnpage):
    if(key=="info"):
        filename = OUTPUT_FOLDER+datapack["activesession"]+"_df_"+type+key+".txt"
    elif(key=="dtypes"):
        filename = OUTPUT_FOLDER+datapack["activesession"]+"_df_"+type+key+".txt"
    elif(key=="describe"):
        filename = OUTPUT_FOLDER+datapack["activesession"]+"_df_"+type+key+".txt"
    elif(key=="unna"):
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

    logger.debug(returnpage)
    
    return(render_template('show_text.html', content=content, rr=returnpage))

@app.route('/loaddictionaries', methods=['GET', 'POST'])
def load_dictionaries():
    if(request.method == 'POST'):
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            filename = datapack["activesession"]+"_dict_"+secure_filename(file.filename)
            file.save(os.path.join(app.config['OUTPUT_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/loaddictionaries')
        else:
            flash('Allowed file types are csv and txt')
            return redirect(request.url)

    elif(request.method == 'GET'):
        try:
            return render_template('load_dictionaries.html')
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

    
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if(request.method == 'POST'):
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
            return redirect('/upload')
        else:
            flash('Allowed file types are csv')
            return redirect(request.url)

    elif(request.method == 'GET'):
        try:
            return render_template('upload.html')
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
      
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
        try:   
            return render_template('file_list.html', your_list=list_of_files)
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/listoutput/<jolly>,<returnpage>')
def list_output(jolly, returnpage):
    #path = os.getcwd()+"/static/output"
    list_of_files = []
    filtered_names = []
    filtere_by_session = []
    for filename in os.listdir(OUTPUT_FOLDER):
        list_of_files.append(filename)

    logger.debug(list_of_files)
    logger.debug(datapack["activesession"])
    logger.debug(returnpage)
    logger.debug(jolly)
    
    
    if(len(list_of_files)==0):
        logger.warn('No files in input')
        return redirect('/')
    else:
        if(jolly=='all'):
            filtered_names = list(filter(lambda item: (datapack["activesession"] in item) , list_of_files))
        else:
            filtere_by_session = filter(lambda item: (datapack["activesession"] in item) , list_of_files)
            filtered_names = list(filter(lambda item: (jolly in item) , filtere_by_session))

        
        for i in filtered_names:
            logger.debug(i)
            
        
        try:
            return render_template('output_list.html', output_list=filtered_names, rr=returnpage)
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)

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
                
                try:
                    return render_template('general_config.html', data=general)
                except Exception as e:
                    return render_template('tb_implemented.html', version=datapack, error=e)


            except:
                logger.debug("Config File write exception", exc_info=True)
                return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
            
        elif(request.method == 'GET'):
            my_file = Path(dst)
            if my_file.is_file():        
                gc.read(dst)
            else:
                try:
                    copyfile(src, dst)
                    gc.read(dst)
                except:
                    logger.debug("Copy file error %s, %s", src, dst)

            general = gc[section_grid]
            try:
                return render_template('general_config.html', data=general) 
            except Exception as e:
                return render_template('tb_implemented.html', version=datapack, error=e)
        else:
            logger.debug("Illegal Method")
            return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
        
@app.route('/loaddata',methods = ['POST'])
def loaddata():
    if request.method == 'POST':
        ResetDatapack()
        result = request.form
        temp = []
        for r in result.getlist('fileselect'):
            temp.append(r)
            #logger.info("fileselect -> *"+r+"*")

        for key, value in result.items():
            logger.info("--> key *"+key+"* --> value *"+value+"*")
            #kkey = key.split("_")[0]
            if key in temp:
                datapack[value] = key
                logger.info("datapack value -> *"+datapack[value]+"* datapack key -> *"+value+"*")


        logger.info(datapack)
        # If selection in datapack contain both train and test loaded files
        # run the ML util and load files
        if(("train" in datapack and datapack["train"] != "") and ("test" in datapack and datapack["test"] != "")):
            MoveUploadedFiles(datapack["train"], OUTPUT_FOLDER, datapack["train"])
            MoveUploadedFiles(datapack["test"], OUTPUT_FOLDER, datapack["test"])
            
            if(m.loadSplittedData(OUTPUT_FOLDER+datapack["activesession"]+"_"+datapack["train"], OUTPUT_FOLDER+datapack["activesession"]+"_"+datapack["test"])):
                datapack["train_loaded"] = True
                datapack["test_loaded"] = True

                # Copy back the dataframe and generate some statistics 
                df_train = m.getTrain()
                df_test = m.getTest()

                ## Get info data on train and copy on session prefixed text file  
                buffer = io.StringIO()
                df_train.info(buf=buffer)
                s = buffer.getvalue()

                try:                
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_traininfo.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

                # Get info data on test and copy on session prefixed text file 
                buffer = io.StringIO()
                df_test.info(buf=buffer)
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_testinfo.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

                # Get dtypes data on train and copy on session prefixed text file
                buffer = io.StringIO()
                df_train.dtypes.to_string(buf=buffer)
                #logger.info(df_train.dtypes.to_dict())
                #logger.info(df_train.dtypes.tolist())
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_traindtypes.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

                # Get dtypes data on test and copy on session prefixed text file
                buffer = io.StringIO()
                df_test.dtypes.to_string(buf=buffer)
                s = buffer.getvalue()
                try:                
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_testdtypes.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
                
                # Get nunique and nan data on train and copy on session prefixed text file
                buffer = io.StringIO()
                na = df_train.isna().sum().to_frame(name='null')
                un = df_train.nunique().to_frame(name='unique')
                result = pd.concat([na, un], axis=1, sort=False)
                result.to_string(buf=buffer)
                
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_trainunna.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

                # Get nunique and nan data on train and copy on session prefixed text file
                buffer = io.StringIO()
                na = df_test.isna().sum().to_frame(name='null')
                un = df_test.nunique().to_frame(name='unique')
                result = pd.concat([na, un], axis=1, sort=False)
                result.to_string(buf=buffer)
                
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_testunna.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
                
                
            else:
                logger.debug("Error loading Dataframes on MUtil class", exc_info=True)
                return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
                        
        # If selection in datapack contain single full file ( pre splitting ) 
        # run the ML util and load the file
        elif(("full" in datapack and datapack["full"] != "")):        
            MoveUploadedFiles(datapack["full"], OUTPUT_FOLDER, datapack["full"])

            if(m.loadSingleData(OUTPUT_FOLDER+datapack["activesession"]+"_"+datapack["full"])):

                datapack["full_loaded"] = True
                context["show_full"] = "show"
                
                # Copy back the dataframe and generate some statistics 
                df = m.getCombined()
                
                ## Get info data on full and copy on session prefixed text file  
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_fullinfo.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
                    
                # Get dtypes data on full and copy on session prefixed text file
                buffer = io.StringIO()
                df.dtypes.to_string(buf=buffer)
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_fulldtypes.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
                
                
                # Get nunique and nan data on train and copy on session prefixed text file
                buffer = io.StringIO()
                na = df.isna().sum().to_frame(name='null')
                un = df.nunique().to_frame(name='unique')
                result = pd.concat([na, un], axis=1, sort=False)
                result.to_string(buf=buffer)
                
                s = buffer.getvalue()
                try:
                    with open(OUTPUT_FOLDER+datapack["activesession"]+"_df_fullunna.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

            else:
                logger.debug("Error loading Dataframes on MUtil class", exc_info=True)
                return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
        else:
            logger.error("Error, u have to select at list test/train csv or full single csv file")
            return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)       

        ## Enable the rest of menus after the data is loaded
        context["datawrangling"] = "dropdown-toggle"
        context["datawrangling_head"] = "dropdown"
        context["dataexploration"] = "dropdown-toggle"
        context["dataexploration_head"] = "dropdown"
        context["featureengineering"] = "dropdown-toggle"
        context["featureengineering_head"] = "dropdown"
        context["fileoutput"] = "dropdown-toggle"
        context["fileoutput_head"] = "dropdown"
        context["ensamble"] = "dropdown-toggle"
        context["ensamble_head"] = "dropdown"

        try:
            logger.debug(context)
            logger.debug(datapack)
            return redirect("/sessionstatus")
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

@app.route('/detail/<key>')
def detailmethod(key):
    mkeys = []
    mval = []
    for keys in config[key]:  
        mval.append(config[key][keys])
        mkeys.append(keys)

    zipped = zip(mkeys, mval)

    try:
        return render_template('section_detail.html', your_list=zipped)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/testpandas/<key>')
def test_pandas(key):

    try:
        df = pd.read_csv(INPUT_FOLDER+key)
    except:
        logger.debug("File read exception", exc_info=True)

    # link_column is the column that I want to add a button to
    try:
        return render_template("test_pandas.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                            link_column="PassengerId", zip=zip)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

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
    try:
        return render_template("test_pandas.html", column_names=cnames, row_data=rdata,
                           link_column="PassengerId", zip=zip)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)


@app.route('/columnlist/<type>,<returnpage>')
def column_list(type, returnpage):
    if(type=="train"):
        df = m.getTrain()
        cnames = df.columns.values.tolist()
        ctypes = df.dtypes.tolist()
    elif(type=="test"):
        df = m.getTest()
        cnames = df.columns.values.tolist()
        ctypes = df.dtypes.tolist()
    elif(type=="full"):
        df = m.getCombined()
        cnames = df.columns.values.tolist()
        ctypes = df.dtypes.tolist()
    else:
        logger.debug("Wrong dataframe type %s", type)
        return("")

    try:
        col = zip(cnames,ctypes)
        logger.info(col)
        return render_template("manage_columns.html", columns=col, rr=returnpage)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/featureseng', methods=['POST'])
def features_eng():
    if(request.method == 'POST'):
        result = request.form
        transformation_toint = []
        transformation_tostring = []
        transformation_subset = []
        transformation_todrop = []
        transformation_todummies = []
        for key, value in result.items():
            kkey = key.split('_')
            if(kkey[1]=="select"):
                if(value=="subset"):
                    transformation_subset.append(kkey[0])
                elif(value=="toint"):
                    transformation_toint.append(kkey[0])                    
                elif(value=="tostring"):
                    transformation_tostring.append(kkey[0])
                elif(value=="drop"):
                    transformation_todrop.append(kkey[0])                    
                elif(value=="dummies"):
                    transformation_todummies.append(kkey[0])                    
        
        logger.info(transformation_subset)
        logger.info(transformation_todummies)
        
        
        ## Now we manage the priorities of elaborations
        ## 1) - Data Types transformations 
        ## 2) - Dropping coluns
        ## 3) - Creating a subset and updating MLUtil object 

        try:           
            return render_template('form_debugger.html', result=result)
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)
        
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   
    
@app.route('/listcolumns/<key>')
def list_columns(key):

    try:
        df = pd.read_csv(INPUT_FOLDER+key)
    except:
        logger.debug("File read exception", exc_info=True)

    mlist = list(df.columns.values)
    df1 = pd.DataFrame({'Colums':mlist})
    # link_column is the column that I want to add a button to

    try:
        return render_template("test_pandas.html", column_names=df1.columns.values, row_data=list(df1.values.tolist()),
                           link_column="Columns", zip=zip)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/formdebugger', methods=['POST'])
def form_debugger():
    if(request.method == 'POST'):
        result = request.form
        logger.info(result)
        try:
            return render_template('form_debugger.html', result=result)
        except Exception as e:
            return render_template('tb_implemented.html', version=datapack, error=e)
    
@app.route('/template')
def template():
    try:
        return render_template('home.html')
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/ensamble', methods=['GET', 'POST'])
def ensamble():
    form = ReusableForm(request.form)
    if(request.method == 'POST' and form.validate()):
        flash('Thanks for registering')
    elif(request.method == 'GET'):
        pass
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=DEFAULT_ERRORMESSAGE)   

    try:
        return render_template('ensamble.html', form=form)
    except Exception as e:
        return render_template('tb_implemented.html', version=datapack, error=e)

@app.route('/about-us/')
def about():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')