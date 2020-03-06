
#logging.debug('message')
#logging.info('message')
#logging.warn('message')
#logging.error('message')
#logging.critical('message')

import re 
import os
from io import StringIO
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
from flask import g
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
from modular.util import Util

#mactive = []
#v_config = "dropdown-menu"
#v_loaddata = "disabled"
#v_featureengineering = "disabled"
#v_fileoutput = "disabled"
#v_ensamble = "disabled"


#from flask_ext.navigation import Navigation

DEBUG = True
app = Flask(__name__)
#app.app_context()
app.config.from_object("config.DevelopmentConfig")

@app.context_processor
def inject_session():
    return dict(context=app.config["CONTEXT"])


# SETTING werkzeug only to ERRORS
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

logging.basicConfig(filename=app.config["LOG_FOLDER"]+'log.txt', 
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                    filemode="w")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from app import views
from app import ensamble
import configparser as cp 

## BLUEPRINTS IMPORT
from datawrangling import datawrangling
from pipelines import pipelines
from sessions import sessions
from files import files
from util import util

## BLUEPRINTS REGISTRATION
app.register_blueprint(datawrangling)
app.register_blueprint(pipelines)
app.register_blueprint(sessions)
app.register_blueprint(files)
app.register_blueprint(util)

#config = app.config["C"]
#config.read('MlUtil.ini')

#gc = cp.ConfigParser()

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
    app.config["C"].read("MlUtil.ini")
    mylist = app.config["C"].sections()
    methodlist = Util.Filter(mylist, sublist, exclude)
    sublist = ['RS_']
    exclude = False 
    randomsearchlist = Util.Filter(mylist, sublist, exclude)
    sublist = ['GS_']
    exclude = False 
    gridsearchlist = Util.Filter(mylist, sublist, exclude)
    sublist = ['ITRS_']
    exclude = False 
    iterationrandomsearchlist = Util.Filter(mylist, sublist, exclude)

    mlist1 = Util.Filter(mylist,sublist, exclude)
    logger.info(app.config["M"])
    return render_template('section_list.html', your_list=methodlist)

@app.route('/featureengineering')
def feature_engineering():
    return render_template('feature_engineering.html')

@app.route('/dataexploration')
def data_exploration():
    return render_template('data_exploration.html')

@app.route('/dummy')
def dummy():
    return render_template('tb_implemented.html', version=app.config["DATAPACK"])
    

            

@app.route('/splitdata/<mkey>')
def split_data(mkey):
    return render_template('tb_implemented.html')
    
 


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
        if file and Util.allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            filename = app.config["DATAPACK"]["activesession"]+"_dict_"+secure_filename(file.filename)
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
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

    

        
@app.route('/loaddata',methods = ['POST'])
def loaddata():
    if request.method == 'POST':
        Util.ResetDatapack()
        result = request.form
        temp = []
        for r in result.getlist('fileselect'):
            temp.append(r)
            #logger.info("fileselect -> *"+r+"*")

        for key, value in result.items():
            logger.info("--> key *"+key+"* --> value *"+value+"*")
            #kkey = key.split("_")[0]
            if key in temp:
                app.config["DATAPACK"][value] = key
                logger.info("datapack value -> *"+app.config["DATAPACK"][value]+"* datapack key -> *"+value+"*")


        logger.info(app.config["DATAPACK"])
        # If selection in datapack contain both train and test loaded files
        # run the ML util and load files
        if(("train" in app.config["DATAPACK"] and app.config["DATAPACK"]["train"] != "") and ("test" in app.config["DATAPACK"] and app.config["DATAPACK"]["test"] != "")):
            Util.MoveUploadedFiles(app.config["DATAPACK"]["train"], app.config["OUTPUT_FOLDER"], app.config["DATAPACK"]["train"])
            Util.MoveUploadedFiles(app.config["DATAPACK"]["test"], app.config["OUTPUT_FOLDER"], app.config["DATAPACK"]["test"])
            
            if(app.config["M"].loadSplittedData(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+app.config["DATAPACK"]["train"], app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+app.config["DATAPACK"]["test"])):
                app.config["DATAPACK"]["train_loaded"] = True
                app.config["DATAPACK"]["test_loaded"] = True

                # Copy back the dataframe and generate some statistics 
                df_train = app.config["M"].getTrain()
                df_test = app.config["M"].getTest()

                ## Get info data on train and copy on session prefixed text file  
                buffer = StringIO()
                df_train.info(buf=buffer)
                s = buffer.getvalue()

                try:                
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_traininfo.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

                # Get info data on test and copy on session prefixed text file 
                buffer = StringIO()
                df_test.info(buf=buffer)
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_testinfo.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

                # Get dtypes data on train and copy on session prefixed text file
                buffer = StringIO()
                df_train.dtypes.to_string(buf=buffer)
                #logger.info(df_train.dtypes.to_dict())
                #logger.info(df_train.dtypes.tolist())
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_traindtypes.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

                # Get dtypes data on test and copy on session prefixed text file
                buffer = StringIO()
                df_test.dtypes.to_string(buf=buffer)
                s = buffer.getvalue()
                try:                
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_testdtypes.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                
                # Get nunique and nan data on train and copy on session prefixed text file
                buffer = StringIO()
                na = df_train.isna().sum().to_frame(name='null')
                un = df_train.nunique().to_frame(name='unique')
                result = pd.concat([na, un], axis=1, sort=False)
                result.to_string(buf=buffer)
                
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_trainunna.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

                # Get nunique and nan data on train and copy on session prefixed text file
                buffer = StringIO()
                na = df_test.isna().sum().to_frame(name='null')
                un = df_test.nunique().to_frame(name='unique')
                result = pd.concat([na, un], axis=1, sort=False)
                result.to_string(buf=buffer)
                
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_testunna.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                
                
            else:
                logger.debug("Error loading Dataframes on MUtil class", exc_info=True)
                return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                        
        # If selection in datapack contain single full file ( pre splitting ) 
        # run the ML util and load the file
        elif(("full" in app.config["DATAPACK"] and app.config["DATAPACK"]["full"] != "")):        
            Util.MoveUploadedFiles(app.config["DATAPACK"]["full"], app.config["OUTPUT_FOLDER"], app.config["DATAPACK"]["full"])

            if(app.config["M"].loadSingleData(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+app.config["DATAPACK"]["full"])):

                app.config["DATAPACK"]["full_loaded"] = True
                app.config["CONTEXT"]["show_full"] = "show"
                
                # Copy back the dataframe and generate some statistics 
                df = app.config["M"].getCombined()
                
                ## Get info data on full and copy on session prefixed text file  
                buffer = StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_fullinfo.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                    
                # Get dtypes data on full and copy on session prefixed text file
                buffer = StringIO()
                df.dtypes.to_string(buf=buffer)
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_fulldtypes.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                
                
                # Get nunique and nan data on train and copy on session prefixed text file
                buffer = StringIO()
                na = df.isna().sum().to_frame(name='null')
                un = df.nunique().to_frame(name='unique')
                result = pd.concat([na, un], axis=1, sort=False)
                result.to_string(buf=buffer)
                
                s = buffer.getvalue()
                try:
                    with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_fullunna.txt", "w", encoding="utf-8") as f:  
                        f.write(s)
                except:
                    logger.debug("File write exception", exc_info=True)
                    return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

            else:
                logger.debug("Error loading Dataframes on MUtil class", exc_info=True)
                return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
        else:
            logger.error("Error, u have to select at list test/train csv or full single csv file")
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])       

        ## Enable the rest of menus after the data is loaded
        app.config["CONTEXT"]["datawrangling"] = "dropdown-toggle"
        app.config["CONTEXT"]["datawrangling_head"] = "dropdown"
        app.config["CONTEXT"]["dataexploration"] = "dropdown-toggle"
        app.config["CONTEXT"]["dataexploration_head"] = "dropdown"
        app.config["CONTEXT"]["featureengineering"] = "dropdown-toggle"
        app.config["CONTEXT"]["featureengineering_head"] = "dropdown"
        app.config["CONTEXT"]["fileoutput"] = "dropdown-toggle"
        app.config["CONTEXT"]["fileoutput_head"] = "dropdown"
        app.config["CONTEXT"]["ensamble"] = "dropdown-toggle"
        app.config["CONTEXT"]["ensamble_head"] = "dropdown"

        try:
            logger.debug(app.config["CONTEXT"])
            logger.debug(app.config["DATAPACK"])
            return redirect("/sessionstatus")
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

@app.route('/detail/<mkey>')
def detailmethod(mkey):
    mkeys = []
    mval = []
    for keys in app.config["C"][mkey]:  
        mval.append(app.config["C"][mkey][keys])
        mkeys.append(keys)

    zipped = zip(mkeys, mval)

    try:
        return render_template('section_detail.html', your_list=zipped)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@app.route('/testpandas/<mkey>')
def test_pandas(mkey):

    try:
        df = pd.read_csv(INPUT_FOLDER+mkey)
    except:
        logger.debug("File read exception", exc_info=True)

    # link_column is the column that I want to add a button to
    try:
        return render_template("test_pandas.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                            link_column="PassengerId", zip=zip)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@app.route('/showdataframe/<mtype>,<where>,<num>')
def show_dataframe(mtype, where, num):
    # Da rivedere completamente
    if(mtype=="train"):
        df = app.config["M"].getTrain()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    elif(mtype=="test"):
        df = app.config["M"].getTest()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    else:
        logger.debug("Wrong dataframe type %s", mtype)
        return("")

    # link_column is the column that I want to add a button to
    try:
        return render_template("test_pandas.html", column_names=cnames, row_data=rdata,
                           link_column="PassengerId", zip=zip)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)



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
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)
        
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
    
@app.route('/listcolumns/<mkey>')
def list_columns(mkey):

    try:
        df = pd.read_csv(app.config["INPUT_FOLDER"]+key)
    except:
        logger.debug("File read exception", exc_info=True)

    mlist = list(df.columns.values)
    df1 = pd.DataFrame({'Colums':mlist})
    # link_column is the column that I want to add a button to

    try:
        return render_template("test_pandas.html", column_names=df1.columns.values, row_data=list(df1.values.tolist()),
                           link_column="Columns", zip=zip)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@app.route('/formdebugger', methods=['POST'])
def form_debugger():
    if(request.method == 'POST'):
        result = request.form
        logger.info(result)
        try:
            return render_template('form_debugger.html', result=result)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)
    
@app.route('/template')
def template():
    try:
        return render_template('home.html')
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@app.route('/ensamble', methods=['GET', 'POST'])
def ensamble():
    form = ReusableForm(request.form)
    if(request.method == 'POST' and forapp.config["M"].validate()):
        flash('Thanks for registering')
    elif(request.method == 'GET'):
        pass
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

    try:
        return render_template('ensamble.html', form=form)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@app.route('/about-us/')
def about():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')