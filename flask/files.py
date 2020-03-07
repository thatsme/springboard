from flask import Blueprint
from flask import render_template
from flask import request
from flask import redirect
from flask import current_app as app
from werkzeug.local import LocalProxy
from werkzeug.utils import secure_filename
from io import StringIO
from os import listdir
from os.path import join as jj
from modular.util import Util
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3)

from modular.util import Util

files = Blueprint('files', __name__)

logger = LocalProxy(lambda: app.logger)


@files.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if(request.method == 'POST'):
        # check if the post request has the file part
        if 'file[]' not in request.files:
            logger.debug('No file part')
            return redirect(request.url)
        mfile = request.files['file[]']
        logger.info(mfile)
        if mfile.filename == '':
            logger.debug('No file selected for uploading')
            return redirect(request.url)
        if mfile and Util.allowed_file(mfile.filename):
            uploaded_files = request.files.getlist("file[]")
            for ffile in uploaded_files:
                logger.info(uploaded_files)  
                filename = secure_filename(ffile.filename)
                #ffile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                ffile.save(jj(app.config['UPLOAD_FOLDER'], filename))
                logger.info('File successfully uploaded')
            return redirect('/upload')
        else:
            logger.debug('Allowed file types are csv')
            return redirect(request.url)

    elif(request.method == 'GET'):
        try:
            return render_template('upload.html')
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
      
@files.route('/listinput')
def list_input():
    list_of_files = []
    for filename in listdir(app.config["INPUT_FOLDER"]):
        list_of_files.append(filename)

    if(len(list_of_files)==0):
        logger.warn('No files in input')
        return redirect('/')
    else:   
        try:   
            return render_template('file_list.html', your_list=list_of_files)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@files.route('/listoutput/<jolly>,<returnpage>')
def list_output(jolly, returnpage):
    
    list_of_files = []
    filtered_names = []
    filtere_by_session = []
    
    for filename in listdir(app.config["OUTPUT_FOLDER"]):
        list_of_files.append(filename)

    #logger.debug(list_of_files)
    #logger.debug(app.config["DATAPACK"]["activesession"])
    #logger.debug(returnpage)
    #logger.debug(jolly)
    
    
    if(len(list_of_files)==0):
        logger.warn('No files in input')
        return redirect('/')
    else:
        if(jolly=='all'):
            filtered_names = list(filter(lambda item: (app.config["DATAPACK"]["activesession"] in item) , list_of_files))
        else:
            filtere_by_session = filter(lambda item: (app.config["DATAPACK"]["activesession"] in item) , list_of_files)
            filtered_names = list(filter(lambda item: (jolly in item) , filtere_by_session))
        
        try:
            return render_template('output_list.html', output_list=filtered_names, rr=returnpage)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)


@files.route('/showlog/<mkey>, <returnpage>')
def show_log(mkey, returnpage):
    '''
    Show log file
    '''
    file = app.config["LOG_FOLDER"]+"log.txt"

    try:
        with open(file, "r") as f:
            content = f.read()
    except:
        logger.debug("File read exception", exc_info=True)
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
    try:
        return(render_template('show_text.html', content=content, rr=returnpage))
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@files.route('/loaddata',methods = ['POST'])
def loaddata():
    if request.method == 'POST':
        Util.ResetDatapack()
        result = request.form
        temp = []
        for r in result.getlist('fileselect'):
            temp.append(r)
            #logger.info("fileselect -> *"+r+"*")

        for key, value in result.items():
            if Util.CheckTypes(value):
                logger.info("--> key *"+key+"* --> value *"+value+"*")
                #kkey = key.split("_")[0]
                if key in temp:
                    app.config["DATAPACK"][value] = key
                    logger.info("datapack value -> *"+app.config["DATAPACK"][value]+"* datapack key -> *"+value+"*")

        logger.info("=loaddata=====================================================================================")
        logger.info(app.config["DATAPACK"])
        # If selection in datapack contain both train and test loaded files
        # run the ML util and load files
        if(("train" in app.config["DATAPACK"] and app.config["DATAPACK"]["train"] != "") and ("test" in app.config["DATAPACK"] and app.config["DATAPACK"]["test"] != "")):
            Util.MoveUploadedFiles(app.config["DATAPACK"]["train"], app.config["OUTPUT_FOLDER"], app.config["DATAPACK"]["train"])
            Util.MoveUploadedFiles(app.config["DATAPACK"]["test"], app.config["OUTPUT_FOLDER"], app.config["DATAPACK"]["test"])
            
            if(app.config["M"].loadSplittedData(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+app.config["DATAPACK"]["train"], app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+app.config["DATAPACK"]["test"])):
                app.config["DATAPACK"]["train_loaded"] = True
                app.config["DATAPACK"]["test_loaded"] = True
                app.config["CONTEXT"]["show_train"] = "show"
                app.config["CONTEXT"]["show_test"] = "show"
                app.config["CONTEXT"]["show_full"] = "hide"

                # Copy back the dataframe and generate some statistics 
                df_train = app.config["M"].getTrain()
                df_test = app.config["M"].getTest()

                ## Get info data on train and copy on session prefixed text file 
                Util.writeInfo(df_train, "train")
                 
                # Get info data on test and copy on session prefixed text file                 
                Util.writeInfo(df_test,"test")

                # Get dtypes data on train and copy on session prefixed text file
                Util.writeDtypes(df_train,"train")

                # Get dtypes data on test and copy on session prefixed text file
                Util.writeDtypes(df_test,"test")
                
                # Get nunique and nan data on train and copy on session prefixed text file
                Util.writeNunique(df_train, "train")

                # Get nunique and nan data on test and copy on session prefixed text file
                Util.writeNunique(df_test, "test")                
                
            else:
                logger.debug("Error loading Dataframes on MUtil class", exc_info=True)
                return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                        
        # If selection in datapack contain single full file ( pre splitting ) 
        # run the ML util and load the file
        elif(("full" in app.config["DATAPACK"] and app.config["DATAPACK"]["full"] != "")):        
            Util.MoveUploadedFiles(app.config["DATAPACK"]["full"], app.config["OUTPUT_FOLDER"], app.config["DATAPACK"]["full"])

            if(app.config["M"].loadSingleData(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+app.config["DATAPACK"]["full"])):

                app.config["DATAPACK"]["full_loaded"] = True
                app.config["CONTEXT"]["show_train"] = "hide"
                app.config["CONTEXT"]["show_test"] = "hide"
                app.config["CONTEXT"]["show_full"] = "show"
                
                # Copy back the dataframe and generate some statistics 
                df = app.config["M"].getCombined()
                
                ## Get info data on full dataset and copy on session prefixed text file  
                Util.writeInfo(df, "full") 
                                   
                # Get dtypes data on full dataset and copy on session prefixed text file
                Util.writeDtypes(df, "full")
                
                # Get nunique and nan data on full dataset and copy on session prefixed text file
                Util.writeNunique(df, "full")
                
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
