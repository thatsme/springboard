from io import StringIO
import os
from pathlib import Path
from os.path import join as jj

from flask import Blueprint
from flask import render_template
from flask import request
from flask import redirect
from flask import current_app as app
from werkzeug.local import LocalProxy
from werkzeug.utils import secure_filename

from modular.util import Util
import pandas as pd
import numpy as np
from zipfile import ZipFile

pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3)


files = Blueprint('files', __name__)

logger = LocalProxy(lambda: app.logger)


@files.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if(request.method == 'POST'):
        # check if the post request has the file part
        upload_path = ""
        mpath = request.form.get('directoryselect')
        logger.info(mpath)
        if mpath != app.config['UPLOAD_FOLDER']:
            upload_path = mpath+"/"
        else:
            upload_path = app.config['UPLOAD_FOLDER']
        
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
                ffile.save(jj(upload_path, filename))
                logger.info('File successfully uploaded')
            return redirect('/upload')
        else:
            logger.debug('Allowed file types are csv')
            return redirect(request.url)

    elif(request.method == 'GET'):
        try:
            directory_list = []
            for path, dirs, files in os.walk(app.config["INPUT_FOLDER"]):
                directory_list.append(path)
                
            return render_template('upload.html', dir_list=directory_list)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

@files.route('/unzipdata', methods=['GET'])
def unzip_data():
    if(request.method == 'GET'):
        mpath = request.args.get('mpath')
        mkey = request.args.get('mkey')

        zf = ZipFile(mpath+mkey, 'r')
        zf.extractall(mpath)
        zf.close()
        
        return redirect('/listfiles?mkey=none&mpath=.')

@files.route('/showimage', methods=['GET'])
def show_image():
    if(request.method == 'GET'):
        mpath = request.args.get('mpath')
        mkey = request.args.get('mkey')
        ## Removing \app\ from current path for displaying as static file
        mpath = mpath[5:] 
        try:   
            return render_template('show_image.html', file_list=mkey, current_path = mpath)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)


@files.route('/listfiles', methods=['GET'])
def list_files():
    if(request.method == 'GET'):
        tabular_data = [".csv", ".txt",".CSV", ".TXT"]
        compress_data = [".zip",".ZIP"]
        image_data = [".png",".jpg",".jpeg",".png", ".gif",".PNG",".JPG",".JPEG",".PNG", ".GIF"]
        mpath = request.args.get('mpath')
        if(mpath != "."):
            start_path = mpath+'/'
        else:
            start_path = app.config["INPUT_FOLDER"]
        
        directory_list = []
        for path, dirs, files in os.walk(app.config["INPUT_FOLDER"]):
            directory_list.append(path)

        list_of_files = []
        extension = ""
        for filename in os.listdir(start_path):
            if os.path.isfile(start_path+filename):
                ff, file_extension = os.path.splitext(start_path+filename)
                if(file_extension in tabular_data):
                    extension = "tabular"
                elif(file_extension in compress_data):
                    extension = "compress"
                elif(file_extension in image_data):
                    extension = "image"
                list_of_files.append((filename,extension))

        if(len(list_of_files)==0):
            logger.warn('No files in input')
            return redirect('/')
        else:   
            try:   
                return render_template('file_list.html', file_list=list_of_files, dir_list=directory_list, current_path = start_path)
            except Exception as e:
                return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)
      
@files.route('/listdirectories', methods=['GET', 'POST'])
def list_directories():
    if(request.method == 'GET'):
        mpath = request.args.get('mpath')
        current_directory = request.args.get('current')

        if(mpath and mpath != "."):
            start_path = current_directory+mpath+'/'
        else:
            start_path = app.config["INPUT_FOLDER"]
        
        list_of_dir = []
        for filename in os.listdir(start_path):
            if os.path.isdir(start_path+filename):
                list_of_dir.append(filename)

        if(len(list_of_dir)==0):
            logger.warn('No directories in input')
        try:   
            return render_template('directory_selection.html', your_list=list_of_dir, current_path = start_path)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        pass

@files.route('/createstructure', methods=['GET', 'POST'])
def create_structure():
    if(request.method == 'GET'):
        mpath = request.args.get('mpath')
        current_directory = request.args.get('current')
        if(mpath and mpath != "."):
            start_path = current_directory+mpath+'/'
        else:
            start_path = app.config["INPUT_FOLDER"]
        
        
        logger.warn(start_path)
        list_of_dir = []
        for filename in os.listdir(start_path):
            if os.path.isdir(start_path+filename):
                list_of_dir.append(filename)

        if(len(list_of_dir)==0):
            logger.warn('No directories in input')

        try:   
            return render_template('directory_list.html', your_list=list_of_dir, current_path = start_path)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        mpath = request.form.get('newdirectory')
        current = request.form.get('currentpath')
        newdir = current+mpath
        logger.info(newdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        return redirect('/createstructure')


@files.route('/listoutput/<jolly>,<returnpage>')
def list_output(jolly, returnpage):
    
    list_of_files = []
    filtered_names = []
    filtere_by_session = []
    
    for filename in os.listdir(app.config["OUTPUT_FOLDER"]):
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
        
        ## First check if we have a file selection
        for r in result.getlist('fileselect'):
            temp.append(r)
            #logger.info("fileselect -> *"+r+"*")

        if(len(temp)>0):
            for key, value in result.items():
                if Util.CheckTypes(value):
                    logger.info("--> key *"+key+"* --> value *"+value+"*")
                    #kkey = key.split("_")[0]
                    if key in temp:
                        app.config["DATAPACK"][value] = key
                        logger.info("datapack value -> *"+app.config["DATAPACK"][value]+"* datapack key -> *"+value+"*")
        else:
            ## First check if we have a file selection
            for r in result.getlist('dirselect'):
                temp.append(r)
                #logger.info("fileselect -> *"+r+"*")
                
            if(len(temp)>0):
                for key, value in result.items():
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
            
        elif(("directory" in app.config["DATAPACK"] and app.config["DATAPACK"]["directory"] != "")):
            
            workpath = Util.fixPath(app.config["DATAPACK"]["directory"])
            
            if(app.config["M"].loadDirectoryData(workpath)):
                
                logger.info("==================================================================")
                logger.info(app.config["DATAPACK"]["directory"])
                
                app.config["DATAPACK"]["train_loaded"] = False
                app.config["DATAPACK"]["test_loaded"] = False
                app.config["DATAPACK"]["directory_loaded"] = True
                
                app.config["CONTEXT"]["show_train"] = "hide"
                app.config["CONTEXT"]["show_test"] = "hide"
                app.config["CONTEXT"]["show_full"] = "hide"
                app.config["CONTEXT"]["show_directory"] = "show"
                
                df_images = app.config["M"].getImagesStats()
                df_images_categories = app.config["M"].getImagesCategories()
                
                Util.exportDf(df_images, "directory")
                Util.exportDf(df_images_categories, "images_cat")
                
                # Get nunique and nan data on train and copy on session prefixed text file
                Util.writeNunique(df_images, "directory")
                
                ## Get info data on full dataset and copy on session prefixed text file  
                Util.writeInfo(df_images, "directory") 
                                   
                # Get dtypes data on full dataset and copy on session prefixed text file
                Util.writeDtypes(df_images, "directory")
                
                Util.writeDescribe(df_images, "describe")

            else:
                logger.error("Error, problem in loading directory structure and image files")
                logger.error(app.config["M"].getLastException())
            
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
        ##
        app.config["CONTEXT"]["trainmodel"] = "dropdown-toggle"
        app.config["CONTEXT"]["trainmodel_head"] = "dropdown"
        app.config["CONTEXT"]["predictmodel"] = "dropdown-toggle"
        app.config["CONTEXT"]["predictmodel_head"] = "dropdown"
        app.config["CONTEXT"]["models"] = "dropdown-toggle"
        app.config["CONTEXT"]["models_head"] = "dropdown"
        ##
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

@files.route('/testpandas',methods = ['GET'])
def test_pandas():
    if(request.method == 'GET'):
        mpath = request.args.get('mpath')
        mkey = request.args.get('mkey')

        try:
            df = pd.read_csv(mpath+mkey)
        except:
            logger.debug("File read exception", exc_info=True)

        # link_column is the column that I want to add a button to
        try:
            return render_template("test_pandas.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                                link_column="PassengerId", zip=zip)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@files.route('/listcolumns', methods = ['GET'])
def list_columns():
    if(request.method == 'GET'):
        mpath = request.args.get('mpath')
        mkey = request.args.get('mkey')

        try:
            df = pd.read_csv(mpath+mkey)
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

@files.route('/loaddictionaries', methods=['GET', 'POST'])
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
            chapter1 = "The file must be in yaml format, see below for an example, the purpose is a category reduction during feature engineering"
            chapter2 =  """
            {
            "Capt" : "Officier",\n
            "Col" : "Officier", \n
            "Major" : "Officier",\n
            "Jonkheer" : "Royalty",\n
            "Don" : "Royalty",\n
            "Sir" : "Royalty",\n
            "Dr" : "Officier",\n
            .....\n
            "Lady" : "Royalty"
            }
            """
            return render_template('load_dictionaries.html', chapter1=chapter1, chapter2=chapter2)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
