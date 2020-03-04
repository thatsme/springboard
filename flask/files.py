from flask import Blueprint
from flask import render_template
from flask import request
from flask import redirect
from flask import current_app as app
from werkzeug.local import LocalProxy
from werkzeug.utils import secure_filename
import os

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
                ffile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
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
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@files.route('/listoutput/<jolly>,<returnpage>')
def list_output(jolly, returnpage):
    #path = os.getcwd()+"/static/output"
    list_of_files = []
    filtered_names = []
    filtere_by_session = []
    for filename in os.listdir(app.config["OUTPUT_FOLDER"]):
        list_of_files.append(filename)

    logger.debug(list_of_files)
    logger.debug(app.config["DATAPACK"]["activesession"])
    logger.debug(returnpage)
    logger.debug(jolly)
    
    
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
