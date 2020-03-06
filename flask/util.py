from flask import Blueprint
from flask import render_template
from flask import request
from flask import redirect
from flask import current_app as app
from flask import jsonify
from werkzeug.local import LocalProxy
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from shutil import copyfile

from modular.util import Util

util = Blueprint('util', __name__)

logger = LocalProxy(lambda: app.logger)

@util.route('/showclick', methods = ['POST'])
def show_click():
    if(request.method == 'POST'):
        try:
            req_data = request.get_json()    
            #recd_data = req_data['toSend']
        
            logger.info(req_data)
        except:
            logger.debug("Error reading click", exc_info=True)
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])       

    logger.debug(req_data["mtype"])
    
    if(req_data["mtype"]=="train"):
        df = app.config["M"].getTrain()    
    elif(req_data["mtype"]=="test"):
        df = app.config["M"].getTest()    
    elif(req_data["mtype"]=="full"):
        df = app.config["M"].getCombined()    
    else:
        pass
    
    d = {}
    mlist = list(df[req_data["column"]].dropna().unique())
    mlist1 = map(str , mlist)
    d["value"] = ', '.join(mlist1)
    d["title"] = "List of unique values on column"
    logger.info(d)
    return jsonify(d)

@util.route('/generalconfig',methods = ['POST', 'GET'])
def general_config():

    if not app.config["MACTIVE"]:
        logger.error("Error, u have at list generate and select an active session for setting General Configuration")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])       
    else:
        src = app.config["APP_FOLDER"]+"general_config.ini"
        dst = app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_"+"general_config.ini"
        section_grid = "GENERAL"
        if(request.method == 'POST'):
            ## Scrive le modifiche sul file di configurazione x sessione
            result = request.form
            logger.info(result.get('logname'))

            for key, value in result.items():
                app.config["C"].set(section_grid, key, value)
 
            try:
                with open(dst, 'w') as configfile:
                    app.config["C"].write(configfile)
                    logger.info("writing config file done")

                general = app.config["C"][section_grid]
                
                try:
                    return render_template('general_config.html', data=general)
                except Exception as e:
                    return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)


            except:
                logger.debug("Config File write exception", exc_info=True)
                return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
            
        elif(request.method == 'GET'):
            my_file = Path(dst)
            if my_file.is_file():        
                app.config["C"].read(dst)
            else:
                try:
                    copyfile(src, dst)
                    app.config["C"].read(dst)
                except:
                    logger.debug("Copy file error %s, %s", src, dst)

            general = app.config["C"][section_grid]
            try:
                return render_template('general_config.html', data=general) 
            except Exception as e:
                return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)
        else:
            logger.debug("Illegal Method")
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
