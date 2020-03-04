from flask import Blueprint
from flask import render_template
from flask import request
from flask import current_app as app
from werkzeug.local import LocalProxy
from modular.util import Util

sessions = Blueprint('sessions', __name__)

logger = LocalProxy(lambda: app.logger)


@sessions.route('/startsession')
def start_session():
    temp = app.config["CONTEXT"]["session"]
    temp.append(app.config["M"].startsession())
    app.config["CONTEXT"]["session"] = temp
    
    app.config["DATAPACK"]["generatedsession"] = str(app.config["M"].uuid)
    app.config["DATAPACK"]["train_loaded"] = False
    app.config["DATAPACK"]["test_loaded"] = False
    app.config["DATAPACK"]["full_loaded"] = False
    
    #return render_template('session_started.html', version=app.config["M"].__version__)
    logger.debug(app.config["DATAPACK"])
    
    try:
        return render_template('session_started.html', version=app.config["DATAPACK"])
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@sessions.route('/sessionstatus')
def session_status():
    
    try:
        return render_template('session_started.html', version=app.config["DATAPACK"])
    except:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"])

@sessions.route('/sessionreset')
def session_reset():
    
    try:
        return render_template('session_reset.html', version=app.config["DATAPACK"])
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@sessions.route('/setsession', methods=['POST'])
def setsession():
    if request.method == 'POST':
        result = request.form
        # check if session anready exist in list 
        if(result.get('activesession') in app.config["MACTIVE"] ):
            app.config["DATAPACK"]["activesession"] = result.get('activesession')
        else:
            # if not append the session to a list 
            if(len(result.get('activesession'))==36):
                temp = []
                temp_active = []
                temp_active = app.config["MACTIVE"]
                temp = app.config["DATAPACK"]["activesession"]
                temp.append(result.get('activesession'))
                temp_active.append(result.get('activesession'))
                app.config["DATAPACK"]["activesession"] = temp_active[-1]
                app.config["MACTIVE"] = temp_active 
            else:
                logger.debug("You must select a valid session")
                return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
                
        app.config["M"].setsession(app.config["DATAPACK"]["activesession"])
        ## Set plot output as image
        app.config["M"].setPlotToImage(True)
        ## Set output folder 
        app.config["M"].setOutputFolder(app.config["OUTPUT_FOLDER"])
        
        ## After a session is set .. enable the menus 
        app.config["CONTEXT"]["loaddata"] = "dropdown-toggle"
        app.config["CONTEXT"]["loaddata_head"] = "dropdown"
            
    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

    try:
        logger.debug(app.config["DATAPACK"])
        return render_template('session_started.html', version=app.config["DATAPACK"])
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)
