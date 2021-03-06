
#logging.debug('message')
#logging.info('message')
#logging.warn('message')
#logging.error('message')
#logging.critical('message')

import re 
import os
from shutil import copyfile
import logging
import yaml
import pandas as pd

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
from dataexploration import dataexploration
from featureengineering import featureengineering
from train import train
from predict import predict
from pipelines import pipelines
from sessions import sessions
from files import files
from util import util

## BLUEPRINTS REGISTRATION
app.register_blueprint(datawrangling)
app.register_blueprint(dataexploration)
app.register_blueprint(featureengineering)
app.register_blueprint(train)
app.register_blueprint(predict)
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
    #return render_template('section_list.html', your_list=methodlist)
    return render_template('home.html', your_list=methodlist)

@app.route('/dummy')
def dummy():
    return render_template('tb_implemented.html', version=app.config["DATAPACK"])
    
@app.route('/splitdata/<mkey>')
def split_data(mkey):
    return render_template('tb_implemented.html')
    
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