import re 
import os
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

ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = '/app/static/input'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#from flask_ext.navigation import Navigation

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from app import views
from app import ensamble
import configparser as cp 


config = cp.ConfigParser()
config.read('MlUtil.ini')

  
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

    return render_template('file_list.html', your_list=list_of_files)

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

    df = pd.read_csv("/app/static/input/"+key)
    # link_column is the column that I want to add a button to
    return render_template("test_pandas.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                           link_column="PassengerId", zip=zip)

@app.route('/listcolumns/<key>')
def list_columns(key):

    df = pd.read_csv("/app/static/input/"+key)

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

        return redirect(url_for('ensamble'))
    return render_template('ensamble.html', form=form)

@app.route('/about-us/')
def about():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')