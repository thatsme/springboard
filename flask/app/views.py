from app import app
from flask import render_template

@app.route('/')
def home():
   return "hello world!"

@app.route('/test')
def test():
    return "this is a test section"

@app.route('/template')
def template():
    return render_template('home.html')