from flask import Blueprint
from flask import render_template
from flask import current_app as app
from flask import request
from werkzeug.local import LocalProxy
from modular.util import Util
from modular.MyPlot import MyPlot

import numpy as np
import json
import pandas as pd

featureengineering = Blueprint('featureengineering', __name__)

logger = LocalProxy(lambda: app.logger)
SEPARATOR = "____"



@featureengineering.route('/featureengineering')
def feature_engineering():
    chapter1 = "Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. Feature engineering can be considered as applied machine learning itself."
    chapter2 = "A feature is an attribute or property shared by all of the independent units on which analysis or prediction is to be done. Any attribute could be a feature, as long as it is useful to the model. The purpose of a feature, other than being an attribute, would be much easier to understand in the context of a problem. A feature is a characteristic that might help when solving the problem."
    return render_template('feature_engineering.html', chapter1=chapter1, chapter2=chapter2)
