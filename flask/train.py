from flask import Blueprint
from flask import render_template
from flask import current_app as app
from flask import request
from werkzeug.local import LocalProxy
from modular.util import Util

import numpy as np
import json
import pandas as pd

train = Blueprint('train', __name__)

logger = LocalProxy(lambda: app.logger)
SEPARATOR = "____"



@train.route('/train')
def train_model():
    chapter1 = ""
    chapter2 = ""
    return render_template('train_model.html', chapter1=chapter1, chapter2=chapter2)
