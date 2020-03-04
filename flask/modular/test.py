from flask import Blueprint

test_blueprint = Blueprint('test_blueprint', __name__)

p = Pippo()

class Pippo(object):
    def __init__(self):
        p = "ciao"