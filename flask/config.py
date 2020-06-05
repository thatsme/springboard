from modular.MlUtil import MlUtil 
import configparser as cp 

class Config(object):
    DEBUG = False
    TESTING = False
  
class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):

    DEBUG = True
    SECRET_KEY = '7d441f27d441f27567d441f2b6176a'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = set(['csv','txt','yaml','png','jpg','jpeg','gif'])
    APP_FOLDER = '/app/'
    UPLOAD_FOLDER = '/app/static/input/'
    INPUT_FOLDER = '/app/static/input/'
    OUTPUT_FOLDER = 'static/output/'
    LOG_FOLDER = '/app/log/'
    DEFAULT_ERRORMESSAGE = "Something wrong, Check error log"
    ENABLED_TYPES = ["train", "test", "full", "freecontent"]
    ENABLED_MASTERS = ["_base.html", "_dataw.html", "_datae.html", "_featuree.html"]
    DATAPACK = {"activesession" : [], "test": "", "train": "", "full": "", "test_loaded" : False, "train_loaded" : False, "full_loaded" : False}
    CONTEXT = {"session" : ["Select a session"], 
               "config" : "dropdown-toggle", 
               "config_head" : "dropdown", 
               "loaddata" : "disabled", 
               "loaddata_head" : "disabled", 
               "datawrangling" : "disabled", 
               "datawrangling_head" : "disabled", 
               "dataexploration" : "disabled", 
               "dataexploration_head" : "disabled",
               "featureengineering" : "disabled",
               "featureengineering_head" : "disabled", 
               "fileoutput" : "disabled", 
               "fileoutput_head" : "disabled",
               "ensamble" : "disabled", 
               "ensamble_head" : "disabled", 
               "show_train" : "hide",
               "show_test" : "hide",
               "show_full" : "hide"
    }
    MACTIVE = []
    M = MlUtil()
    C = cp.ConfigParser()
    
class TestingConfig(Config):
    TESTING = True