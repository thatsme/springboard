from flask import current_app as app
from shutil import copyfile
from werkzeug.local import LocalProxy

logger = LocalProxy(lambda: app.logger)

class Util():
    @staticmethod
    def CheckTypes(type):
        if(type in app.config["ENABLED_TYPES"]):
            return True
        return False
    
    @staticmethod
    def CheckMasters(type):
        if(type in app.config["ENABLED_MASTERS"]):
            return True
        return False
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

    @staticmethod
    def MoveUploadedFiles(src, destpath, dst):
        if(app.config["DATAPACK"]["activesession"]):
            msrc = app.config["UPLOAD_FOLDER"]+src
            mdst = destpath+app.config["DATAPACK"]["activesession"]+"_"+dst
            try:
                copyfile(msrc,mdst)
                return True
            except:
                logger.debug("Copy file error %s, %s", msrc, mdst)
                return False

    @staticmethod
    def Filter(mstring, msubstr, flag): 
        if flag:
            return [str for str in mstring if not any(sub in str for sub in msubstr)]
        else:
            return [str for str in mstring if any(sub in str for sub in msubstr)]

    @staticmethod
    def ResetDatapack():
        ## Reset previouse values
        app.config["DATAPACK"]["test"] = ""
        app.config["DATAPACK"]["train"] = ""
        app.config["DATAPACK"]["full"] = ""
        app.config["DATAPACK"]["test_loaded"] = False
        app.config["DATAPACK"]["train_loaded"] = False
        app.config["DATAPACK"]["full_loaded"] = False
        app.config["DATAPACK"]["column_list"] = []
