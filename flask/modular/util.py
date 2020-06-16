from flask import current_app as app
from shutil import copyfile
from werkzeug.local import LocalProxy
from io import StringIO
from flask import render_template
import pandas as pd 

logger = LocalProxy(lambda: app.logger)

class Util():
    
    @staticmethod
    def getCheckFromForm(r, postfix):
        col = []
        act = []
        filtered_dict = {k:v for (k,v) in r.items() if postfix in k}
        logger.debug(filtered_dict)
        for k,v in filtered_dict.items():
            kk = k.split("_")
            col.append(kk[0])
            act.append(kk[1])
        
        return col, act
    
    
    @staticmethod
    def fixPath(mpath):
        if(mpath[-1] != '/'):
            return mpath+'/'
        else:
            return mpath

    @staticmethod
    def extractCategoryFromPath(mpath):
        return mpath.split('/')[-2]
    
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
        app.config["DATAPACK"]["directory"] = ""
        
        app.config["DATAPACK"]["test_loaded"] = False
        app.config["DATAPACK"]["train_loaded"] = False
        app.config["DATAPACK"]["full_loaded"] = False
        app.config["DATAPACK"]["directory_loaded"] = False
        app.config["DATAPACK"]["column_list"] = []
        
        

    @staticmethod
    def exportDf(df, mtype):
        try:
            df.to_csv(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+"_head.txt", index=True, sep= ' ', mode='a')        
        except IOError as e:
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=e)   
        except :
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

    @staticmethod
    def writeDescribe(df, mtype):
        data = df.describe()
        Util.exportDf(data, mtype)

    @staticmethod
    def writeInfo(df, mtype):
        
        ## Get info data on full and copy on session prefixed text file  
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        try:
            with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+"info.txt", "w", encoding="utf-8") as f:  
                f.write(s)
        except IOError as e:
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=e)   
        except :
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
            
    @staticmethod
    def writeDtypes(df, mtype):
        buffer = StringIO()
        df.dtypes.to_string(buf=buffer)
        #logger.info(df_train.dtypes.to_dict())
        #logger.info(df_train.dtypes.tolist())
        s = buffer.getvalue()
        try:
            with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+"dtypes.txt", "w", encoding="utf-8") as f:  
                f.write(s)
        except IOError as e:
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=e)  
        except:
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

    @staticmethod
    def writeNunique(df, mtype):
        buffer = StringIO()
        na = df.isna().sum().to_frame(name='null')
        un = df.nunique().to_frame(name='unique')
        result = pd.concat([na, un], axis=1, sort=False)
        result.to_string(buf=buffer)
        
        s = buffer.getvalue()
        try:
            with open(app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+"unna.txt", "w", encoding="utf-8") as f:  
                f.write(s)
        except IOError as e:
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=e)  
        except:
            logger.debug("File write exception", exc_info=True)
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
