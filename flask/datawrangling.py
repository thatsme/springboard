import os
from flask import Blueprint
from flask import render_template
from flask import current_app as app
from werkzeug.local import LocalProxy
from modular.util import Util
import cv2

datawrangling = Blueprint('datawrangling', __name__)

logger = LocalProxy(lambda: app.logger)


@datawrangling.route('/datawrangling')
def data_wrangling():
    chapter1 = "Data wrangling, sometimes referred to as data munging, is the process of transforming and mapping data from one <b>raw</b> data form into another format with the intent of making it more appropriate and valuable for a variety of downstream purposes such as analytics. A data wrangler is a person who performs these transformation operations."
    chapter2 = "The data transformations are typically applied to distinct entities (e.g. fields, rows, columns, data values etc.) within a data set, and could include such actions as extractions, parsing, joining, standardizing, augmenting, cleansing, consolidating and filtering to create desired wrangling outputs that can be leveraged downstream."
    return render_template('data_wrangling.html', chapter1=chapter1, chapter2=chapter2)


@datawrangling.route('/showtext/<mkey>, <mtype>, <returnpage>, <action>')
def show_text(mkey, mtype, returnpage, action):
    if(Util.CheckTypes(mtype) and Util.CheckMasters(returnpage)):
        if(mkey=="info"):
            filename = app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+mkey+".txt"
        elif(mkey=="dtypes"):
            filename = app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+mkey+".txt"
        elif(mkey=="describe"):
            filename = app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+mkey+".txt"
        elif(mkey=="unna"):
            filename = app.config["OUTPUT_FOLDER"]+app.config["DATAPACK"]["activesession"]+"_df_"+mtype+mkey+".txt"
        else:
            if(mtype=='freecontent'):
                filename = app.config["OUTPUT_FOLDER"]+mkey
            else:

                logger.debug("Missing key value")
                return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

        try:
            logger.info("File to open %s %s ", filename, mkey)
            with open(filename, "r") as f:
                content = f.read()
        except:
            logger.debug("File read exception", exc_info=True)
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

        logger.debug(returnpage)

        try:        
            return(render_template('show_text.html', content=content, rr=returnpage))
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("GET proibited parameters")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

@datawrangling.route('/showimages/<mkey>, <mtype>, <returnpage>, <action>')
def show_images(mkey, mtype, returnpage, action):
    
    start_directory = Util.fixPath(app.config["DATAPACK"]["directory"])
    
    list_of_dir = []
    for filename in os.listdir(start_directory):
        if os.path.isdir(start_directory+filename):
            list_of_dir.append(start_directory+filename+'/')
                
    list_of_files = []
    list_of_categories = []
    list_of_image_h = []
    list_of_image_w = []
    list_of_image_c = []
    
    for directory in list_of_dir:
        for filename in os.listdir(directory):
            if os.path.isfile(directory+filename):
                ff, file_extension = os.path.splitext(directory+filename)
                list_of_files.append(directory[4:]+filename)
                #list_of_categories.append(directory)
                list_of_categories.append(Util.extractCategoryFromPath(directory))
                try:
                    im = cv2.imread(directory+filename)
                    h, w, c = im.shape
                except Exception as e:
                    return False
                    
                list_of_image_h.append(h)
                list_of_image_w.append(w)
                list_of_image_c.append(c)

    logger.info("Ended with not errors, maybe")
    logger.info(len(list_of_categories))    
    logger.info(len(list_of_files))    
    try:        
        return(render_template('show_images.html', categories=list_of_categories, files=list_of_files, h = list_of_image_h, w=list_of_image_w, rr=returnpage, zip=zip, enumerate=enumerate))
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@datawrangling.route('/createmask/<mkey>, <mtype>, <returnpage>, <action>')
def create_mask(mkey, mtype, returnpage, action):
    pass

@datawrangling.route('/showdescribe/<mkey>,<mtype>, <returnpage>')
def show_describe(mkey, mtype, returnpage):
    '''
    Fixing DataFrame.describe visualization for dataframe to list 
    plus index column as standard column
    '''
    
    df = app.config["M"].getDescribe(mtype)
        
    df.round(3)
    cnames = df.columns.values.tolist()
    cnames.insert(0, "value")
    logger.info(df.info)
    dvalues = df.values.tolist()
    ivalues = df.index.values.tolist()
    final = []
    for c in dvalues:
        logger.info(c)
        for p in ivalues:
            #logger.info(p)
            c.insert(0, p)
            #logger.info(c)
            final.append(c)
            #logger.info(final)
            ivalues.pop(0)
            break

    rdata = list(final)

    try:
        return render_template("show_describe.html", column_names=cnames, link_column="Index", row_data=rdata, zip=zip, rr=returnpage)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)


@datawrangling.route('/columnlist/<mtype>,<returnpage>')
def column_list(mtype, returnpage):
    if(mtype=="train"):
        df = app.config["M"].getTrain()
        cnames = df.columns.values.tolist()
        ctypes = df.dtypes.tolist()
    elif(mtype=="test"):
        df = app.config["M"].getTest()
        cnames = df.columns.values.tolist()
        ctypes = df.dtypes.tolist()
    elif(mtype=="full"):
        df = app.config["M"].getCombined()
        cnames = df.columns.values.tolist()
        ctypes = df.dtypes.tolist()
    else:
        logger.debug("Wrong dataframe type %s", mtype)
        return("")

    try:
        col = zip(cnames,ctypes)
        logger.info(col)
        return render_template("manage_columns.html", columns=col, rr=returnpage)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)


@datawrangling.route('/showdataframe/<mtype>, <where>, <num>, <returnpage>')
def show_dataframe(mtype, where, num, returnpage):
    # Da rivedere completamente
    if(mtype=="train"):
        df = app.config["M"].getTrain()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    elif(mtype=="test"):
        df = app.config["M"].getTest()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    elif(mtype=="full"):
        df = app.config["M"].getCombined()
        cnames = df.columns.values
        if(where=="head"):
            rdata = list(df.head(int(num)).values.tolist())
        else:
            rdata = list(df.tail(int(num)).values.tolist())

    else:
        logger.debug("Wrong dataframe type %s", mtype)
        return("")

    # link_column is the column that I want to add a button to
    try:
        return render_template("show_df.html", column_names=cnames, row_data=rdata,
                           link_column="PassengerId", zip=zip, rr=returnpage, mtype=mtype)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)
