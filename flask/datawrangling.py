from flask import Blueprint
from flask import render_template
from flask import current_app as app
from werkzeug.local import LocalProxy
from modular.util import Util

datawrangling = Blueprint('datawrangling', __name__)

logger = LocalProxy(lambda: app.logger)


@datawrangling.route('/datawrangling')
def data_wrangling():
    return render_template('data_wrangling.html')


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
