from flask import Blueprint
from flask import render_template
from flask import request
from flask import current_app as app
from werkzeug.local import LocalProxy
from modular.util import Util

DW_pipeline = []
DW_content = {}

pipelines = Blueprint('pipelines', __name__)

logger = LocalProxy(lambda: app.logger)


def DummyDWContent():
    DW_content = {}
    DW_content["colname"] = "colname"
    DW_content["newname"] = ""
    DW_content["prefix"] = "new_"
    DW_content["action"] = "toint"
    DW_content["actioname"] = "To Integer"
    
    DW_pipeline.append(DW_content)

@pipelines.route('/loadpipeline/<returnpage>', methods=['GET', 'POST'])
def load_pipeline(returnpage):
    pass

@pipelines.route('/createpipeline/<mtype>, <returnpage>', methods=['GET', 'POST'])
def create_pipeline(mtype, returnpage):
    if(Util.CheckTypes(mtype) and Util.CheckMasters(returnpage)):
                        
        if(request.method == 'GET'):
            #DummyDWContent()
            stuff = {}
            colnames = app.config["M"].getColumns(mtype)
            dtypes = app.config["M"].getDtypes(mtype)
            dtypes_list = [x.name for x in dtypes]
            #colnames_list = [x.ljust(15,'_') for x in colnames]
            
            
            select_actions = []
            select_actions.append({'name':'drop', 'value':'Drop'})
            select_actions.append({'name':'toint', 'value':'To Integer'})
            select_actions.append({'name':'tostring', 'value':'To String'})
            select_actions.append({'name':'hotencode', 'value':'Hot Encode (sklearn)'})
            select_actions.append({'name':'dummies', 'value':'Dummies (pandas)'})
            select_actions.append({'name':'map', 'value':'Map/reduction w Dictionaries'})
            stuff["actions"] = select_actions
            stuff["colnames"] = colnames
            stuff["dtypes"] = dtypes_list
            stuff["mtype"] = mtype
                    
            try:
                return render_template('show_pipelines.html',rr=returnpage, pipelines=DW_pipeline, stuff=stuff, zip=zip)
            except Exception as e:
                return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

        else:
            logger.debug("Illegal Method")
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
    else:
        logger.debug("GET proibited parameters")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   


@pipelines.route('/updatepipeline', methods=['POST'])
def update_pipeline():
    if(request.method == 'POST'):
        result = request.form
        tempcolname = []
        tempaction = []
        vvalue = {}
        DW_content = {}
        for r in result.getlist('colname'):
            tempcolname.append(r)
            #logger.info("fileselect -> *"+r+"*")

        for r in result.getlist('action'):
            tempaction.append(r)
            #logger.info("fileselect -> *"+r+"*")

        logger.info(tempcolname)
        logger.info(tempaction)
        
        for key, value in result.items():
            logger.info("--> key *"+key+"* --> value *"+value+"*")
            #kkey = key.split("_")[0]
            if value in tempcolname:
                logger.info("vvalue value -> *"+value+"* vvalue key -> *"+key+"*"+key)    
                DW_content[key] = value
                        
            if value in tempaction:
                logger.info("vvalue value -> *"+value+"* vvalue key -> *"+key+"*"+key)            
                DW_content[key] = value
                #DW_content["actionname"] = 

        DW_pipeline.append(DW_content)
        #logger.info(DW_pipeline)
        #logger.info(result.get("mtype"))
        #logger.info(result.get("returnpage"))
        
        stuff = {}
        colnames = app.config["M"].getColumns(result.get("mtype"))
        dtypes = app.config["M"].getDtypes(result.get("mtype"))
        dtypes_list = [x.name for x in dtypes]
        colnames_list = [x.ljust(10,' ') for x in colnames]
                    
        select_actions = []
        select_actions.append({'name':'drop', 'value':'Drop'})
        select_actions.append({'name':'toint', 'value':'To Integer'})
        select_actions.append({'name':'tofloat', 'value':'To Float'})
        select_actions.append({'name':'tostring', 'value':'To String'})
        select_actions.append({'name':'hotencode', 'value':'Hot Encode (sklearn)'})
        select_actions.append({'name':'dummies', 'value':'Dummies (pandas)'})
        select_actions.append({'name':'map', 'value':'Map/reduction w Dictionaries'})

        stuff["actions"] = select_actions
        stuff["colnames"] = colnames_list
        stuff["dtypes"] = dtypes_list
        stuff["mtype"] = result.get("mtype")

        
        try:
            return render_template('show_pipelines.html',rr=result.get("returnpage"), pipelines=DW_pipeline, stuff=stuff, zip=zip)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   

@pipelines.route('/runpipeline', methods=['POST'])
def run_pipeline():
    if(request.method == 'POST'):
        result = request.form
        tempcolname = []
        tempaction = []
        vvalue = {}
        DW_content = {}
        dummy_work = []
        dummy_prefix = []
        drop_work = []
        toint_work = []
        tofloat_work = []
        tostring_work = []
        map_work = []
        logger.info("==== runpipeline =========================================================")
        
        ## Zip object with columns and action 
        column, action = Util.getCheckFromForm(result, "check")

        # The method accept two list of columns and prefixes .. so we have to change that
        #
        for col, act in zip(column, action):
            logger.info(col)
            if(act=="dummies"):
                dummy_work.append(col)
                dummy_prefix.append(col)           
            elif(act=="drop"):
                drop_work.append(col)
            elif(act=="toint"):
                toint_work.append(col)
            elif(act=="tofloat"):
                tofloat_work.append(col)
            elif(act=="tostring"):
                tostring_work.append(col)
            elif(act=="map"):
                map_work.append(col)
            else:
                logger.debug("Action not implemented yet")
        
        done = app.config["M"].transform_dummies(dummy_work, dummy_prefix)
        logger.info(f"Dummies done {done}")

        done = app.config["M"].transform_int(toint_work, "int")
        logger.info(f"To int done {done}")

        done = app.config["M"].transform_float(tofloat_work, "float")
        logger.info(f"To float done {done}")

        #done = app.config["M"].transform_str(tostr_work)
        #logger.info(f"To str done {done}")

        ## Drop pipeline always at end         
        done = app.config["M"].transform_drop(drop_work)
        logger.info(f"Drop done {done}")

        ## Reload all statistics 
        # Copy back the dataframe and generate some statistics 
        if(result.get("mtype")=="full"):
            df = app.config["M"].getCombined()
        elif(result.get("mtype")=="train"):
            df = app.config["M"].getTrain()
        elif(result.get("mtype")=="test"):
            df = app.config["M"].getTest()
                
        ## Get info data on full dataset and copy on session prefixed text file  
        Util.writeInfo(df, result.get("mtype")) 
                            
        # Get dtypes data on full dataset and copy on session prefixed text file
        Util.writeDtypes(df, result.get("mtype"))
        
        # Get nunique and nan data on full dataset and copy on session prefixed text file
        Util.writeNunique(df, result.get("mtype"))

        #logger.info(DW_pipeline)
        #logger.info(result.get("mtype"))
        #logger.info(result.get("returnpage"))
        
        stuff = {}
        colnames = app.config["M"].getColumns(result.get("mtype"))
        dtypes = app.config["M"].getDtypes(result.get("mtype"))
        dtypes_list = [x.name for x in dtypes]
        colnames_list = [x.ljust(10,' ') for x in colnames]
                    
        select_actions = []
        select_actions.append({'name':'drop', 'value':'Drop'})
        select_actions.append({'name':'toint', 'value':'To Integer'})
        select_actions.append({'name':'tofloat', 'value':'To Float'})
        select_actions.append({'name':'tostring', 'value':'To String'})
        select_actions.append({'name':'hotencode', 'value':'Hot Encode (sklearn)'})
        select_actions.append({'name':'dummies', 'value':'Dummies (pandas)'})
        select_actions.append({'name':'map', 'value':'Map/reduction w Dictionaries'})
        stuff["actions"] = select_actions
        stuff["colnames"] = colnames_list
        stuff["dtypes"] = dtypes_list
        stuff["mtype"] = result.get("mtype")

        
        try:
            return render_template('show_pipelines.html',rr=result.get("returnpage"), pipelines=DW_pipeline, stuff=stuff, zip=zip)
        except Exception as e:
            return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

    else:
        logger.debug("Illegal Method")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   



@pipelines.route('/preparepipeline/<mtype>, <returnpage>', methods=['GET', 'POST'])
def prepare_pipeline(mtype, returnpage):
    if(Util.CheckTypes(mtype) and Util.CheckMasters(returnpage)):
                        
        if(request.method == 'GET'):
            #DummyDWContent()
            stuff = {}
            colnames = app.config["M"].getColumns(mtype)
            dtypes = app.config["M"].getDtypes(mtype)
            dtypes_list = [x.name for x in dtypes]
            #colnames_list = [x.ljust(15,'_') for x in colnames]
            
            
            select_actions = []
            select_actions.append({'name':'drop', 'value':'Drop'})
            select_actions.append({'name':'toint', 'value':'To Integer'})
            select_actions.append({'name':'tofloat', 'value':'To Float'})
            select_actions.append({'name':'tostring', 'value':'To String'})
            select_actions.append({'name':'hotencode', 'value':'Hot Encode (sklearn)'})
            select_actions.append({'name':'dummies', 'value':'Dummies (pandas)'})
            select_actions.append({'name':'map', 'value':'Map/reduction w Dictionaries'})
            stuff["actions"] = select_actions
            stuff["colnames"] = colnames
            stuff["dtypes"] = dtypes_list
            stuff["mtype"] = mtype
                    
            try:
                return render_template('prepare_pipelines.html',rr=returnpage, pipelines=DW_pipeline, stuff=stuff, zip=zip)
            except Exception as e:
                return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

        else:
            logger.debug("Illegal Method")
            return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
    else:
        logger.debug("GET proibited parameters")
        return render_template('show_error.html', content=app.config["DEFAULT_ERRORMESSAGE"])   
