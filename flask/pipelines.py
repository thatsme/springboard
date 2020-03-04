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
        select_actions.append({'name':'tostring', 'value':'To String'})
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

@pipelines.route('/runepipeline', methods=['POST'])
def run_pipeline():
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
        select_actions.append({'name':'tostring', 'value':'To String'})
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
            select_actions.append({'name':'tostring', 'value':'To String'})
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
