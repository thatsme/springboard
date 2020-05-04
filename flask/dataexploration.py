from flask import Blueprint
from flask import render_template
from flask import current_app as app
from flask import request
from werkzeug.local import LocalProxy
from modular.util import Util
from modular.MyPlot import MyPlot

import plotly
import plotly.graph_objs as go
import numpy as np
import json
import pandas as pd

test_plot = MyPlot()

dataexploration = Blueprint('dataexploration', __name__)

logger = LocalProxy(lambda: app.logger)


@dataexploration.route('/dataexploration')
def data_exploration():
    return render_template('data_exploration.html')


@dataexploration.route('/plotlist/<mtype>,<returnpage>')
def plot_list(mtype, returnpage):
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
        return render_template("plot_columns.html", columns=col, rr=returnpage)
    except Exception as e:
        return render_template('tb_implemented.html', version=app.config["DATAPACK"], error=e)

@dataexploration.route('/plots', methods=['GET', 'POST'])
def plots():
    if(request.method == 'POST'):
        df = app.config["M"].getCombined()
        result = request.form
        logger.info(result)
        plot_to_hist_x = []
        plot_to_hist_y = []
        hist = False
        plot_to_scatter_x = []
        plot_to_scatter_y = []
        scatter = False
        plot_to_box_x = []
        plot_to_box_y = []
        box = False
        for key, value in result.items():
            #logger.info(key+" .. "+ value)
            kkey = key.split('_')
            if(kkey[1]=="select"):
                mvalue = value.split("|")
                if(mvalue[0]=="hist"):
                    hist = True
                    if(mvalue[1]=="x"):
                        plot_to_hist_x.append(kkey[0])
                    else:
                        plot_to_hist_y.append(kkey[0])
                elif(mvalue[0]=="scatter"):
                    scatter = True
                    if(mvalue[1]=="x"):
                        plot_to_scatter_x.append(kkey[0])           
                    else:
                        plot_to_scatter_y.append(kkey[0])                                            
                elif(mvalue[0]=="box"):
                    box = True
                    if(mvalue[1]=="x"):
                        plot_to_box_x.append(kkey[0])
                    else:
                        plot_to_box_y.append(kkey[0])
        if(hist):
            test_plot.create_hist(df[plot_to_hist_x+plot_to_hist_y], plot_to_hist_x, plot_to_hist_y)
        elif(scatter):
            test_plot.create_scatter(df[plot_to_scatter_x+plot_to_scatter_y], plot_to_scatter_x, plot_to_scatter_y)
        elif(box):
            test_plot.create_box(df[plot_to_box_x+plot_to_box_y], plot_to_box_x, plot_to_box_y)
            
        logger.info(list(df[plot_to_scatter_x].values.tolist()))
        logger.info(list(df[plot_to_scatter_y].values.tolist()))
            
    try:
        #p = test_plot.testplot()
        p = test_plot.get_plot()
        #p = create_plot("Scatter")
        logger.info(p)
        #return render_template("test_plot.html", columns=col, rr=returnpage)
        return render_template("test_plot.html", rr=result.get("returnpage_layout"), plot = p)
    except Exception as e:
        return render_template('tb_implemented.html',  version=app.config["DATAPACK"], error=e)
