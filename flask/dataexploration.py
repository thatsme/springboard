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
SEPARATOR = "____"

@dataexploration.route('/dataexploration')
def data_exploration():
    chapter1 = "Data exploration is an approach similar to initial data analysis, whereby a data analyst uses visual exploration to understand what is in a dataset and the characteristics of the data, rather than through traditional data management systems. These characteristics can include size or amount of data, completeness of the data, correctness of the data, possible relationships amongst data elements or files/tables in the data."
    chapter2 = "Data exploration is typically conducted using a combination of automated and manual activities. Automated activities can include data profiling or data visualization or tabular reports to give the analyst an initial view into the data and an understanding of key characteristics."
    return render_template('data_exploration.html', chapter1=chapter1, chapter2=chapter2)


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
        groupby = result.get("group"+SEPARATOR+"by")
        typeofplot = result.get("type"+SEPARATOR+"plot")
        logger.info(result)
        plot_x = []
        plot_y = []
        plot_size = []
        plot_hover = []
        plot_to_hist_x = []
        plot_to_hist_y = []
        hist = False
        plot_to_scatter_x = []
        plot_to_scatter_y = []
        plot_to_scatter_c = []
        scatter = False
        plot_to_box_x = []
        plot_to_box_y = []
        box = False
        for key, value in result.items():
            logger.info(key+" .. "+ value)
            kkey = key.split(SEPARATOR)
            if(kkey[1]=="select"):
                mvalue = value.split("|")
                if(mvalue[0]=="hist"):
                    typeofplot = "bar"
                    if(mvalue[1]=="x"):
                        plot_to_hist_x.append(kkey[0])
                    elif(mvalue[1]=="y"):
                        plot_to_hist_y.append(kkey[0])
                elif(mvalue[0]=="scatter"):
                    typeofplot = "scatter"
                    if(mvalue[1]=="x"):
                        plot_to_scatter_x.append(kkey[0])           
                    elif(mvalue[1]=="y"):
                        plot_to_scatter_y.append(kkey[0])                                            
                    elif(mvalue[1]=="c"):
                        plot_to_scatter_c.append(kkey[0])                                            
                elif(mvalue[0]=="box"):
                    typeofplot = "box"
                    if(mvalue[1]=="x"):
                        plot_to_box_x.append(kkey[0])
                    else:
                        plot_to_box_y.append(kkey[0])
            elif(kkey[1]=="checkx"):
                if(value=="on"):
                    plot_x.append(kkey[0])
            elif(kkey[1]=="checky"):
                if(value=="on"):
                    plot_y.append(kkey[0])
            elif(kkey[1]=="checksize"):
                if(value=="on"):
                    plot_size.append(kkey[0])
            elif(kkey[1]=="checkhover"):
                if(value=="on"):
                    plot_hover.append(kkey[0])
        if(typeofplot=="bar"):
            test_plot.create_hist(df, plot_x, plot_y, groupby)
        elif(typeofplot=="scatter"):
            test_plot.create_scatter(df, plot_x, plot_y, plot_size, plot_hover, groupby)
        elif(typeofplot=="box"):
            test_plot.create_box(df, plot_x, plot_y, groupby)
                        
    try:
        #p = test_plot.testplot()
        p = test_plot.get_plot()
        #p = create_plot("Scatter")
        logger.info(p)
        #return render_template("test_plot.html", columns=col, rr=returnpage)
        return render_template("test_plot.html", rr=result.get("returnpage"+SEPARATOR+"layout"), plot = p)
    except Exception as e:
        return render_template('tb_implemented.html',  version=app.config["DATAPACK"], error=e)
