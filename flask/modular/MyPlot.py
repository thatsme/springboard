import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Layout
import pandas as pd
import numpy as np
import json


class MyPlot():
    def __init__(self, train=None, test=None, vb=0, model=None):
        self.graphJSON = None
        self.my_x_columns = None
        self.my_y_columns = None
    
    def get_plot(self):    
        return self.graphJSON
    
    def create_hist(self, df,  my_x_columns, my_y_columns):
        mtitle = "BAR Plot"
        self.my_x_columns = my_x_columns
        self.my_y_columns = my_y_columns
        fig = go.Figure()

        for yvalue in my_y_columns:
            fig.add_trace(go.Bar(
                x=df[''.join(my_x_columns)],
                y=df[''.join(yvalue)],
                name="Trace "+''.join(yvalue),       # this sets its legend entry
            ))

        fig.update_layout(
            title=mtitle,
            xaxis_title=''.join(my_x_columns),
            yaxis_title='/'.join(my_y_columns),
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )
        )        


        self.graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_scatter(self, df, my_x_columns, my_y_columns):
        mtitle = "SCATTER Plot"
        self.my_x_columns = my_x_columns
        self.my_y_columns = my_y_columns
        
        fig = go.Figure()

        for yvalue in my_y_columns:
            fig.add_trace(go.Scatter(
                x=df[''.join(my_x_columns)],
                y=df[''.join(yvalue)],
                name="Trace "+''.join(yvalue),       # this sets its legend entry
                mode = 'markers'
            ))

        fig.update_layout(
            title=mtitle,
            xaxis_title=''.join(my_x_columns),
            yaxis_title='/'.join(my_y_columns),
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )
        )        

        self.graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_box(self, df, my_x_columns, my_y_columns):
        mtitle = "BOX Plot"
        self.my_x_columns = my_x_columns
        self.my_y_columns = my_y_columns

        fig = go.Figure()

        for yvalue in my_y_columns:
            fig.add_trace(go.Box(
                x=df[''.join(my_x_columns)],
                y=df[''.join(yvalue)],
                name="Trace "+''.join(yvalue),       # this sets its legend entry
            ))

        fig.update_layout(
            title=mtitle,
            xaxis_title=''.join(my_x_columns),
            yaxis_title='/'.join(my_y_columns),
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )
        )        
        self.graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    def testplot(self):
        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


        data = [
            go.Bar(
                x=df['x'], # assign x as the dataframe column 'x'
                y=df['y']
            )
        ]

        graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON
    
    
    