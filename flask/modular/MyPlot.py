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
        title = "BAR Plot"
        self.my_x_columns = my_x_columns
        self.my_y_columns = my_y_columns
        data = [
            go.Bar(
                x = df[''.join(my_x_columns)],
                y = df[''.join(my_y_columns)]
                
            ),
            go.Layout(
                title=title,
                xaxis=dict(
                    title='X axis',
                    type='log',
                    autorange=True,
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=15,
                        color='#7f7f7f'
                    )
                ),
                yaxis=dict(
                    title='Y axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=15,
                        color='#7f7f7f'
                    )
                ),
                hovermode="closest"
            )
        ]


        self.graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    
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

        #fig.add_trace(go.Scatter(
        #    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        #    y=[1, 0, 3, 2, 5, 4, 7, 6, 8],
        #    name="Name of Trace 2"
        #))

        fig.update_layout(
            title="Plot Title",
            xaxis_title=''.join(my_x_columns),
            yaxis_title='/'.join(my_y_columns),
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )
        )        

        self.graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_box(self, my_x_columns, my_y_columns):
        pass
    
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
    
    
    