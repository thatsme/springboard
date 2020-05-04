import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Layout
import pandas as pd
import numpy as np
import json
import plotly.express as px

class MyPlot():
    def __init__(self, train=None, test=None, vb=0, model=None):
        self.graphJSON = None
        self.my_x_columns = None
        self.my_y_columns = None
    
    def get_plot(self):    
        return self.graphJSON
    
    def create_hist(self, df,  my_x_columns, my_y_columns, groupby):
        mtitle = "BAR Plot"
        self.my_x_columns = my_x_columns
        self.my_y_columns = my_y_columns
        fig = go.Figure()

        if(groupby=="y"):
            for yvalue in my_y_columns:
                fig.add_trace(go.Bar(
                    x=df[''.join(my_x_columns)],
                    y=df[''.join(yvalue)],
                    name="Trace "+''.join(yvalue),       # this sets its legend entry
                ))
        else:
            for xvalue in my_x_columns:
                fig.add_trace(go.Bar(
                    x=df[''.join(xvalue)],
                    y=df[''.join(my_y_columns)],
                    name="Trace "+''.join(yvalue),       # this sets its legend entry
                ))
            
        fig.update_layout(
            title=mtitle,
            xaxis_title='/'.join(my_x_columns),
            yaxis_title='/'.join(my_y_columns),
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )
        )        


        self.graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_scatter(self, df, my_x_columns, my_y_columns, my_size_columns, my_hover_columns, groupby):
        mtitle = "SCATTER Plot"
        self.my_x_columns = my_x_columns
        self.my_y_columns = my_y_columns
        test = False
        hover_text = []
        mksize = df[''.join(my_size_columns)] if (len(my_size_columns)>0) else None
        data = ''
        for index, row in df.iterrows():
            for col in my_hover_columns:
                data += row[col]+'<br>'
            hover_text.append(('{data}<br>').format(data=data))
            
        if(test):
            fig = px.scatter(df, my_x_columns, my_y_columns )
        else:
            fig = go.Figure()
            
            if(groupby=="y"):
                for yvalue in my_y_columns:
                    fig.add_trace(go.Scatter(
                        x=df[''.join(my_x_columns)],
                        y=df[''.join(yvalue)],
                        name="Trace "+''.join(yvalue),       # this sets its legend entry
                        mode = 'markers',
                        marker_size = mksize,
                        text=hover_text,
                    ))
            else:
                for xvalue in my_x_columns:
                    fig.add_trace(go.Scatter(
                        x=df[''.join(xvalue)],
                        y=df[''.join(my_y_columns)],
                        #fillcolor=df[''.join(my_c_columns)],
                        name="Trace "+''.join(xvalue),       # this sets its legend entry
                        mode = 'markers',
                        marker_size = df[''.join(my_size_columns)]
                    ))

            fig.update_layout(
                title=mtitle,
                xaxis_title='/'.join(my_x_columns),
                yaxis_title='/'.join(my_y_columns),
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                    color="#7f7f7f"
                )
            )        

        self.graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_box(self, df, my_x_columns, my_y_columns, groupby):
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
    
    
    