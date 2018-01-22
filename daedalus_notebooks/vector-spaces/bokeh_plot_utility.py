
import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LabelSet, Label, Arrow, OpenHead
from bokeh.plotting import figure, show, output_notebook, output_file

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"
def bokeh_scatter_plot_xy_words(df_words, title='', xlabel='', ylabel='', line_data=False, filename=None, default_color='blue', plot_width=700, plot_height=600):

    plot = bp.figure(plot_width=plot_width, plot_height=plot_height, title=title, tools=TOOLS, toolbar_location="above") #, x_axis_type=None, y_axis_type=None, min_border=1)

    color = 'color' if 'color' in df_words.columns else default_color
    
    plot.scatter(x='x', y='y', size=8, source=df_words, alpha=0.5, color=color)
    
    plot.xaxis[0].axis_label = xlabel
    plot.yaxis[0].axis_label = ylabel

    source = ColumnDataSource(df_words)
    labels = LabelSet(x='x', y='y', text='words', level='glyph',text_font_size="9pt", x_offset=5, y_offset=5, source=source, render_mode='canvas')

    hover = plot.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}
    
    if line_data is True:
        end_arrow = OpenHead(line_color="firebrick", line_width=1, size=10)
        for i in range(1, len(df_words.index), 2):
            x_start, y_start = df_words.iloc[i-1]['x'], df_words.iloc[i-1]['y']
            x_end, y_end = df_words.iloc[i]['x'], df_words.iloc[i]['y']
            plot.add_layout(Arrow(end=end_arrow, x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end))

    plot.add_layout(labels)
    return plot

def bokeh_scatter_plot_xy_words2(df_words, title='', xlabel='', ylabel='', line_data=False):

    plot = bp.figure(plot_width=700, plot_height=600, title=title,
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave", toolbar_location="above") #, x_axis_type=None, y_axis_type=None, min_border=1)

    # plotting. the corresponding word appears when you hover on the data point.
    plot.scatter(x='x', y='y', size=8, source=df_words, alpha=0.5)
    plot.xaxis[0].axis_label = xlabel
    plot.yaxis[0].axis_label = ylabel

    source = ColumnDataSource(df_words)
    labels = LabelSet(x='x', y='y', text='words', level='glyph',text_font_size="9pt",
                      x_offset=5, y_offset=5, source=source, render_mode='canvas')

    hover = plot.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}
    # df.transform(lambda x: list(zip(x, df[2])))
    if line_data is True:
        end_arrow = OpenHead(line_color="firebrick", line_width=1, size=10)
        for i in range(1, len(df_words.index), 2):
            x_start, y_start = df_words.iloc[i-1]['x'], df_words.iloc[i-1]['y']
            x_end, y_end = df_words.iloc[i]['x'], df_words.iloc[i]['y']
            plot.add_layout(Arrow(end=end_arrow, x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end))

    plot.add_layout(labels)
    show(plot, notebook_handle=True)
