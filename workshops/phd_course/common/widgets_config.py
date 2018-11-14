# from __future__ import print_function
import ipywidgets as widgets

# if __package__:
#    print('Package named {!r}; __name__ is {!r}'.format(__package__, __name__))

from common import extend
import common.config as config

def kwargser(d):
    args = dict(d)
    if 'kwargs' in args:
        kwargs = args['kwargs']
        del args['kwargs']
        args.update(kwargs)
    return args

# FIXME Keep project specific stuff in widget_config", move generic to "widget_utility"

def toggle(description, value, **kwargs):  # pylint: disable=W0613
    return widgets.ToggleButton(**kwargser(locals()))

def toggles(description, options, value, **kwopts):  # pylint: disable=W0613
    return widgets.ToggleButtons(**kwargser(locals()))

def dropdown(description, options, value, **kwargs):  # pylint: disable=W0613
    return widgets.Dropdown(**kwargser(locals()))

def slider(description, min, max, value, **kwargs):  # pylint: disable=W0613, W0622
    return widgets.IntSlider(**kwargser(locals()))

def rangeslider(description, min, max, value, **kwargs):  # pylint: disable=W0613, W0622
    return widgets.IntRangeSlider(**kwargser(locals()))

def sliderf(description, min, max, step, value, **kwargs):  # pylint: disable=W0613, W0622
    return widgets.FloatSlider(**kwargser(locals()))

def progress(min, max, step, value, **kwargs):  # pylint: disable=W0613, W0622
    return widgets.IntProgress(**kwargser(locals()))

def itext(min, max, value, **kwargs):  # pylint: disable=W0613, W0622
    return widgets.BoundedIntText(**kwargser(locals()))

def button(description):
    return widgets.Button(**kwargser(locals()))

def glyph_hover_js_code(element_id, id_name, text_name, glyph_name='glyph', glyph_data='glyph_data'):
    return """
        var indices = cb_data.index['1d'].indices;
        var current_id = -1;
        if (indices.length > 0) {
            var index = indices[0];
            var id = parseInt(""" + glyph_name + """.data.""" + id_name + """[index]);
            if (id !== current_id) {
                current_id = id;
                var text = """ + glyph_data + """.data.""" + text_name + """[id];
                $('.""" + element_id + """').html('ID ' + id.toString() + ': ' + text);
            }
    }
    """
def aggregate_function_widget(**kwopts):
    default_opts = dict(
        options=['mean', 'sum', 'std', 'min', 'max'],
        value='mean',
        description='Aggregate',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

def years_widget(**kwopts):
    default_opts = dict(
        options=[],
        value=None,
        description='Year',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

