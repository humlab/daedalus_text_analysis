import sys
sys.path.insert(0, '.')
import ipywidgets as widgets
import common.utility as utility
from bokeh.models import ColumnDataSource, CustomJS

BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')
def kwargser(d):
    args = dict(d)
    if 'kwargs' in args:
        kwargs = args['kwargs']
        del args['kwargs']
        args.update(kwargs)
    return args

# FIXME These functions are copied from widgets_config

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

def years_widget(**kwopts):
    default_opts = dict(
        options=[],
        value=None,
        description='Year',
        layout=widgets.Layout(width='200px')
    )
    return widgets.Dropdown(**extend(default_opts, kwopts))

def text_widget(self, element_id=None, default_value='', style='', line_height='20px'):
    value = "<span class='{}' style='line-height: {};{}'>{}</span>".format(element_id, line_height, style, default_value) if element_id is not None else ''
    return widgets.HTML(value=value, placeholder='', description='')

def create_text_widget(self, element_id=None, default_value=''):
    value = "<span class='{}'>{}</span>".format(element_id, default_value) if element_id is not None else ''
    return widgets.HTML(value=value, placeholder='', description='')
    
class WidgetUtility():

    @staticmethod
    def create_js_callback(axis, attribute, source):
        return CustomJS(args=dict(source=source), code="""
            var data = source.data;
            var start = cb_obj.start;
            var end = cb_obj.end;
            data['""" + axis + """'] = [start + (end - start) / 2];
            data['""" + attribute + """'] = [end - start];
            source.change.emit();
        """)

    @staticmethod
    def glyph_hover_callback(glyph_source, glyph_id, text_ids, text, element_id):
        source = ColumnDataSource(dict(text_id=text_ids, text=text))
        code = glyph_hover_js_code(element_id, glyph_id, 'text', glyph_name='glyph', glyph_data='glyph_data')
        callback = CustomJS(args={'glyph': glyph_source, 'glyph_data': source}, code=code)
        return callback

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.__dict__.update(kwargs)
        
    def reset(self, description=None):
        if 'progress' in self.__dict__.keys():
            self.progress.value = 0
            
    def forward(self, description=None):
        if 'progress' in self.__dict__.keys():
            self.progress.value = self.progress.value + 1
            if description is not None:
                self.progress.description = description

    def create_select_widget(self, label='', values=None, default=None, **kwargs):
        opts = dict(
            options=values or [],
            value=default if default is not None and default in values else values[0] if len(values or []) > 0 else None,
            description=label,
            disabled=False
        )
        opts = utility.extend(opts, kwargs)
        return widgets.Dropdown(**opts)

    def create_int_slider(self, description, **args):
        args = utility.extend(dict(min=0, max=0, step=1, value=0, disabled=False, continuous_update=False), args)
        return widgets.IntSlider(description=description, **args)

    def create_float_slider(self, description, **args):
        args = utility.extend(dict(min=0.0, max=0.0, step=0.1, value=0.0, disabled=False, continuous_update=False), args)
        return widgets.FloatSlider(description=description, **args)

    def layout_algorithm_widget(self, options, default=None):
        return dropdown('Layout', options, default)

    def topic_id_slider(self, n_topics):
        return slider('Topic ID', 0, n_topics - 1, 0, step=1)

    def word_count_slider(self, min=1, max=500):  # pylint: disable=W0622
        return slider('Word count', min, max, 50)

    def topic_count_slider(self, n_topics):
        return slider('Topic count', 0, n_topics - 1, 3, step=1)

    def create_button(self, description, style=None, callback=None):
        style = style or dict(description_width='initial', button_color='lightgreen')
        button = widgets.Button(description=description, style=style)
        if callback is not None:
            button.on_click(callback)
        return button

    def create_text_input_widget(self, **opts):
        return widgets.Text(**opts)

    def create_text_widget(self, element_id=None, default_value=''):
        value = "<span class='{}'>{}</span>".format(element_id, default_value) if element_id is not None else ''
        return widgets.HTML(value=value, placeholder='', description='')

    def create_prev_button(self, callback):
        return self.create_button(description="<<", callback=callback)

    def create_next_button(self, callback):
        return self.create_button(description=">>", callback=callback)

    def create_next_id_button(self, name, count):
        that = self

        def f(_):
            control = getattr(that, name, None)
            if control is not None:
                control.value = (control.value + 1) % count

        return self.create_button(description=">>", callback=f)

    def create_prev_id_button(self, name, count):
        that = self

        def f(_):
            control = getattr(that, name, None)
            if control is not None:
                control.value = (control.value - 1) % count

        return self.create_button(description="<<", callback=f)

    def next_topic_id_clicked(self, _):
        self.topic_id.value = (self.topic_id.value + 1) % self.n_topics

    def prev_topic_id_clicked(self, _):
        self.topic_id.value = (self.topic_id.value - 1) % self.n_topics


class TopicWidgets(WidgetUtility):

    def __init__(self, n_topics, years=None, word_count=None, text_id=None):

        self.n_topics = n_topics
        self.text_id = text_id
        self.text = self.create_text_widget(text_id)
        self.year = years_widget(options=years) if years is not None else None
        self.topic_id = self.topic_id_slider(n_topics)

        self.word_count = self.word_count_slider(1, 500) if word_count is not None else None

        self.prev_topic_id = self.create_prev_button(self.prev_topic_id_clicked)
        self.next_topic_id = self.create_next_button(self.next_topic_id_clicked)

class TopTopicWidgets(WidgetUtility):

    def __init__(self, n_topics=0, years=None, aggregates=None, text_id='text_id', layout_algorithms=None):

        self.n_topics = n_topics
        self.text_id = text_id
        self.text = self.create_text_widget(text_id) if text_id is not None else None
        self.year = years_widget(options=years) if years is not None else None

        self.topics_count = self.topic_count_slider(n_topics) if n_topics > 0 else None

        self.aggregate = self.select_aggregate_fn_widget(aggregates, default='mean') if aggregates is not None else None
        self.layout_algorithm = self.layout_algorithm_widget(layout_algorithms, default='Fruchterman-Reingold') \
            if layout_algorithms is not None else None

wf = WidgetUtility()
