# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import wordcloud
from topic_modelling.compute_topic_models.model_store import ModelStore as store

def plot_topic_wordclouds(df_topic_token_weights, topics=None, ncols=5):

    topics = topics or df_topic_token_weights.groupby("topic_id").groups
    nrows = len(topics) // ncols

    _, axes = plt.subplots(figsize=(25.0, 25.0), nrows=nrows, ncols=ncols, subplot_kw={'xticks': [], 'yticks': []})
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    for topic in topics.keys():
        token_weights = dict({ tuple(x) for x in df_topic_token_weights.loc[(df_topic_token_weights.topic_id == topic)][['token', 'weight']].values })
        wc = wordcloud.WordCloud()
        wc.fit_words(token_weights)
        x = topic // ncols
        y = topic % ncols
        ax = axes[x, y]
        ax.imshow(wc)
        ax.set_title('Topic ' + str(topic))

    plt.tight_layout()
    plt.show()

def plot_topic_wordcloud(df_weights, topic_id, max_font_size=40, background_color="white", max_words=1000, width=400, height=200):

    token_weights = dict({ tuple(x) for x in df_topic_token_weights.loc[(df_topic_token_weights.topic_id == topic_id)][['token', 'weight']].values })

    image = wordcloud.WordCloud(max_font_size=80, background_color=background_color, max_words=max_words) #, width=width, height=height)
    image.fit_words(token_weights)

    plt.figure(figsize=(6, 3)) #, dpi=100)
    plt.imshow(image) #, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return token_weights

if __name__ == '__main__':

    source_folder = '/tmp'
    models_names = store.get_model_names(source_folder)

    basename = models_names[-1]
    data_folder = os.path.join(source_folder, basename)

    df_topic_token_weights =  store.load_topic_tokens(source_folder, basename)

    plot_topic_wordcloud(df_topic_token_weights, 0)
