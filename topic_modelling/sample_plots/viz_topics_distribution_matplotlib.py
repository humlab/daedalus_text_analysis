# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
from topic_modelling.compute_topic_models.model_store import ModelStore as store

def plot_yearly_distributions(df):

    min_year, max_year = df['year'].min(), df['year'].max()
    n_topics = df['topic_id'].max() + 1
    # width = 0.35
    ind = np.arange(n_topics)
    years = list(range(min_year, max_year + 1))
    for year in years:
        df_year = df[df.year == year]
        topic_weights = [
            tuple(x) for x in df_year[['topic_id', 'weight']].values if x[1] > 0.10
        ]
        plt.title("Topic distribution")
        plt.ylabel("P(topic)")
        plt.ylim(0.0, 1.0)
        plt.xticks(ind, [ str(x % 100) for x in years] )
        #plt.grid(True)
        plt.bar( [ x[0] for x in topic_weights ], [ x[0] for x in topic_weights ], align="center")
        plt.show()

if __name__ == '__main__':

    source_folder = '/tmp'
    models_names = store.get_model_names(source_folder)

    basename = models_names[-1]
    data_folder = os.path.join(source_folder, basename)

    if 'df_yearly_mean_topics' not in globals():
        df_doc_topics = df = store.load_document_topics('/tmp', basename, True)
        df_yearly_mean_topics = df_doc_topics.groupby(['year', 'topic_id']).mean()[['weight']].reset_index()

    plot_yearly_distributions(df_yearly_mean_topics)