# encoding: utf-8
"""
@author: BrikerMan
@contact: eliyar917@gmail.com
@blog: https://eliyar.biz
@version: 1.0
@license: Apache Licence
@file: w2v_visualizer.py
@time: 2017/7/30 上午9:37
"""
import sys
import os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

import logging
logger = logging.getLogger(__name__)

class W2V_TensorFlow():

    # http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/

    def convert_model(self, model, output_path, tensor_name='w2x_metadata', dimension=100):

        if not os.path.exists(output_path):
            raise Exception('Unknown path {}'.format(output_path))

        meta_file = tensor_name + ".tsv"
        placeholder = np.zeros((len(model.wv.index2word), dimension))

        with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
            for i, word in enumerate(model.wv.index2word):
                placeholder[i] = model[word]
                # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
                if word == '':
                    logger.warning("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                    file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
                else:
                    file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

        # define the model without training
        sess = tf.InteractiveSession()

        _ = tf.Variable(placeholder, trainable=False, name=tensor_name)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(output_path, sess.graph)

        # adding into projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = tensor_name
        embed.metadata_path = meta_file

        # Specify the width and height of a single thumbnail.
        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))

        logger.info('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))

    def convert_file(self, model_path, target_dir=None, dimension=100):

        if target_dir is None:
            target_dir = os.path.split(model_path)[0]

        basename = 'tensor_flow_' + os.path.basename(model_path).split('.')[0]
        target_dir = os.path.join(target_dir, basename)

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        model = Word2Vec.load(model_path)
        self.convert_model(model, target_dir, dimension=dimension)
        return model

if __name__ == "__main__":
    """
    Just run `python w2v_visualizer.py word2vec.model visualize_result`
    """
    try:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        model_path = '../data/model_output.dat'
        output_path = './projector'

    model = Word2Vec.load(model_path)
    W2V_TensorFlow().convert(model, output_path)