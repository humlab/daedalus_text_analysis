# encoding: utf-8
import sys
from gensim.models import Word2Vec
from . import W2V_TensorFlow

if __name__ == "__main__":
    """
    run `python w2v_visualizer.py word2vec.model to visualize_result`
    """
    try:
        model_path = sys.argv[1]
        output_path  = sys.argv[2]
    except:
        model_path = '../data/model_output.dat'
        output_path  = './projector'

        #print("Please provice model path and output path")

    model = Word2Vec.load(model_path)
    W2V_TensorFlow().convert(model, output_path)