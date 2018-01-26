import sys
import os

__cwd__ = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
__root_path__ = os.path.abspath(os.path.join(__cwd__, '..'))

sys.path.append(__root_path__)

from .topic_modelling import compute, generate_notebook_friendly_data, LdaMalletService, NotebookDataGenerator, ModelUtility, convert_to_pyLDAvis
from common.utility import FileUtility, extend
