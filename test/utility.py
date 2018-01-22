import os

current_folder = os.path.dirname(__file__)
test_data_folder = os.path.join(current_folder, 'test_data')

def join_test_data_path(filename):
    return os.path.join(test_data_folder, filename)

