import numpy as np

seed = 0
n_jobs = 1

data_path_base = 'vtest_new2'

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
vggish_frame_rate = 0.96

# test end-to-end
experiment_test = {
    'data_path_base': {data_path_base},
    'conv_model_name': {'resnet'},
    'num_epochs': {10},
    'feature_extract': {False},
    'batch_size': {64},
    'lr': {0.001},
    'use_vggish': {True},
    'momentum': {0.9}
}

experiment_test_3d = {
    'data_path_base': {data_path_base},
    'conv_model_name': {'resnet'},
    'num_epochs': {15},
    'feature_extract': {True, False},
    'batch_size': {64},
    'lr': {0.001},
    'use_vggish': {True, False},
    'momentum': {0.9},
    'use_3d': {True}
}

experiments = {
    'data_path_base': {data_path_base},
    'conv_model_name': {'resnet', None, 'vgg', 'densenet', 'squeezenet'},
    'num_epochs': {10},
    'feature_extract': {True, False},
    'batch_size': {64},
    'lr': {1e-3, 1e-2},
    'use_vggish': {False, True},
    'momentum': {0.9, 0.95}
}
