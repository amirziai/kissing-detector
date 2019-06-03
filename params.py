seed = 0
n_jobs = 1

data_path_base = 'vtest_new2'

# test end-to-end
experiment_test = {
    'data_path_base': {data_path_base},
    'conv_model_name': {'resnet'},
    'num_epochs': {10},
    'feature_extract': {True},
    'batch_size': {64},
    'lr': {0.001},
    'use_vggish': {False},
    'momentum': {0.9}
}

experiments = {
    'data_path_base': {data_path_base},
    'conv_model_name': {'resnet', None, 'densenet', 'squeezenet'},  # vgg
    'num_epochs': {10},
    'feature_extract': {True, False},
    'batch_size': {64},
    'lr': {1e-3, 1e-2},
    'use_vggish': {False, True},
    'momentum': {0.9, 0.95}
}
