from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from gcn.utils import preprocess_model_config

configuration ={
    # The default model configuration
    'default':{
        'dataset'       : 'cora',       # 'Dataset string. (cora | citeseer | pubmed)'
        'conv'          : 'gcn',        # 'conv type. (gcn | cheby)'
        'learning_rate' : 0.03,         # 'Initial learning rate.'
        'epochs'        : 200,          # 'Number of epochs to train.'

        'connection'    : 'cc',
        # A string contains only char "c" or "d".
        # "c" stands for convolution.
        # "d" stands for dense.
        # See layer_size for details.

        'layer_size'    : [16],
        # A list or any sequential object. Describe the size of each layer.
        # e.g. "--connection ccd --layer_size [7,8]"
        #     This combination describe a network as follow:
        #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
        #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)

        'dropout'       : 0.5,          # 'Dropout rate (1 - keep probability).'
        'weight_decay'  : 5e-4,         # 'Weight for L2 loss on embedding matrix.'
        'early_stopping': 0,            # 'Tolerance for early stopping (# of epochs).'
        'max_degree'    : 3,            # 'Maximum Chebyshev polynomial degree.'
        'random_seed'   : int(time.time()),     #'Random seed.'
        'feature'       : 'bow',        # 'bow (bag of words) or tfidf.'

        'logging'       : True,         # 'Weather or not to record log'
        'logdir'        : '',           # 'Log directory.''
        'name'          : '',           # 'name of the model.'
        # if logdir or name are empty string,
        # dir or name will be auto generated according to the
        # structure of the network

        'threads'       : cpu_count(),  #'Number of threads'
        'train_size'    : 5,            # 'Use TRAIN_SIZE%% data to train model'
    },

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':[
        {
            'connection': 'cc',
            'layer_size': [16],
        },
        {
            'connection': 'ddd',
            'layer_size': [16,16],
        },
        {
            'connection': 'cc',
            'conv'      : 'cheby',
            'layer_size': [16],
        }
    ]
}

def set_default_attr(model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config

configuration['model_list'] = list(map(set_default_attr,
    configuration['model_list']))

for model_config in configuration['model_list']:
    preprocess_model_config(model_config)
