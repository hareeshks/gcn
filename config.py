from multiprocessing import cpu_count
import time
from copy import deepcopy
configuration ={
    'default':{
        'dataset'       : 'cora',     # 'Dataset string. (cora | citeseer | pubmed)'
        'conv'          : 'gcn',      # 'conv type. (gcn | cheby)'
        'learning_rate' : 0.03,       # 'Initial learning rate.'
        'epochs'        : 200,        # 'Number of epochs to train.'

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

        'dropout'       : 0.5,        # 'Dropout rate (1 - keep probability).'
        'weight_decay'  : 5e-4,       # 'Weight for L2 loss on embedding matrix.'
        'early_stopping': 0,         # 'Tolerance for early stopping (# of epochs).'
        'max_degree'    : 3,          #'Maximum Chebyshev polynomial degree.'
        'random_seed'   : int(time.time()),     #'Random seed.'
        'feature'       : 'bow',      # 'bow (bag of words) or tfidf.'
        'logdir'        : '',         # 'Log directory. Default is ""'
        'threads'       : cpu_count(),     #'Number of threads'
        'train_size'    : 5,          # 'Use TRAIN_SIZE%% data to train model'
        'name'          : '',         # 'name of the model'
    },
    'model_list':[
        {
            'name'      : 'c16c',
            'logdir'    : 'log/c16c',
            'connection': 'cc',
            'layer_size': [16],
        },
        {
            'name'      : 'd16d16d',
            'logdir'    : 'log/d16d16d',
            'connection': 'ddd',
            'layer_size': [16,16],
        },
    ]
}
def set_default_attr(model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config

configuration['model_list'] = list(map(set_default_attr,
    configuration['model_list']))

for model_config in configuration['model_list']:
    model_config['connection'] = list(model_config['connection'])
    for c in model_config['connection']:
        if c not in ['c', 'd']:
            raise ValueError('connection string specified by --connection can only contain "c" or "d", but "{}" found' % c)
    for i in model_config['layer_size']:
        if not isinstance(i, int):
            raise ValueError('layer_size should be a list of int, but found {}' % model_config['layer_size'])
        if i <= 0:
            raise ValueError('layer_size must be greater than 0, but found {}' % i)
    if not len(model_config['connection']) == len(model_config['layer_size']) + 1:
        raise ValueError('length of connection string should be equal to length of layer_size list plus 1')
