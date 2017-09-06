from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from gcn.utils import preprocess_model_config
# Model 11: nearest weighted
# Model 12: see draft
# Model 13: Model 9 + Model 11
# Model 14: variant of weighted Model 11
# Model 15: Model 9 + Model 14
# Model 16: use gcn to extend train set
#           'Model_to_add_label':   the model used to extend train set
#           'Model_to_predict'  :   the final model
#           't' :   if t is a scalar, it's the total number of additional labels
#                   if t is a list, it's the number of additional labels of each class
# 17 label propagation
# 18 self learning with label propagation
# Model 19: gcn and lp
#           'Model19' : 'union',        # 'union' | 'intersection'
# validate: (True | False) Whether use validation set
configuration ={
    # repeating times
    'repeating'             : 1,

    # The default model configuration
    'default':{
        'dataset'           : 'nell.0.001',     # 'Dataset string. (cora | citeseer | pubmed | CIFAR-Fea | USPS-Fea)'
        # 'train_size'        : 1,         # if train_size is a number, then use TRAIN_SIZE%% data to train model
        'train_size'        : [20, 20, 20, 20, 20, 20, 20], # if train_size is a list of numbers, then it specifies training lables for each class.
        'validation_size'   : 20,           # 'Use VALIDATION_SIZE% data to train model'
        'validate'          : False,        # Whether use validation set
        'conv'              : 'gcn',        # 'conv type. (gcn | cheby | chebytheta | gcn_rw)'
        'max_degree'        : 2,            # 'Maximum Chebyshev polynomial degree.'
        'learning_rate'     : 0.02,         # 'Initial learning rate.'
        'epochs'            : 200,          # 'Number of epochs to train.'

        # config the absorption probability
        'Model'             : 0,
        # if Model == 0 do not construct new adjacency matrix
        # if Model == 1, use Model1
        # if Model == 2, use Model2
        's'                 : 100,
        # 's' in the construction of  absorption probability
        # if s = -1 and Model == 1, non zero elements in each row equals to the original adjacency matrix
        'alpha'             : 1e-6,         # 'alpha' in the construction of  absorption probability
        'absorption_type'   : 'binary',     # When Model == 1, the new constructed adjacency matrix is either 'binary' or 'weighted'
        'mu'                : 1,          # 'mu' in the Model5
        't'                 : 100,           # In model9, top 't' nodes will be reserved.
        't2'                : 100,          # In model 21
        'lambda'            : 0,
        'Model11'           : 'nearest',     # 'weighted' | 'nearest'
        'k'                 : -1,           # k in model 11. if k<0, then it is determined by program.
        'Model_to_add_label': { 'Model' :0 }, # for model 16
        'Model_to_predict'  : { 'Model' :0 }, # for model 16
        'Model19'           : 'union',        # 'union' | 'intersection'

        'connection'        : 'cc',
        # A string contains only char "c" or "d".
        # "c" stands for convolution.
        # "d" stands for dense.
        # See layer_size for details.

        'layer_size'        : [16],
        # A list or any sequential object. Describe the size of each layer.
        # e.g. "--connection ccd --layer_size [7,8]"
        #     This combination describe a network as follow:
        #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
        #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)

        'dropout'           : 0.5,          # 'Dropout rate (1 - keep probability).'
        'weight_decay'      : 5e-4,         # 'Weight for L2 loss on embedding matrix.'

        'early_stopping'    : 0,
        # 'Tolerance for early stopping (# of epochs).
        # Non positive value means never early stop.'

        'random_seed'       : int(time.time()),     #'Random seed.'
        'feature'           : 'bow',        # 'bow (bag of words) or tfidf.'

        'logging'           : False,         # 'Weather or not to record log'
        'logdir'            : '',           # 'Log directory.''
        'name'              : '',           # 'name of the model. Serve as an ID of model.'
        # if logdir or name are empty string,
        # logdir or name will be auto generated according to the
        # structure of the network

        'threads'           : cpu_count(),  #'Number of threads'
        'train'             : True,
        'drop_inter_class_edge': False,
    },

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':[
        # {
        #     'Model' : 0,
        #     'dropout'           : 0.1,
        #     'weight_decay'      : 1e-5,
        #     'connection'        : 'cc',
        #     'layer_size'        : [128],
        #     'learning_rate'     : 0.01,
        #     'validate'          : True,
        #     # 'seed'              : 123,
        # },
        {
            'Model': 19,
            'Model19' : 'union',
            't'                 : 100,
            'dropout'           : 0.1,
            'weight_decay'      : 1e-5,
            'connection'        : 'cc',
            'layer_size'        : [64],
            'epochs'            : 3,
        },
        # {
        #     'Model'     : 0,
        #     'layer_size'        : [],
        #     'connection'        : 'd',
        # },
        # {
        #     'Model'     : 0,
        #     'layer_size'        : [],
        #     'connection'        : 'c',
        # },
        # {
        #     'Model'     : 0,
        #     'layer_size'        : [16],
        #     'connection'        : 'dd',
        # },
        # {
        #     'Model'     : 19,
        #     't'         : 50,
        #     'Model19'   : 'intersection'
        # },
        # {
        #     'Model'     : 20,
        #     't'         : 200,
        # },
        # {
        #     'Model'     : 21,
        #     't'         : 200,
        #     't2'        : [50, 50, 50, 50, 50, 50, 50]
        # }
        # {
        #     'Model'         : 16,
        #         't'                 : [100, 100, 100, 100, 100, 100, 100],
        #     'Model_to_add_label': {
        #         'Model'             :16,
        #         't'             : [50, 50, 50, 50, 50, 50, 50],
        #         'Model_to_add_label': {
        #             'Model'             :16,
        #             't'             : [25, 25, 25, 25, 25, 25, 25],
        #             'Model_to_add_label': { 'Model' :0 }, # for model 16
        #             'Model_to_predict'  : { 'Model' :0 }, # for model 16
        #         },
        #         'Model_to_predict'  : { 'Model' :0 }, # for model 16
        #     },
        #     'Model_to_predict'  : { 'Model' :0 }, # for model 16
        # },
        # {
        #     'Model'         : 19,
        #     'Model19'       : 'union',
        #     't'             : [50, 50, 50, 50, 50, 50, 50]
        # },
        # {
        #     'Model'                 : 0,
        #     'validate'          : True,
        # },
        # {
        #     'Model'                 : 0,
        #     'conv'                  : 'gcn_rw',
        #     'validate'          : True,
        # },
        # {
        #     'Model'                 :0,
        #     'connection'        : 'ccc',
        #     'layer_size'        : [32, 16],
        #     'conv'              : 'gcn_rw',
        #     'validate'          : True,
        # },
        # {
        #     'Model'                 :0,
        #     'connection'        : 'cccc',
        #     'layer_size'        : [64, 32, 16],
        #     'conv'              : 'gcn_rw',
        #     'validate'          : True,
        # },
        # {
        #     'Model'                 :0,
        #     'connection'        : 'ccccc',
        #     'layer_size'        : [128, 64, 32, 16],
        #     'conv'              : 'gcn_rw',
        #     'validate'          : True,
        # },
        # {
        #     'Model'                 :0,
        #     'connection'        : 'cccccc',
        #     'layer_size'        : [256, 128, 64, 32, 16],
        #     'conv'              : 'gcn_rw',
        #     'validate'          : True,
        # },

        # {
        #     'Model'                 :17
        # },
        # {
        #     'Model'                 :17,
        #     'drop_inter_class_edge' : True,
        # },
        # {
        #     'Model'             : 17,
        #     't'                 : 20,
        #     'train_size'        : [20, 20, 20, 20, 20, 20, 20]
        # },
        # {
        #     'Model'             : 17,
        #     'train_size'        : [2, 2, 2, 2, 2, 2, 2]
        # },
        # {
        #     'Model'             : 0,
        # },
        # {
        #     'Model'             : 0,
        #     'conv'              : 'cheby',
        #     'max_degree'        : 2,
        # },
        # {
        #     'Model'             : 16,
        #     'validate'          : False,        # Whether use validation set
        #     't'                 : [50, 50, 50, 50, 50, 50, 50], # t additional labels for each class
        #     'Model_to_add_label': {
        #         'Model' :0
        #     },
        #     'Model_to_predict'  : {
        #         'Model' :0
        #     }
        # },
        # {
        #     'Model'         : 5,
        # },
        # {
        #     'Model'         : 0,
        # },
        # {
        #     'Model'         : 0,
        #     'Model5'        : 1.5
        # },
        # {
        #     'Model'         : 0,
        #     'Model5'        : 1.5
        # },
        # {
        #     'Model'         : 0,
        #     'Model5'        : 1.5
        # },
        # {
        #     'Model'         : 0,
        #     'Model5'        : 1.0
        # },
        # {
        #     'Model'         : 0,
        #     'Model5'        : 1.0,
        #     'conv'          : 'cheby',        # 'conv type. (gcn | cheby)'
        #     'max_degree'    : 2,            # 'Maximum Chebyshev polynomial degree.'
        # },
        # {
        #     'Model'         : 9,
        #     's'             : 100,
        #     't'             : 50,
        # },
        # {
        #     'Model'         : 13,
        #     'Model11'       : 'weighted',
        #     's'             : 100,
        #     't'             : 50,
        # },
        # {
        #     'Model'         : 15,
        #     's'             : 100,
        #     't'             : 50,
        # }
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
