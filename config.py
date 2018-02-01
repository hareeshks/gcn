from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from gcn.utils import preprocess_model_config
import argparse
import pprint
# import math
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
# 22 LP with features
# validate: (True | False) Whether use validation set
# 23 other classifier
# 24 Test21
# 25 Test22
# 26 Model9 with extra edges
# 27 Test21 with different threshold for different row (preserve beta energy)
# 28 imitate spectral clustering
configuration ={
    # repeating times
    'repeating'             : 1,

    # The default model configuration
    'default':{
        'dataset'           : 'cora',     # 'Dataset string. (cora | citeseer | pubmed | CIFAR-Fea | Cifar_10000_fea | Cifar_R10000_fea | USPS-Fea | MNIST-Fea | MNIST-10000)'
        'shuffle'           : True,
        'train_size'        : 20,         # if train_size is a number, then use TRAIN_SIZE labels per class.
        # 'train_size'        : [20 for i in range(10)], # if train_size is a list of numbers, then it specifies training labels for each class.
        'validation_size'   : 500,           # 'Use VALIDATION_SIZE data to train model'
        'validate'          : False,        # Whether use validation set
        'conv'              : 'gcn',        # 'conv type. (gcn | cheby | chebytheta | gcn_rw | taubin | test21)'
        'max_degree'        : 2,            # 'Maximum Chebyshev polynomial degree.'
        'learning_rate'     : 0.005,         # 'Initial learning rate.'
        'epochs'            : 200,          # 'Number of epochs to train.'

        # config the absorption probability
        'Model'             : 0,
        # 's'                 : 100,
        # 's' in the construction of  absorption probability
        # if s = -1 and Model == 1, non zero elements in each row equals to the original adjacency matrix
        'absorption_type'   : 'binary',     # When Model == 1, the new constructed adjacency matrix is either 'binary' or 'weighted'
        'mu'                : 1,          # 'mu' in the Model5
        't'                 : 500,           # In model9, top 't' nodes will be reserved.
        't2'                : 100,          # In model 21
        'lambda'            : 0,
        'Model11'           : 'nearest',     # 'weighted' | 'nearest'
        'k'                 : -1,           # k in model 11. if k<0, then it is determined by program.
        'Model_to_add_label': { 'Model' :0 }, # for model 16
        'Model_to_predict'  : { 'Model' :0 }, # for model 16
        'Model19'           : 'union',        # 'union' | 'intersection'
        'classifier'        : 'svm',            # 'svm' | 'tree'
        'svm_kernel'        : 'rbf',        # 'rbf' | 'poly' | 'rbf' | 'sigmoid', model 23
        'gamma'             : 1e-5,         # gamma for svm, see scikit-learn document, model 23
        'svm_degree'        : 4,
        'tree_depth'        : None,

        'connection'        : 'cc',
        # A string contains only char "c" or "f" or "d".
        # "c" stands for convolution.
        # "f" stands for fully connected.
        # "d" stands for dense net.
        # See layer_size for details.

        'layer_size'        : [16],
        # A list or any sequential object. Describe the size of each layer.
        # e.g. "--connection ccd --layer_size [7,8]"
        #     This combination describe a network as follow:
        #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
        #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)

        'dropout'           : 0.2,          # 'Dropout rate (1 - keep probability).'
        'weight_decay'      : 5e-4,         # 'Weight for L2 loss on embedding matrix.'

        'early_stopping'    : 0,
        # 'Tolerance for early stopping (# of epochs).
        # Non positive value means never early stop.'

        'random_seed'       : int(time.time()),     #'Random seed.'
        'feature'           : 'bow',        # 'bow' | 'tfidf' | 'none'.

        'smoothing'         : None,        # 'poly'| 'ap'  | 'taubin' | 'test21' | 'test21_norm' | None
        'alpha'             : 1e-6,         # 'alpha' in the construction of  absorption probability
        'beta'              : 10,
        'poly_parameters'   : [1,-2,1],           # coefficients of p(L_rw)
        'taubin_lambda'     : 0.3,
        'taubin_mu'         : -0.31,
        'taubin_repeat'     : 5,
        'taubin_f'          : 0.7,
        'taubin_t'          : 0.2,

        'logging'           : False,         # 'Weather or not to record log'
        'logdir'            : None,           # 'Log directory.''
        'model_dir'         : './model/',
        'name'              : None,           # 'name of the model. Serve as an ID of model.'
        # if logdir or name are empty string,
        # logdir or name will be auto generated according to the
        # structure of the network

        'threads'           : 2*cpu_count(),  #'Number of threads'
        'train'             : True,
        'drop_inter_class_edge': False,
        'loss_func'         :'default',     #'imbalance', 'triplet'
        'ws_beta'           : 20,
		'max_triplet':1000  #for triplet, 1000 for cora to get all tripets
    },

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':
    [
        # GCN
        {
            'Model'     :0,
            'connection': 'cc',
            'layer_size': [256],
        },
        # LP
        {
            'Model': 17,
            'alpha': 1e-6,
        },
        # MLP
        {
            'Model'     :0,
            'connection': 'ff',
            'layer_size': [256],
        },
    ]+[
        {
            'Model'     :0,
            'connection': 'ff',
            'layer_size': [256],

            'smoothing': 'taubin',
            'taubin_lambda': 1,
            'taubin_mu': 0,
            'taubin_repeat': repeat,
        } for repeat in [3]
    ]+[
        {
            'Model'     :0,
            'connection': 'ff',
            'layer_size': [256],

            'smoothing': 'taubin',
            'taubin_lambda': 0.5,
            'taubin_mu': 0,
            'taubin_repeat': repeat,
        } for repeat in [6]
    ]+[
        {
            'Model'     :0,
            'connection': 'ff',
            'layer_size': [256],

            'smoothing': 'ap',
            'alpha': 0.5,
        }
    ]+[
        {
            'Model'     :0,
            'connection': 'ff',
            'layer_size': [256],

            'smoothing': 'test21',
            'alpha': 0.5,
            'beta' : beta,
        } for beta in [5, 10, 15, 20, 30, 40]
    ]
    # +[
    #     {
    #         'train_size'        : 60,
    #         'Model'             : 23,
    #         'classifier'        : 'cnn',
    #         'learning_rate'     : 0.001,
    #         'epochs'            : 400,
    #
    #         'smoothing'         : 'taubin',
    #         'taubin_lambda'     : 1,
    #         'taubin_mu'         : 0,
    #         'taubin_repeat'     : repeat,
    #     } for repeat in [0]
    # ]
    # +
    # [
    #     {
    #         'train_size'        : 60,
    #         'Model'             : 23,
    #         'classifier'        : 'cnn',
    #         'learning_rate'     : 0.001,
    #         'epochs'            : 400,
    #
    #         'smoothing'         : 'test21',
    #         'alpha'             : 0.3,
    #         'beta'              : 200,
    #     }
    # ]
    # +
    # [
    #     {
    #         'Model'     :0,
    #         'connection': 'ff',
    #         'layer_size': [64],
    #         'smoothing': 'taubin',
    #         # 'smoothing'         : None,
    #         'taubin_lambda': 1,
    #         'taubin_mu': 0,
    #         'taubin_repeat': taubin_repeat,
    #     } for taubin_repeat in [4,2,0]
    # ]
    # +
    # [
    #     {
    #         'Model'     :0,
    #         'connection': 'ff',
    #         'layer_size': [256],
    #         'epochs'    : 200,
    #         'learning_rate': 0.005,
    #     },
    # ]
    #     +
    # [
    #     {
    #         'Model'     :0,
    #         'connection': 'cc',
    #         'layer_size': [256],
    #         'epochs'    : 200,
    #         'learning_rate': 0.005,
    #     },
    # ]
    # +
    # [
    #     {
    #         'Model': 17,
    #         'alpha': alpha,
    #     } for alpha in [1e-6]
    # ]
    # +
    # [
    #     # smoothing by test21
    #     {
    #         'train_size': train_size,
    #         'Model' : 0,
    #         'smoothing'         :'test21',
    #         'alpha'             : alpha,
    #         'beta'              : beta,
    #         'connection'        : 'ff',
    #         'conv'              : 'gcn',
    #         'layer_size': [64],
    #     } for train_size in [4,8,12,16,20] for beta in [200, 300, 400] for alpha in [0.05, 0.1, 0.3]
    # ]
    # [
    #     # gcn_taubin
    #     {
    #         'Model' : 0,
    #         'smoothing'         :  'taubin',
    #         'connection'        : 'ff',
    #         'taubin_lambda'     : 1,
    #         'taubin_mu'         : 0,
    #         'taubin_repeat'     : repeat,
    #     } for repeat in [3,4,5]
    # ] +
    # [
    #     # Ideal low-pass filter
    #     {
    #         'Model' : 28,
    #         'smoothing'         :  None,
    #         'connection'        : 'ff',
    #         'conv'              : 'gcn',
    #         'k'                 : k
    #     } for k in [150, 200, 300]
    # ]
    # +
    # [
    #     # only one convolutional layer
    #     {
    #         'Model': 0,
    #         'connection': 'cf',
    #         'conv': 'test21',
    #         'alpha': 0.3,
    #         'beta': 0.001,
    #     },
    # ] +
}

# Parse args
parser = argparse.ArgumentParser(description=(
    "This is used to train and test Graph Convolution Network for node classification problem.\n"
    "Most configuration are specified in config.py, please read it and modify it as you want."))
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--dataset", type=str)
parser.add_argument("--train_size", type=str)
parser.add_argument("--repeating", type=int)
parser.add_argument("--validate", type=bool, help='0 | 1')
parser.add_argument("--loss_func", type=str)
parser.add_argument("--ws_beta", type=int)

args = parser.parse_args()
print(args)
if args.dataset is not None:
    configuration['default']['dataset'] = args.dataset
if args.train_size is not None:
    configuration['default']['train_size'] = eval(args.train_size)
if args.repeating is not None:
    configuration['repeating']=args.repeating
if args.validate is not None:
    configuration['default']['validate']=args.validate
if args.loss_func is not None:
    configuration['default']['loss_func']=args.loss_func
if args.ws_beta is not None:
    configuration['default']['ws_beta']=args.ws_beta
pprint.PrettyPrinter(indent=4).pprint(configuration)
# exit()

def set_default_attr(model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config

configuration['model_list'] = list(map(set_default_attr,
    configuration['model_list']))

for model_config in configuration['model_list']:
    preprocess_model_config(model_config)
