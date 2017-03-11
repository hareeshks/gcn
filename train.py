from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN_MLP

from config import configuration
import argparse

# Parse args
parser = argparse.ArgumentParser(description=
                   '''This is used to train and test Graph Convolution Network for node classification problem.
                   All configuration are specified in config.py, please read it and modify it as you want.
                   ''')
parser.parse_args()

# Read configuration
model_config = configuration['model_list'][0]

# Set random seed
seed = model_config['random_seed']
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data(model_config['dataset'], train_size=model_config['train_size'])

# Some preprocessing
features = preprocess_features(features, feature_type=model_config['feature'])
if model_config['conv'] == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
elif model_config['conv'] == 'cheby':
    support = chebyshev_polynomials(adj, model_config['max_degree'])
    num_supports = 1 + model_config['max_degree']
else:
    raise ValueError('Invalid argument for model: ' + str(model_config['conv']))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Initialize session
sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=model_config['threads']))

# Create model
model = GCN_MLP(model_config, placeholders, input_dim=features[2][1])

# Initialize summary
merged = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
train_writer = None
valid_writer = None
if model_config['logdir']:
    train_writer = tf.summary.FileWriter(model_config['logdir'] + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(model_config['logdir'] + '/valid')


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, merged], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
sess.run(tf.global_variables_initializer())

valid_loss_list = []
max_valid_acc = 0
# Train model
for epoch in range(model_config['epochs']):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: model_config['dropout']})

    # Training step
    sess.run(model.opt_op, feed_dict=feed_dict)
    train_loss, train_acc, train_summary = sess.run(
        [model.loss, model.accuracy, merged], feed_dict=feed_dict)

    # Validation
    valid_loss, valid_acc, valid_duration, valid_summary = evaluate(features, support, y_val, val_mask, placeholders)
    valid_loss_list.append(valid_loss)

    # Logging
    if model_config['logdir']:
        train_writer.add_summary(train_summary, epoch)
        valid_writer.add_summary(valid_summary, epoch)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(valid_loss),
          "val_acc=", "{:.5f}".format(valid_acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > model_config['early_stopping'] and valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        test_cost, test_acc, test_duration, test_summary = evaluate(features, support, y_test, test_mask, placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    if 0 < model_config['early_stopping'] < epoch \
            and valid_loss_list[-1] > np.mean(valid_loss_list[-(model_config['early_stopping'] + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration, test_summary = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
