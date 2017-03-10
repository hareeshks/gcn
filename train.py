from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN_MLP

from multiprocessing import cpu_count

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string. (cora | citeseer | pubmed)')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('conv', 'gcn', 'conv type. (gcn | cheby)')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.03, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_string('connection', 'd',
                    '''
                    A string contains only char "c" or "d".
                    "c" stands for convolution.
                    "d" stands for dense.
                    See layer_size for details.
                    ''')

flags.DEFINE_string('layer_size', '[]',
                    '''
                    A list or any sequential object. Describe the size of each layer.
                    e.g. "--connection ccd --layer_size [7,8]"
                        This combination describe a network as follow:
                        input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
                        (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)
                    ''')

flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('random_seed', int(time.time()), 'Random seed.')
flags.DEFINE_string('feature', 'bow', 'bow (bag of words) or tfidf.')
flags.DEFINE_string('logdir', '', 'Log directory. Default is ""')
# TODO: auto create logdir name
flags.DEFINE_integer('threads', cpu_count(), 'Number of threads')
flags.DEFINE_integer('train_size', 5, 'use TRAIN_SIZE%% data to train model')

# Parse network structure
FLAGS.layer_size = eval(FLAGS.layer_size)
FLAGS.connection = list(FLAGS.connection)
for c in FLAGS.connection:
    if c not in ['c', 'd']:
        raise ValueError('connection string specified by --connection can only contain "c" or "d", but "{}" found' % c)
for i in FLAGS.layer_size:
    if not isinstance(i, int):
        raise ValueError('layer_size should be a list of int, but found {}' % FLAGS.layer_size)
    if i <= 0:
        raise ValueError('layer_size must be greater than 0, but found {}' % i)
if not len(FLAGS.connection) == len(FLAGS.layer_size) + 1:
    raise ValueError('length of connection string should be equal to length of layer_size list plus 1')

# Set random seed
seed = FLAGS.random_seed
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.conv == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
elif FLAGS.conv == 'cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.conv))

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
    intra_op_parallelism_threads=FLAGS.threads))

# Create model
model = GCN_MLP(placeholders, input_dim=features[2][1])

# Initialize summary
merged = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
train_writer = None
valid_writer = None
if FLAGS.logdir:
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
    valid_writer = tf.summary.FileWriter(FLAGS.logdir + '/valid')


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
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    sess.run(model.opt_op, feed_dict=feed_dict)
    train_loss, train_acc, train_summary = sess.run(
        [model.loss, model.accuracy, merged], feed_dict=feed_dict)

    # Validation
    valid_loss, valid_acc, valid_duration, valid_summary = evaluate(features, support, y_val, val_mask, placeholders)
    valid_loss_list.append(valid_loss)

    # Logging
    if FLAGS.logdir:
        train_writer.add_summary(train_summary, epoch)
        valid_writer.add_summary(valid_summary, epoch)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(valid_loss),
          "val_acc=", "{:.5f}".format(valid_acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        test_cost, test_acc, test_duration, test_summary = evaluate(features, support, y_test, test_mask, placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    if 0 < FLAGS.early_stopping < epoch \
            and valid_loss_list[-1] > np.mean(valid_loss_list[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration, test_summary = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
