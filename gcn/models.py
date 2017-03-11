from gcn.layers import Dense, GraphConvolution
from gcn.metrics import masked_accuracy, masked_softmax_cross_entropy
import tensorflow as tf

class GCN_MLP(object):
    def __init__(self, model_config, placeholders, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = tf.nn.relu

        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config['learning_rate'])

        self.build()
        return

    def build(self):
        self.model_config['connection'] = list(map(
            lambda x: {'c': GraphConvolution, 'd': Dense}.get(x),
            self.model_config['connection']))
        self.model_config['layer_size'].insert(0, self.input_dim)
        self.model_config['layer_size'].append(self.output_dim)
        sparse = True
        with tf.name_scope(self.name):
            # create Variables 
            for input_dim, output_dim, layer_cls in \
                    zip(self.model_config['layer_size'][:-1],
                        self.model_config['layer_size'][1:],
                        self.model_config['connection']):
                self.layers.append(layer_cls(input_dim=input_dim,
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=self.act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging))
                sparse = False

            # Build sequential layer model
            self.activations.append(self.inputs)
            for layer in self.layers:
                hidden = layer(self.activations[-1])  # build the graph, give layer inputs, return layer outpus
                self.activations.append(hidden)
            self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
