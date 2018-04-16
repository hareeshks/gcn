from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from scipy import sparse
from sklearn import svm, tree
from os import path
from gcn.utils import construct_feed_dict, preprocess_features, drop_inter_class_edge,\
    preprocess_adj, chebyshev_polynomials, load_data, sparse_to_tuple, \
    Model1, Model2, Model3, Model4, Model5, Model6, Model7, Model8, Model9, \
    Model10, Model11, Model12, Model16, Model17, Model19, Model20, Model22, taubin_smoothor, \
    smooth, Model26, Test21, Model28, construct_knn_graph
from gcn.models import GCN_MLP
import cnn
import pprint
from config import configuration, args


def train(model_config, sess, seed, repeat_state, data_split = None):
    # Print model_config
    very_begining = time.time()
    print('',
          'name           : {}'.format(model_config['name']),
          'logdir         : {}'.format(model_config['logdir']),
          'dataset        : {}'.format(model_config['dataset']),
          'train_size     : {}'.format(model_config['train_size']),
          'learning_rate  : {}'.format(model_config['learning_rate']),
          'feature        : {}'.format(model_config['feature']),
          'logging        : {}'.format(model_config['logging']),
          sep='\n')

    if data_split:
        adj         = data_split['adj']
        features    = data_split['features']
        y_train     = data_split['y_train']
        y_val       = data_split['y_val']
        y_test      = data_split['y_test']
        train_mask  = data_split['train_mask']
        val_mask    = data_split['val_mask']
        test_mask   = data_split['test_mask']
        triplet     = data_split['triplet']
    else:
        # Load data
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, size_of_each_class, triplet = \
            load_data(model_config['dataset'],train_size=model_config['train_size'],
                      validation_size=model_config['validation_size'],
                      model_config=model_config, shuffle=model_config['shuffle'], repeat_state=repeat_state)
        stored_A = model_config['dataset']
        if model_config['drop_inter_class_edge']:
            adj = drop_inter_class_edge(adj)
            stored_A = model_config['dataset']+'_drop'
        # preprocess_features
        begin = time.time()
        features = smooth(features, adj, model_config['smoothing'], model_config, stored_A=stored_A + '_A_I')
        print(time.time()-begin,'s')
        data_split = {
            'adj' : adj,
            'features' : features,
            'y_train' : y_train,
            'y_val' : y_val,
            'y_test' : y_test,
            'train_mask' : train_mask,
            'val_mask' : val_mask,
            'test_mask' : test_mask,
            'triplet' : triplet
        }
    laplacian = sparse.diags(adj.sum(1).flat, 0) - adj
    laplacian = laplacian.astype(np.float32).tocoo()
    if type(model_config['t'])==int and model_config['t'] < 0:
        eta = adj.shape[0]/(adj.sum()/adj.shape[0])**len(model_config['connection'])
        model_config['t'] = (y_train.sum(axis=0)*3*eta/y_train.sum()).astype(np.int64)
        print('t=',model_config['t'])

    # origin_adj = adj
    if model_config['Model'] == 0:
        pass
    elif model_config['Model'] in [1, 2, 3, 4]:
        # absorption probability
        print('Calculating Absorption Probability...',
              # 's        :{}'.format(model_config['s']),
              'alpha    :{}'.format(model_config['alpha']),
              'type     :{}'.format(model_config['absorption_type']),
              sep='\n')
        if model_config['Model'] == 1:
            adj = Model1(adj, model_config['t'], model_config['alpha'], model_config['absorption_type'])
        elif model_config['Model'] == 2:
            adj = Model2(adj, model_config['s'], model_config['alpha'], y_train)
        elif model_config['Model'] == 3:
            # original_y_train = y_train
            y_train, train_mask = Model3(adj, model_config['s'], model_config['alpha'], y_train, train_mask)
        elif model_config['Model'] == 4:
            y_train, train_mask = Model4(adj, model_config['s'], model_config['alpha'], y_train, train_mask)
    elif model_config['Model'] == 5:
        adj = Model5(features, adj, model_config['mu'])
    elif model_config['Model'] == 6:
        adj = Model6(adj)
    elif model_config['Model'] == 7:
        y_train, train_mask = Model7(adj, model_config['s'], model_config['alpha'], y_train, train_mask, features)
    elif model_config['Model'] == 8:
        # original_y_train = y_train
        y_train, train_mask = Model8(adj, model_config['s'], model_config['alpha'], y_train, train_mask)
    elif model_config['Model'] == 9:
        y_train, train_mask = Model9(adj, model_config['t'], model_config['alpha'],
                                     y_train, train_mask, stored_A = stored_A+'_A_I')
    elif model_config['Model'] == 10:
        y_train, train_mask = Model10(adj, model_config['s'], model_config['t'], model_config['alpha'],
                                      y_train, train_mask, features, stored_A = stored_A+'_A_H')
    elif model_config['Model'] == 11:
        y = np.sum(train_mask)
        label_per_sample, sample2label = Model11(y, y_train, train_mask)
    elif model_config['Model'] == 12:
        pass
    elif model_config['Model'] == 13:
        y_train, train_mask = Model9(adj, model_config['t'], model_config['alpha'],
                                     y_train, train_mask, stored_A = stored_A+'_A_I')
        y = np.sum(train_mask)
        label_per_sample, sample2label = Model11(y, y_train, train_mask)
    elif model_config['Model'] == 14:
        y = np.sum(train_mask)
        label_per_sample, sample2label = Model11(y, y_train, train_mask)
    elif model_config['Model'] == 15:
        y_train, train_mask = Model9(adj, model_config['t'], model_config['alpha'],
                                     y_train, train_mask, stored_A = stored_A+'_A_I')
        y = np.sum(train_mask)
        label_per_sample, sample2label = Model11(y, y_train, train_mask)
    elif model_config['Model'] == 16:
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=model_config['threads'])) as sub_sess:
                tf.set_random_seed(seed)
                test_acc, test_acc_of_class, prediction = train(model_config['Model_to_add_label'], sub_sess, seed, data_split=data_split)
        y_train, train_mask = Model16(prediction, model_config['t'], y_train, train_mask)
        model_config = model_config['Model_to_predict']
        print('',
              'name           : {}'.format(model_config['name']),
              'logdir         : {}'.format(model_config['logdir']),
              'dataset        : {}'.format(model_config['dataset']),
              'train_size     : {}'.format(model_config['train_size']),
              'learning_rate  : {}'.format(model_config['learning_rate']),
              'feature        : {}'.format(model_config['feature']),
              'logging        : {}'.format(model_config['logging']),
              sep='\n')
    elif model_config['Model'] == 17:
        if model_config['smoothing'] is not None:
            stored_A = None
            adj = construct_knn_graph(features, model_config['k'])
        else:
            stored_A = stored_A + '_A_I'
        if model_config['drop_inter_class_edge']:
            stored_A = None
        test_acc, test_acc_of_class, prediction = Model17(adj, model_config['alpha'], y_train, train_mask, y_test,
                                                          stored_A=stored_A)
        print("Test set results: accuracy= {:.5f}".format(test_acc))
        print("accuracy of each class=", test_acc_of_class)
        print("Total time={}s".format(time.time()-very_begining))
        return test_acc, test_acc_of_class, prediction, size_of_each_class, time.time()-very_begining
    elif model_config['Model'] == 18:
        y_train, train_mask = Model9(adj, model_config['t'], model_config['alpha'],
                                 y_train, train_mask, stored_A=stored_A + '_A_I')
        alpha = 1e-6
        test_acc, test_acc_of_class, prediction = Model17(adj, alpha, y_train, train_mask, y_test,
                                                          stored_A=stored_A + '_A_I')
        print("Test set results: accuracy= {:.5f}".format(test_acc))
        print("accuracy of each class=", test_acc_of_class)
        return test_acc, test_acc_of_class, prediction
    elif model_config['Model'] == 19:
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=model_config['threads'])) as sub_sess:
                tf.set_random_seed(seed)
                test_acc, test_acc_of_class, prediction = train(model_config['Model_to_add_label'], sub_sess, seed, data_split=data_split)
        stored_A = stored_A + '_A_I'
        # print(time.time()-very_begining)
        y_train, train_mask = Model19(prediction, model_config['t'], y_train, train_mask, adj, model_config['alpha'], stored_A, model_config['Model19'])
        # print(time.time()-very_begining)
        model_config = model_config['Model_to_predict']
        print('',
              'name           : {}'.format(model_config['name']),
              'logdir         : {}'.format(model_config['logdir']),
              'dataset        : {}'.format(model_config['dataset']),
              'train_size     : {}'.format(model_config['train_size']),
              'learning_rate  : {}'.format(model_config['learning_rate']),
              'feature        : {}'.format(model_config['feature']),
              'logging        : {}'.format(model_config['logging']),
              sep='\n')
    elif model_config['Model'] == 20:
        pass
    elif model_config['Model'] == 21:
        pass
    elif model_config['Model'] == 22:
        alpha = model_config['alpha']
        stored_A = stored_A + '_A_I'
        features = Model22(adj, features, alpha, stored_A)
    elif model_config['Model'] == 23:
        if model_config['classifier'] == 'tree':
            clf = tree.DecisionTreeClassifier(max_depth=model_config['tree_depth'])
            t = time.time()
            clf.fit(features[train_mask], np.argmax(y_train[train_mask], axis=1))
            t = time.time()-t
            prediction = clf.predict(features[test_mask])
        elif model_config['classifier'] == 'svm':
            clf = svm.SVC()#kernel='rbf', gamma=model_config['gamma'], class_weight='balanced', degree=model_config['svm_degree'])
            t = time.time()
            clf.fit(features[train_mask], np.argmax(y_train[train_mask], axis=1))
            t = time.time()-t
            prediction = clf.predict(features[test_mask])
        elif model_config['classifier'] == 'cnn':
            prediction, t = cnn.train(model_config, features, train_mask, y_train, test_mask, y_test)
        else:
            raise ValueError("model_config['classifier'] should be in ['svm', 'tree']")
        test_acc = np.sum(prediction == np.argmax(y_test[test_mask], axis=1))/np.sum(test_mask)
        # test_acc = test_acc[0]
        one_hot_prediction = np.zeros(y_test[test_mask].shape)
        one_hot_prediction[np.arange(one_hot_prediction.shape[0]), prediction] = 1
        test_acc_of_class = np.sum(one_hot_prediction*y_test[test_mask], axis=0)/np.sum(y_test[test_mask], axis=0) #TODO
        print("Test set results: cost= {:.5f} accuracy= {:.5f} time= {:.5f}".format(0.,test_acc,0.))
        print("accuracy of each class=", test_acc_of_class)
        print("Total time={}s".format(time.time()-very_begining))
        return test_acc, test_acc_of_class, prediction, size_of_each_class, t
    elif model_config['Model'] == 26:
        adj = Model26(adj, model_config['t'], model_config['alpha'],
                                     y_train, train_mask, stored_A = stored_A+'_A_I')
    elif model_config['Model'] == 28:
        features = Model28(adj, features, stored_A, model_config['k'])
    else:
        raise ValueError(
            '''model_config['Model'] must be in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,'''
            ''' 11, 12, 13, 14, 15, 16, 17, 18], but is {} now'''.format(model_config['Model']))

    # Some preprocessing
    if model_config['connection'] == ['f' for i in range(len(model_config['connection']))]:
        train_features = features[train_mask]
        val_features = features[val_mask]
        test_features = features[test_mask]
    else:
        train_features = features
        val_features = features
        test_features = features
    if sparse.issparse(features):
            train_features = sparse_to_tuple(train_features)
            val_features = sparse_to_tuple(val_features)
            test_features = sparse_to_tuple(test_features)
            features = sparse_to_tuple(features)

    if model_config['Model'] == 12:
        if model_config['k'] < 0:
            if hasattr(model_config['train_size'], '__getitem__'):
                eta = 0
                for i in model_config['train_size']:
                    eta += i
                eta /= adj.shape[0]
            else:
                eta = model_config['train_size']/100
            k = (1/eta) ** (1/len(model_config['connection']))
            k = int(k)
        else:
            k = model_config['k']
        model_config['name'] += '_k{}'.format(k)
        support = Model12(adj, k)
        num_supports = len(support)
    elif model_config['conv'] == 'taubin':
        support = [sparse_to_tuple(taubin_smoothor(adj, model_config['taubin_lambda'], model_config['taubin_mu'], model_config['taubin_repeat']))]
        num_supports = 1
    elif model_config['conv'] == 'test21':
        support = [sparse_to_tuple(Test21(adj, model_config['alpha'], beta=model_config['beta'], stored_A=stored_A + '_A_I'))]
        num_supports = 1
    elif model_config['conv'] == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
    elif model_config['conv'] == 'gcn_unnorm':
        support = [sparse_to_tuple(adj.astype(np.float32))]
        num_supports = 1
    elif model_config['conv'] == 'gcn_noloop':
        support = [preprocess_adj(adj, loop=False)]
        num_supports = 1
    elif model_config['conv'] =='gcn_rw':
        support = [preprocess_adj(adj, type='rw')]
        num_supports = 1
    elif model_config['conv'] in ['cheby', 'chebytheta']:
        # origin_adj_support = chebyshev_polynomials(origin_adj, model_config['max_degree'])
        support = chebyshev_polynomials(adj, model_config['max_degree'])
        num_supports = 1 + model_config['max_degree']
    else:
        raise ValueError('Invalid argument for model_config["conv"]: ' + str(model_config['conv']))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support' + str(i)) for i in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, name='features') if isinstance(features, tf.SparseTensorValue) else tf.placeholder(tf.float32, shape=[None, features.shape[1]], name='features'),
        'labels': tf.placeholder(tf.int32, name='labels', shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32, name='labels_mask'),
        'dropout': tf.placeholder_with_default(0., name='dropout', shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32, name='num_features_nonzero'),
        # helper variable for sparse dropout
        'laplacian' : tf.SparseTensor(indices=np.vstack([laplacian.row, laplacian.col]).transpose()
                                      , values=laplacian.data, dense_shape=laplacian.shape),
        'triplet': tf.placeholder(tf.int32, name='triplet', shape=(None, None)),
        'noise_sigma': tf.placeholder(tf.float32, name='noise_sigma'),
        'noise'     : tf.sparse_placeholder(tf.float32, name='features') if isinstance(features, tf.SparseTensorValue) else tf.placeholder(tf.float32, shape=[None, features.shape[1]], name='features')
    }
    if model_config['Model'] in [11, 13, 14, 15]:
        placeholders['label_per_sample'] = tf.placeholder(tf.float32, name='label_per_sample', shape=(None, label_per_sample.shape[1]))
        placeholders['sample2label'] = tf.placeholder(tf.float32, name='sample2label', shape=(label_per_sample.shape[1], y_train.shape[1]))

    # Create model
    model = GCN_MLP(model_config, placeholders, input_dim=train_features[2][1])

    # Random initialize
    sess.run(tf.global_variables_initializer())

    # Initialize FileWriter, saver & variables in graph
    train_writer = None
    valid_writer = None
    saver = None

    # Construct feed dictionary
    if model_config['connection'] == ['f' for i in range(len(model_config['connection']))]:
        train_feed_dict = construct_feed_dict(
            train_features, support, y_train[train_mask], np.ones(train_mask.sum(), dtype=np.bool),
            triplet, model_config['noise_sigma'], placeholders)
        train_feed_dict.update({placeholders['dropout']: model_config['dropout']})
        valid_feed_dict = construct_feed_dict(
            val_features, support, y_val[val_mask],
            np.ones(val_mask.sum(), dtype=np.bool), triplet, 0, placeholders)
        test_feed_dict = construct_feed_dict(
            test_features, support, y_test[test_mask],
            np.ones(test_mask.sum(), dtype=np.bool), triplet, 0, placeholders)
    else:
        train_feed_dict = construct_feed_dict(train_features, support, y_train, train_mask,
                                              triplet, model_config['noise_sigma'], placeholders)
        train_feed_dict.update({placeholders['dropout']: model_config['dropout']})
        valid_feed_dict = construct_feed_dict(val_features, support, y_val, val_mask, triplet, 0, placeholders)
        test_feed_dict = construct_feed_dict(test_features, support, y_test, test_mask, triplet, 0, placeholders)

    if model_config['Model'] in [11, 13, 14, 15]:
        train_feed_dict.update({placeholders['label_per_sample']: label_per_sample})
        train_feed_dict.update({placeholders['sample2label']: sample2label})
        valid_feed_dict.update({placeholders['label_per_sample']: label_per_sample})
        valid_feed_dict.update({placeholders['sample2label']: sample2label})
        test_feed_dict.update({placeholders['label_per_sample']: label_per_sample})
        test_feed_dict.update({placeholders['sample2label']: sample2label})

    # tmp = sess.run([model.prediction, model.sample2label], feed_dict=test_feed_dict)

    # Some support variables
    valid_loss_list = []
    max_valid_acc = 0
    max_train_acc = 0
    t_test = time.time()
    test_cost, test_acc, test_acc_of_class, prediction = sess.run([model.loss, model.accuracy, model.accuracy_of_class, model.prediction], feed_dict=test_feed_dict)
    test_duration = time.time() - t_test
    timer = 0
    begin = time.time()

    # print(time.time() - very_begining)
    if model_config['train']:
        # Train model
        print('training...')
        for step in range(model_config['epochs']):
            if model_config['Model']in [20, 21] and step == model_config['epochs']/2:
                stored_A = stored_A + '_A_I'
                y_train, train_mask = Model20(prediction, model_config['t'], y_train, train_mask, adj,
                                              model_config['alpha'], stored_A)
                if model_config['Model'] == 21:
                    y_train, train_mask = Model16(prediction, model_config['t2'], y_train, train_mask)
                train_feed_dict = construct_feed_dict(features, support, y_train, train_mask,
                                                      model_config['noise_sigma'], placeholders)
                train_feed_dict.update({placeholders['dropout']: model_config['dropout']})
                max_valid_acc = 0
                max_train_acc = 0

            # Training step
            if model_config['logdir'] and step % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                t = time.time()
                sess.run(model.opt_op, feed_dict=train_feed_dict, options=run_options, run_metadata=run_metadata)
                t = time.time()-t
                train_writer.add_run_metadata(run_metadata, 'step%d' % step)
                # Create the Timeline object, and write it to a json
                with open(path.join(model_config['logdir'], 'timeline.json'), 'w') as f:
                    f.write(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
            else:
                t = time.time()
                if isinstance(train_features, tf.SparseTensorValue):
                    train_feed_dict.update({placeholders['features']:
                        tf.SparseTensorValue(train_features.indices,
                                             train_features.values + np.random.normal(0,
                                                    model_config['noise_sigma'], train_features.indices.shape[0]),
                                             train_features.dense_shape)})
                else:
                    train_feed_dict.update({placeholders['features']:
                        train_features+np.random.normal(0,model_config['noise_sigma'],train_features.shape)})
                sess.run(model.opt_op, feed_dict=train_feed_dict)
                t = time.time()-t
            timer += t
            train_loss, train_acc, train_summary = sess.run([model.loss, model.accuracy, model.summary],
                                                            feed_dict=train_feed_dict)

            # Logging
            if model_config['logdir']:
                global_step = model.global_step.eval(session=sess)
                train_writer.add_summary(train_summary, global_step)
                valid_writer.add_summary(valid_summary, global_step)

            # If it's best performence so far, evalue on test set
            if model_config['validate']:
                valid_loss, valid_acc, valid_summary = sess.run(
                    [model.loss, model.accuracy, model.summary],
                    feed_dict=valid_feed_dict)
                valid_loss_list.append(valid_loss)
                if valid_acc >= max_valid_acc:
                    max_valid_acc = valid_acc
                    t_test = time.time()
                    test_cost, test_acc, test_acc_of_class = sess.run(
                        [model.loss, model.accuracy, model.accuracy_of_class],
                        feed_dict=test_feed_dict)
                    test_duration = time.time() - t_test
                    prediction = sess.run(model.prediction,train_feed_dict)
                    if args.verbose:
                        print('*', end='')
            else:
                if train_acc >= max_train_acc:
                    max_train_acc = train_acc
                    t_test = time.time()
                    test_cost, test_acc, test_acc_of_class = sess.run(
                        [model.loss, model.accuracy, model.accuracy_of_class],
                        feed_dict=test_feed_dict)
                    test_duration = time.time() - t_test
                    prediction = sess.run(model.prediction,train_feed_dict)
                    if args.verbose:
                        print('*', end='')

            # Print results
            if args.verbose:
                print("Epoch: {:04d}".format(step),
                      "train_loss= {:.3f}".format(train_loss),
                      "train_acc= {:.3f}".format(train_acc), end=' ')
                if model_config['validate']:
                    print(
                      "val_loss=", "{:.3f}".format(valid_loss),
                      "val_acc= {:.3f}".format(valid_acc),end=' ')
                print("time=", "{:.5f}".format(t))

            if 0 < model_config['early_stopping'] < step \
                    and valid_loss_list[-1] > np.mean(valid_loss_list[-(model_config['early_stopping'] + 1):-1]):
                print("Early stopping...")
                break
        else:
            print("Optimization Finished!")

        # Testing
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        print("accuracy of each class=", test_acc_of_class)

        # Saving
        if model_config['logdir']:
            print('Save model to "{:s}"'.format(saver.save(
                sess=sess,
                save_path=path.join(model_config['logdir'], 'model.ckpt'),
                global_step=global_step)))
    print("Total time={}s".format(time.time()-very_begining))
    return test_acc, test_acc_of_class, prediction, size_of_each_class, time.time()-begin


if __name__ == '__main__':

    acc = [[] for i in configuration['model_list']]
    acc_of_class = [[] for i in configuration['model_list']]
    duration = [[] for i in configuration['model_list']]
    # Read configuration
    for r in range(configuration['repeating']):
        for model_config, i in zip(configuration['model_list'], range(len(configuration['model_list']))):
            # Set random seed
            seed = model_config['random_seed']
            np.random.seed(seed)
            model_config['random_seed'] = np.random.random_integers(1073741824)

            # Initialize session
            with tf.Graph().as_default():
                tf.set_random_seed(seed)
                with tf.Session(config=tf.ConfigProto(
                        intra_op_parallelism_threads=model_config['threads'])) as sess:
                    test_acc, test_acc_of_class, prediction, size_of_each_class, t = train(model_config, sess, seed, r)
                    acc[i].append(test_acc)
                    acc_of_class[i].append(test_acc_of_class)
                    duration[i].append(t)
        print('repeated ', r, 'rounds')

    acc_means = np.mean(acc, axis=1)
    acc_stds = np.std(acc, axis=1)
    acc_of_class_means = np.mean(acc_of_class, axis=1)
    duration = np.mean(duration, axis=1)
    # print mean, standard deviation, and model name
    print()
    pprint.pprint(acc)
    print("REPEAT\t{}".format(configuration['repeating']))
    print("{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format('DATASET', 'train_size', 'valid_size', 'RESULTS', 'STD', 'TRAIN_TIME', 'NAME'))
    for model_config, acc_mean, acc_std, t in zip(configuration['model_list'], acc_means, acc_stds, duration):
        print("{:<8}\t{:<8}\t{:<8}\t{:<8.6f}\t{:<8.6f}\t{:<8.2f}\t{:<8}".format(model_config['dataset'],
                                                                          str(model_config['train_size']) + ' per class',
                                                                          str(model_config['validation_size']),
                                                                          acc_mean,
                                                                          acc_std,
                                                                          t,
                                                                          model_config['name']))

    for model_config, acc_of_class_mean in zip(configuration['model_list'], acc_of_class_means):
        print(str(size_of_each_class)+' ', end='')
        print('[',end='')
        for acc_of_class in acc_of_class_mean:
            print('{:0<5.3}'.format(acc_of_class),end=', ')
        print(']',end='')
        print('\t{:<8}'.format(model_config['name']))
