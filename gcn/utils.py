from __future__ import print_function

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import sys
from os import path
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import random
import tensorflow as tf
# import matplotlib.pyplot as plt

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_triplet(y_train, train_mask, max_triplets):
#    print('y_train----',y_train.shape)        
    index_nonzero = y_train.nonzero()
#    for i in range(y_train.shape[1]):
#        label_count.append(index_nonzero[1][[index_nonzero[1]==i]].size)
    label_count = np.sum(y_train, axis=0)
    all_count = np.sum(label_count)
    
    index_nonzero = np.transpose(np.concatenate((index_nonzero[0][np.newaxis,:], index_nonzero[1]\
                                                 [np.newaxis, :]),axis=0)).tolist()
        
    index_nonzero = sorted(index_nonzero, key = lambda s: s[1])
    #print(index_nonzero)
    #print(label_count)
 
    def get_one_triplet(input_index, index_nonzero, label_count, all_count, max_triplets):
        triplet = []
        if label_count[input_index[1]]==0:
            return 0
        else:
 #           print('max_triplets', max_triplets)
  #          print(all_count)
   #         print(label_count[input_index[1]])
            n_triplets = min(max_triplets, int(all_count-label_count[input_index[1]]))
   #         print('----------')

            for j in range(int(label_count[input_index[1]])-1):
                positives = []
                negatives = []           
                for k, (value, label) in enumerate(index_nonzero):
                    #find a postive sample, and if only one sample then choose itself
                    if label == input_index[1] and (value != input_index[0] or label_count[input_index[1]]==1):
                        positives.append(index_nonzero[k])
                    if label != input_index[1]:
                        negatives.append(index_nonzero[k])
 #               print('positives' ,positives)
 #               print('negatives', negatives)
                negatives = random.sample(list(negatives), n_triplets)
                for value, label in negatives:
                    triplet.append([input_index[0], positives[j][0], value])
            return triplet
                
                                   
    triplet = []
    for i, j in enumerate(index_nonzero):
        triple = get_one_triplet(j, index_nonzero, label_count, all_count,max_triplets)
        
        if triple == 0:
            continue
        else:
            triplet.extend(triple)  
    np_triple = np.concatenate(np.array([triplet]), axis = 1)
    return np_triple

def load_data(dataset_str, train_size, validation_size, model_config, shuffle=True, repeat_state=None):
    """Load data."""
    if dataset_str in ['USPS-Fea', 'CIFAR-Fea', 'Cifar_10000_fea', 'Cifar_R10000_fea', 'MNIST-Fea',
                       'MNIST-10000', 'MNIST-5000', 'USPS-2-100', 'USPS-2-10']:
        data = sio.loadmat('data/{}.mat'.format(dataset_str))
        l = data['labels'].flatten()
        labels = np.zeros([l.shape[0],np.max(data['labels'])+1])
        labels[np.arange(l.shape[0]), l.astype(np.int8)] = 1
        features = data['X']
        sample = features[0].copy()
        adj = data['G']
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        # if dataset_str == 'citeseer':
        #     # Fix citeseer dataset (there are some isolated nodes in the graph)
        #     # Find isolated nodes, add them as zero-vecs into the right position
        #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        #     tx_extended[test_idx_range - min(test_idx_range), :] = tx
        #     tx = tx_extended
        #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        #     ty_extended[test_idx_range - min(test_idx_range), :] = ty
        #     ty = ty_extended
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1] - 1))
            ty_extended_ = np.ones((len(test_idx_range_full), 1))  # add dummy labels
            ty_extended = np.hstack([ty_extended, ty_extended_])
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        # features = sp.eye(features.shape[0]).tolil()
        # features = sp.lil_matrix(allx)

        labels = np.vstack((ally, ty))
        # labels = np.vstack(ally)

        if dataset_str.startswith('nell'):
            # Find relation nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(allx.shape[0], len(graph))
            isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - allx.shape[0], :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - allx.shape[0], :] = ty
            ty = ty_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

            idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

            if not os.path.isfile("data/planetoid/{}.features.npz".format(dataset_str)):
                print("Creating feature vectors for relations - this might take a while...")
                features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                              dtype=np.int32).todense()
                features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
                features = sp.csr_matrix(features_extended, dtype=np.float32)
                print("Done!")
                save_sparse_csr("data/planetoid/{}.features".format(dataset_str), features)
            else:
                features = load_sparse_csr("data/planetoid/{}.features.npz".format(dataset_str))

            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = preprocess_features(features, feature_type=model_config['feature'])

    global all_labels
    all_labels = labels.copy()

    # split the data set
    idx = np.arange(len(labels))
    if shuffle:
        np.random.shuffle(idx)
    test_size = model_config['test_size']
    if isinstance(train_size, int):
        assert train_size>0, "train size must bigger than 0."
        no_class = labels.shape[1]  # number of class
        train_size = [train_size for i in range(labels.shape[1])]
        idx_train = []
        count = [0 for i in range(no_class)]
        label_each_class = train_size
        next = 0
        for i in idx:
            if count == label_each_class:
                break
            next += 1
            for j in range(no_class):
                if labels[i, j] and count[j] < label_each_class[j]:
                    idx_train.append(i)
                    count[j] += 1
                    break

        if model_config['validate']:
            if test_size:
                assert next+validation_size<len(idx)
            assert next < len(idx), "Too many train data, no data left for validation."
            idx_val = idx[next:next+validation_size]
            next = next+validation_size
            assert next+test_size < len(idx)
            assert next < len(idx), "Too many train and validation data, no data left for testing."
            idx_test = idx[-test_size:] if test_size else idx[next+validation_size:]
        else:
            if test_size:
                assert next+test_size<len(idx)
            assert next < len(idx), "Too many train data, no data left for testing."
            idx_val = idx[-test_size:] if test_size else idx[next:]
            idx_test = idx[-test_size:] if test_size else idx[next:]
    else:
        # train
        assert isinstance(train_size, float)
        assert 0<train_size<1, "float train size must be between 0-1"
        labels_of_class = [0]
        train_size = int(len(idx) * train_size)
        next = 0
        while (np.prod(labels_of_class) == 0):
            np.random.shuffle(idx)
            idx_train = idx[next:next+train_size]
            labels_of_class = np.sum(labels[idx_train], axis=0)
        next = train_size

        # validate
        if model_config['validate']:
            assert isinstance(validation_size, float)
            validation_size = int(len(idx) * validation_size)
            idx_val = idx[next: next+validation_size]
            next += validation_size
        else:
            idx_val = idx[next:]

        # test
        if test_size:
            assert isinstance(test_size, float)
            test_size = int(len(idx) * test_size)
            idx_test = idx[next: next+test_size]
        else:
            idx_test = idx[next:]

    if dataset_str in ['USPS-2-100', 'USPS-2-10']:
        assert model_config['validate'] == False
        splits = data['idxLabs'].shape[0]
        k = repeat_state%splits if shuffle else np.random.randint(splits)
        idx_train = data['idxLabs'][k]
        idx_test = data['idxUnls'][k]

    print('labels of each class : ', np.sum(labels[idx_train], axis=0))
    # idx_val = idx[len(idx) * train_size // 100:len(idx) * (train_size // 2 + 50) // 100]
    # idx_test = idx[len(idx) * (train_size // 2 + 50) // 100:len(idx)]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    # else:
    #     idx_test = test_idx_range.tolist()
    #     idx_train = range(len(y))
    #     idx_val = range(len(y), len(y) + 500)
    #
    #     train_mask = sample_mask(idx_train, labels.shape[0])
    #     val_mask = sample_mask(idx_val, labels.shape[0])
    #     test_mask = sample_mask(idx_test, labels.shape[0])
    #
    #     y_train = np.zeros(labels.shape)
    #     y_val = np.zeros(labels.shape)
    #     y_test = np.zeros(labels.shape)
    #     y_train[train_mask, :] = labels[train_mask, :]
    #     y_val[val_mask, :] = labels[val_mask, :]
    #     y_test[test_mask, :] = labels[test_mask, :]

    size_of_each_class = np.sum(labels[idx_train], axis=0)
    if model_config['loss_func'] == 'triplet':
        triplet = get_triplet(y_train, train_mask, model_config['max_triplet'])
    else:
        triplet = []
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, size_of_each_class, triplet


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return tf.SparseTensorValue(coords, values, np.array(shape, dtype=np.int64))

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, feature_type):
    if feature_type == 'bow':
        # """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        # normalize(features, norm='l1', axis=1, copy=False)
    elif feature_type == 'tfidf':
        transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
        features = transformer.fit_transform(features)
    elif feature_type == 'none':
        features = sp.csr_matrix(sp.eye(features.shape[0]))
    else:
        raise ValueError('Invalid feature type: ' + str(feature_type))
    return features


def normalize_adj(adj, type='sym'):
    """Symmetrically normalize adjacency matrix."""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        # d_inv_sqrt = np.power(rowsum, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)  #
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], laplacian))

    return sparse_to_tuple(t_k)


def Model1(W, s, alpha, absorption_type):
    count = np.sum(W, axis=1).flatten() + 1
    L = np.diag(W.sum(1).flat) - W
    L = L + alpha * np.eye(W.shape[0])
    # print(time.time())
    A = np.array(np.linalg.inv(L))
    # print(time.time())
    sorted = -np.sort(-A, axis=1)
    if s == -1:
        gate = sorted[np.arange(sorted.shape[0]), count]
    elif s > 0:
        gate = sorted[:, s]
    else:
        raise ValueError('s must be -1 or 0 or positive number, but is {} now'.format(s))

    gate = np.reshape(gate, [-1, 1])
    if absorption_type == 'binary':
        A = np.array(A > gate, dtype=np.int64)
    elif absorption_type == 'weighted':
        A[A <= gate] = 0
    else:
        raise ValueError(
            "'absorption_type' must be 'weighted' or 'binary', but is {} now".format(repr(absorption_type)))
    # adj += 10*np.eye(adj.shape[0], dtype=np.int64)
    return sp.csr_matrix(A)


def Model2(W, s, alpha, y_train):
    W = W.copy().astype(np.float32)
    L = np.diag(W.sum(1).flat) - W
    A = np.array(np.linalg.inv(L + alpha * np.eye(W.shape[0])))
    already_labeled = np.sum(y_train, axis=1)
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        a[W.dot(y) > 0] = 0
        sorted = -np.sort(-a, axis=0)
        gate = (-np.sort(-a, axis=0))[s]
        tmp_a = np.array(a > gate, dtype=np.int32)
        tmp_b = np.array(y, dtype=np.int32)
        indicator = np.zeros(W.shape)
        indicator[:, a.flat > gate] += 1
        indicator[y.flat > 0, :] += 1
        W[indicator > 1.5] = 1
        # index = y_train[:, i]
        # index = np.where(index > 0)
        # indicator = np.zeros(W.shape)
        # indicator[:, index] += 1
        # indicator[index, :] += 1
        # W[indicator > 1.5]  = 1
    W = W + W.T
    W[W > 0] = 1
    return W


def Model3(W, s, alpha, y_train, train_mask):
    W = W.copy().astype(np.float32)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    L = np.diag(W.sum(1).flat) - W
    A = np.array(np.linalg.inv(L + alpha * np.eye(W.shape[0])))
    already_labeled = np.sum(y_train, axis=1)
    print("Additional Label:")
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[s]
        y_train[a.flat > gate, i] = 1
        train_index = np.hstack([train_index, np.where(a.flat > gate)[0]])
        correct_label_count(a.flat > gate, i)
    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def Model4(W, s, alpha, y_train, train_mask):
    W = W.copy().astype(np.float32)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    L = np.diag(W.sum(1).flat) - W
    A = np.array(np.linalg.inv(L + alpha * np.eye(W.shape[0])))
    already_labeled = np.sum(y_train, axis=1)
    print("Additional Label:")
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        neighbor = W.dot(y)
        neighbor_neighbor = W.dot(neighbor)
        a = A.dot(y)
        a[already_labeled > 0] = 0
        a[neighbor > 0] = 0
        a[neighbor_neighbor > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[s]
        y_train[a.flat > gate, i] = 1
        train_index = np.hstack([train_index, np.where(a.flat > gate)[0]])
        correct_label_count(a.flat > gate, i)
    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def drop_inter_class_edge(adj):
    adj_coo = adj.tocoo()
    L = all_labels.shape[1]
    class_pair_indicator = all_labels[adj_coo.row].reshape([-1, L, 1]) * all_labels[adj_coo.col].reshape([-1, 1, L])
    class_count = np.sum(class_pair_indicator, axis=0)
    same_class_indicator = np.trace(class_pair_indicator, axis1=1, axis2=2)
    W = sp.coo_matrix((np.ones(np.sum(same_class_indicator > 0)),
                       (adj_coo.row[same_class_indicator > 0], adj_coo.col[same_class_indicator > 0])), shape=adj.shape)
    adj_coo = W.tocoo()
    class_pair_indicator = all_labels[adj_coo.row].reshape([-1, L, 1]) * all_labels[adj_coo.col].reshape([-1, 1, L])
    class_count = np.sum(class_pair_indicator, axis=0)
    return W


def Model5(features, adj, mu, gate=None):
    # features = sp.lil_matrix([
    #     [1,3],
    #     [2,4],
    #     [5,6],
    # ])
    # adj = sp.csr_matrix([
    #     [0,1,1],
    #     [1,0,1],
    #     [1,1,0],
    # ])
    print('Calculating New Adjacency Matrix...')
    adj_coo = adj.tocoo()
    L = all_labels.shape[1]

    I = sp.eye(adj.shape[0])
    local_weight = I + adj * (I + adj)
    local_weight = local_weight.sqrt()
    local_features = np.array(local_weight.dot(features) / np.sum(local_weight, axis=1))
    D2 = np.sum((local_features[adj_coo.row] - local_features[adj_coo.col]) ** 2, axis=1)
    D = D2 ** 0.5
    d_mean = np.mean(D)

    class_pair_indicator = all_labels[adj_coo.row].reshape([-1, L, 1]) * all_labels[adj_coo.col].reshape([-1, 1, L])
    # class_mean_distance = class_pair_indicator * D.reshape([-1, 1, 1])
    # bins = np.histogram(D, bins='auto')[1][::1]
    # # bins = np.arange(100)**2/1000
    # # bins = 4
    # # bins = np.arange(bins+1)*(D.max()-D.min())/bins+D.min()
    # class_hist = np.array([[np.histogram(class_mean_distance[class_pair_indicator[:,i,j]>0,i,j], bins=bins)[0]
    #                         for j in range(L)] for i in range(L)])
    # bins_sum = np.sum(class_hist.reshape([L * L, -1]), axis=0)
    # accumulation = np.add.accumulate(bins_sum)/np.sum(bins_sum)
    # same_class = np.trace(class_hist, axis1=0, axis2=1)
    # ratio_of_same_class = same_class / bins_sum
    # same_class_accumulation = np.add.accumulate(same_class)/np.sum(same_class)
    # different_class = bins_sum-same_class
    # different_class_accumulation = np.add.accumulate(different_class)/np.sum(different_class)
    #
    # class_mean_distance = np.sum(class_mean_distance, axis=0)
    class_count = np.sum(class_pair_indicator, axis=0)
    # class_mean_distance /= class_count
    # closity = np.trace(class_mean_distance)/np.sum(class_mean_distance)*class_mean_distance.shape[0]
    #
    #
    # import matplotlib.pyplot as plt
    # plt.plot(ratio_of_same_class)
    # plt.plot(accumulation, label='total')
    # plt.plot(same_class_accumulation, label='inner')
    # plt.plot(different_class_accumulation, label='inter')
    # plt.legend(loc='lower right')
    # plt.show()
    # plt.hist(D2,bins='auto')
    # plt.hist(D2/d_mean,bins='auto')
    # plt.hist(D2/d_mean/d_mean,bins='auto')
    # plt.show()

    if gate:
        D = sp.coo_matrix((D, (adj_coo.row, adj_coo.col)))
        W = adj.copy()
        W[D > gate] = 0
    else:
        D2 = sp.coo_matrix((D2, (adj_coo.row, adj_coo.col)))
        W = (-D2 / d_mean / d_mean / mu).expm1() + adj
    return W


def Model6(adj):
    adj = adj.tocoo()
    new_adj = []
    for i, j in zip(adj.row, adj.col):
        y_i = np.where(all_labels[i])[0]
        y_j = np.where(all_labels[j])[0]
        new_adj.append(1 if y_i == y_j else 0)
    return sp.coo_matrix((new_adj, (adj.row, adj.col)))


def Model7(W, s, alpha, y_train, train_mask, features):
    W = W.copy().astype(np.float32)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    L = np.diag(W.sum(1).flat) - W
    A = np.array(np.linalg.inv(L + alpha * np.eye(W.shape[0])))
    already_labeled = np.sum(y_train, axis=1)
    additional_train_index = []
    additional_train_lable = []
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[s]
        additional_train_index.append(np.where(a.flat > gate)[0])
        additional_train_lable.append(np.ones(additional_train_index[i].shape[0]) * i)
        # y_train[a.flat > gate, i] = 1
        # train_index = np.hstack([train_index, additional_train_index[i]])
        # correct_label_count(a.flat > gate, i)
    additional_train_index = np.hstack(additional_train_index)
    additional_train_lable = np.hstack(additional_train_lable).astype(np.int64)
    # additional_train_index = np.array([0, 1, 2, 3, 4])
    # additional_train_lable = np.array([0, 1, 2, 3, 4])
    class_distance = []
    features = features.toarray()
    for i in range(y_train.shape[1]):
        x1 = features[additional_train_index, :].reshape((-1, 1, features.shape[1]))
        x2 = features[y_train[:, i].astype(np.bool)].reshape((1, -1, features.shape[1]))
        D = np.sum((x1 - x2) ** 2, axis=2) ** 0.5
        class_distance.append(np.mean(D, axis=1, keepdims=True))
    class_distance = np.hstack(class_distance)
    mean_distance_class = np.argmin(class_distance, axis=1)
    additional_train_index = additional_train_index[additional_train_lable == mean_distance_class]
    additional_train_lable = additional_train_lable[additional_train_lable == mean_distance_class]
    y_train[additional_train_index, additional_train_lable] = 1
    train_index = np.hstack([train_index, additional_train_index])
    train_mask = sample_mask(train_index, y_train.shape[0])
    print("Additional Label:")
    for i in range(y_train.shape[1]):
        index = additional_train_index[additional_train_lable == i]
        indicator = np.zeros(y_train.shape[0])
        indicator[index] = 1
        indicator = indicator.astype(np.bool)
        correct_label_count(indicator, i)
    return y_train, train_mask


def Model8(W, s, alpha, y_train, train_mask):
    W = W.copy().astype(np.float32)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    L = np.diag(W.sum(1).flat) - W
    A = np.array(np.linalg.inv(L + alpha * np.eye(W.shape[0])))
    already_labeled = np.sum(y_train, axis=1)
    print("Additional Label:")
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[s]
        indicator = a.flat > gate
        indicator = indicator * all_labels[:, i].astype(np.bool)
        y_train[indicator, i] = 1
        train_index = np.hstack([train_index, np.where(indicator)[0]])
        correct_label_count(indicator, i)
    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def absorption_probability(W, alpha, stored_A=None, column=None):
    try:
        # raise Exception('DEBUG')
        A = np.load(stored_A + str(alpha) + '.npz')['arr_0']
        print('load A from ' + stored_A + str(alpha) + '.npz')
        if column is not None:
            P = np.zeros(W.shape)
            P[:, column] = A[:, column]
            return P
        else:
            return A
    except:
        # W=sp.csr_matrix([[0,1],[1,0]])
        # alpha = 1
        n = W.shape[0]
        print('Calculate absorption probability...')
        W = W.copy().astype(np.float32)
        D = W.sum(1).flat
        L = sp.diags(D, dtype=np.float32) - W
        L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
        L = sp.csc_matrix(L)
        # print(np.linalg.det(L))

        if column is not None:
            A = np.zeros(W.shape)
            # start = time.time()
            A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
            # print(time.time()-start)
            return A
        else:
            # start = time.time()
            A = slinalg.inv(L).toarray()
            # print(time.time()-start)
            if stored_A:
                np.savez(stored_A + str(alpha) + '.npz', A)
            return A
            # fletcher_reeves

            # slinalg.solve(L, np.ones(L.shape[0]))
            # A_ = np.zeros(W.shape)
            # I = sp.eye(n)
            # Di = sp.diags(np.divide(1,np.array(D)+alpha))
            # for i in range(10):
            #     # A_=
            #     A_ = Di*(I+W.dot(A_))
            # print(time.time()-start)


def gaussian_seidel(A, B):
    X = np.ones(B.shape)
    D = A.diagonal().reshape(A.shape[0], 1)
    R = A.copy()
    R.setdiag(0)
    for i in range(10000):
        # X = R.dot(X)
        # X = B-X
        # X = X/D
        X = (B - R.dot(X)) / D
    return X


def fletcher_reeves(A, B):
    # A=np.array(A)
    X = np.zeros(B.shape)
    r = np.array(B - A.dot(X))
    rsold = (r * r).sum(0)
    p = r
    for i in range(10):
        Ap = np.array(A.dot(p))
        pAp = (p * Ap).sum(0)
        alpha = rsold / pAp
        X += alpha * p
        r -= alpha * Ap
        rsnew = (r * r).sum(0)
        if True:
            pass
        p = r + rsnew / rsold * p
        rsold = rsnew
    return X


def Model9(W, t, alpha, y_train, train_mask, stored_A=None):
    A = absorption_probability(W, alpha, stored_A, train_mask)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    # if not isinstance(features, np.ndarray):
    #     features = features.toarray()
    print("Additional Label:")
    if not hasattr(t, '__getitem__'):
        t = [t for _ in range(y_train.shape[1])]
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]

        # x1 = features[index, :].reshape((-1, 1, features.shape[1]))
        # x2 = features[y_train[:, i].astype(np.bool)].reshape((1, -1, features.shape[1]))
        # D = np.sum((x1 - x2) ** 2, axis=2) ** 0.5
        # D = np.mean(D, axis=1)
        # gate = 100000000 if t[i] >= D.shape[0] else np.sort(D, axis=0)[t[i]]
        # index = index[D<gate]
        train_index = np.hstack([train_index, index])
        y_train[index, i] = 1
        correct_label_count(index, i)

    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def Model10(W, s, t, alpha, y_train, train_mask, features, stored_A=None):
    W = W.copy().astype(np.float32)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    D = W.sum(1).flat
    L = np.diag(D) - W
    d_hat = np.median(D)
    H = D.copy()
    H[H > d_hat] = d_hat
    H = np.diag(H.flat)
    try:
        A = np.load(stored_A + '.npy')
    except:
        A = np.array(np.linalg.inv(L + alpha * H))
        np.save(stored_A, A)

    try:
        A = np.load(stored_A + str(alpha) + '.npy')
    except:
        A = np.array(np.linalg.inv(L + alpha * H))
        if stored_A:
            np.save(stored_A + str(alpha) + '.npy', A)

    already_labeled = np.sum(y_train, axis=1)
    if not isinstance(features, np.ndarray):
        features = features.toarray()
    print("Additional Label:")
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[s]
        index = np.where(a.flat > gate)[0]

        x1 = features[index, :].reshape((-1, 1, features.shape[1]))
        x2 = features[y_train[:, i].astype(np.bool)].reshape((1, -1, features.shape[1]))
        D = np.sum((x1 - x2) ** 2, axis=2) ** 0.5
        D = np.mean(D, axis=1)
        gate = 100000000 if t >= D.shape[0] else np.sort(D, axis=0)[t]
        index = index[D < gate]
        train_index = np.hstack([train_index, index])
        y_train[index, i] = 1
        correct_label_count(index, i)
    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def Model11(y, y_train, train_mask):
    label_per_sample = np.vstack([np.zeros(y), np.eye(y)])[np.add.accumulate(train_mask) * train_mask]
    sample2label = label_per_sample.T.dot(y_train)
    return label_per_sample, sample2label


def Model12(adj, k):
    alpha = 1e-2
    L = np.diag(adj.sum(1).flat) - adj
    P = np.aray(np.linalg.inv(L + alpha * np.eye(adj.shape[0])))
    n = adj.shape[0]

    support = [sp.eye(n)]
    argsort = np.argsort(P, axis=1)
    for i in range(1, k + 1):
        # support[0] += sp.coo_matrix((np.ones(n), (np.arange(n), argsort[:, -i])), shape=(n, n))
        support.append(sp.coo_matrix((np.ones(n), (np.arange(n), argsort[:, -i])), shape=(n, n)))
    return sparse_to_tuple(support)


def Model16(prediction, t, y_train, train_mask):
    new_gcn_index = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    no_class = y_train.shape[1]  # number of class
    if hasattr(t, '__getitem__'):
        assert len(t) >= no_class
        index = []
        count = [0 for i in range(no_class)]
        for i in sorted_index:
            for j in range(no_class):
                if new_gcn_index[i] == j and count[j] < t[j] and not train_mask[i]:
                    index.append(i)
                    count[j] += 1
    else:
        index = sorted_index[:t]
    indicator = np.zeros(train_mask.shape, dtype=np.bool)
    indicator[index] = True
    indicator = np.logical_and(np.logical_not(train_mask), indicator)

    prediction = np.zeros(prediction.shape)
    prediction[np.arange(len(new_gcn_index)), new_gcn_index] = 1.0
    prediction[train_mask] = y_train[train_mask]

    correct_labels = np.sum(prediction[indicator] * all_labels[indicator], axis=0)
    count = np.sum(prediction[indicator], axis=0)
    print('Additiona Label:')
    for i, j in zip(correct_labels, count):
        print(int(i), '/', int(j), sep='', end='\t')
    print()

    y_train = np.copy(y_train)
    train_mask = np.copy(train_mask)
    train_mask[indicator] = 1
    y_train[indicator] = prediction[indicator]
    return y_train, train_mask


def Model17(adj, alpha, y_train, train_mask, y_test, stored_A=None):
    P = absorption_probability(adj, alpha, stored_A=stored_A, column=train_mask)
    P = P[:, train_mask]

    # nearest clssifier
    predicted_labels = np.argmax(P, axis=1)
    # prediction = alpha*P
    prediction = np.zeros(P.shape)
    prediction[np.arange(P.shape[0]), predicted_labels] = 1

    y = np.sum(train_mask)
    label_per_sample = np.vstack([np.zeros(y), np.eye(y)])[np.add.accumulate(train_mask) * train_mask]
    sample2label = label_per_sample.T.dot(y_train)
    prediction = prediction.dot(sample2label)

    test_acc = np.sum(prediction * y_test) / np.sum(y_test)
    test_acc_of_class = np.sum(prediction * y_test, axis=0) / np.sum(y_test, axis=0)
    # print(test_acc, test_acc_of_class)
    return test_acc, test_acc_of_class, prediction


def Model19(prediction, t, y_train, train_mask, W, alpha, stored_A, union_or_intersection):
    no_class = y_train.shape[1]  # number of class

    # gcn index
    new_labels_gcn = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    if not hasattr(t, '__getitem__'):
        t = [t for i in range(no_class)]

    assert len(t) >= no_class
    count = [0 for i in range(no_class)]
    index_gcn = [[] for i in range(no_class)]
    for i in sorted_index:
        j = new_labels_gcn[i]
        if count[j] < t[j] and not train_mask[i]:
            index_gcn[j].append(i)
            count[j] += 1

    # lp
    A = absorption_probability(W, alpha, stored_A, train_mask)
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    index_lp = []
    for i in range(no_class):
        y = y_train[:, i:i + 1]
        a = np.sum(A[:, y.flat > 0], axis=1)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]
        index_lp.append(index)

    # print(list(map(len, index_gcn)))
    # print(list(map(len, index_lp)))

    y_train = y_train.copy()
    print("Additional Label:")
    for i in range(no_class):
        assert union_or_intersection in ['union', 'intersection']
        if union_or_intersection == 'union':
            index = list(set(index_gcn[i]) | set(index_lp[i]))
        else:
            index = list(set(index_gcn[i]) & set(index_lp[i]))
        y_train[index, i] = 1
        train_mask[index] = True
        print(np.sum(all_labels[index, i]), '/', len(index), sep='', end='\t')
    return y_train, train_mask


def Model20(prediction, t, y_train, train_mask, W, alpha, stored_A):
    no_class = y_train.shape[1]  # number of class

    # gcn index
    new_labels_gcn = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    if not hasattr(t, '__getitem__'):
        t = [t for i in range(no_class)]

    # assert len(t) >= no_class
    # count = [0 for i in range(no_class)]
    # index_gcn = [[] for i in range(no_class)]
    # for i in sorted_index:
    #     for j in range(no_class):
    #         if new_labels_gcn[i] == j and count[j] < t[j] and not train_mask[i]:
    #             index_gcn[j].append(i)
    #             count[j] += 1

    predicted_labels = np.argmax(prediction, axis=1)
    prediction = np.zeros(prediction.shape)
    prediction[np.arange(len(predicted_labels)), predicted_labels] = 1.0

    # lp
    A = absorption_probability(W, alpha, stored_A, train_mask)
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    index_lp = []
    for i in range(no_class):
        y = y_train[:, i:i + 1]
        a = A.dot(sp.csc_matrix(y))
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]
        index = index[prediction[index, i] > 0]
        index_lp.append(index)

    # print(list(map(len, index_gcn)))
    # print(list(map(len, index_lp)))
    y_train = y_train.copy()
    print("Additional Label:")
    for i in range(no_class):
        # assert union_or_intersection in ['union', 'intersection']
        # if union_or_intersection == 'union':
        #     index = list(set(index_gcn[i]) | set(index_lp[i]))
        # else:
        #     index = list(set(index_gcn[i]) & set(index_lp[i]))
        index = index_lp[i]
        y_train[index, i] = 1
        train_mask[index] = True
        print(np.sum(all_labels[index, i]), '/', len(index), sep='', end='\t')
    print()
    return y_train, train_mask


def smooth(features, adj, smoothing, model_config, stored_A=None):
    print(smoothing, 'Smoothing...',end='')
    if smoothing is None:
        return features
    if smoothing == 'poly':
        poly_parameters = model_config['poly_parameters']
        adj = normalize_adj(adj + sp.eye(adj.shape[0]), type='rw')
        n = adj.shape[0]
        # adj = adj.copy().astype(np.float32)
        D = adj.sum(1).flat
        L = sp.diags(D) - adj
        new_feature = sp.csr_matrix(np.zeros(features.shape))
        for a in poly_parameters[::-1]:
            new_feature = L.dot(new_feature) + a * features
        return new_feature
    elif smoothing == 'ap':
        return Model22(adj, features, model_config['smooth_alpha'], stored_A)
    elif smoothing == 'taubin':
        return taubin_smoothing(adj, model_config['taubin_lambda'], model_config['taubin_mu'], model_config['taubin_repeat'], features)
    elif smoothing == 'ap_appro':
        k = int(np.ceil(4/model_config['smooth_alpha']))
        return ap_approximate(adj, features, model_config['smooth_alpha'], k)
    elif smoothing == 'test21':
        smoothor = Test21(adj, model_config['smooth_alpha'], model_config['beta'], stored_A)
        features = smoothor * features
        if sp.issparse(features):
            features = features.toarray()
        return features
    elif smoothing == 'test21_norm':
        smoothor = Test21(adj, model_config['smooth_alpha'], model_config['beta'], stored_A)
        features = sp.csr_matrix(smoothor * features)
        return normalize(features, norm='l1', axis=1, copy=False)
    elif smoothing == 'test27':
        return Test27(adj, features, model_config['smooth_alpha'], model_config['beta'], stored_A)
    elif smoothing == 'manifold_denoising':
        return md(adj, features, model_config['smooth_alpha'], model_config['k'], model_config['md_repeat'])
    else:
        raise ValueError("smoothing must be one of 'poly' | 'ap' | 'taubin' | 'test21' | 'test27' ")


def construct_knn_graph(features, k):
    nbrs = NearestNeighbors(n_neighbors=5).fit(features)
    adj = nbrs.kneighbors_graph()
    adj = adj + adj.T
    adj[adj != 0] = 1
    return adj

def md(adj, features, alpha, k, repeat):
    for i in range(repeat):
        features = ap_approximate(adj, features, alpha, int(np.ceil(4/alpha)))
        adj = construct_knn_graph(features, k)
    return adj


def sparse_encoding(features, adj, stored):

    n = adj.shape[0]
    adj_I = adj + sp.diags(np.ones(n))
    adj_I_sym = normalize_adj(adj_I)

    try:
        # raise Exception('DEBUG')
        vals, vecs, vec_inv = np.load(stored+'_I_vals.npy'), \
                              np.load(stored+'_I_vecs.npy'), \
                              np.load(stored + '_I_vec_inv.npy')
        print('load vals, vecs, vec_inv from files')
    except:
        # vals, vecs = slinalg.eigsh(adj_I_sym.toarray(), k=adj_I_sym.shape[0]-1)
        vals, vecs = linalg.eigh(adj_I_sym.toarray())
        vec_inv = vecs.T

        np.save(stored+'_I_vals.npy', vals)
        np.save(stored+'_I_vecs.npy', vecs)
        np.save(stored+'_I_vec_inv.npy', vec_inv)
    c = vec_inv*features
    c_abs = np.abs(c)
    sorted = np.sort(c_abs.flatten())
    # acc = np.add.accumulate(sorted)
    # import matplotlib.pyplot as plt
    # # plt.plot(acc)
    # for i in range(c.shape[1]):
    #     plt.plot(vals, c_abs[:,i], 'o')
    #     plt.show()

    sparsity = 4 # int(n*0.1)
    # print(sorted[-features.shape[1]*sparsity])
    # c[c_abs<sorted[-features.shape[1]*sparsity]]=0
    # features = vecs.dot(c)
    c[:n-200,:]=0
    features = vecs.dot(c)
    return features

def Model22(adj, features, alpha, stored_A=None):
    adj = normalize(adj + sp.eye(adj.shape[0]), 'l1', axis=1)
    if stored_A:
        stored_A += '_r'
    P = absorption_probability(adj, alpha, stored_A=stored_A)
    P *= alpha
    if sp.issparse(features):
        return  P * features
    else:
        return  P.dot(features)


def ap_approximate(adj, features, alpha, k):
    adj = normalize(adj + sp.eye(adj.shape[0]), 'l1', axis=1) / (alpha + 1)
    # D = sp.diags(np.array(adj.sum(axis=1)).flatten())+alpha*sp.eye(adj.shape[0])
    # D = D.power(-1)
    # adj = D*adj
    # features = D*alpha*features
    if sp.issparse(features):
        features = features.toarray()
    new_feature = np.zeros(features.shape)
    for _ in range(k):
        new_feature = adj * new_feature + features
    new_feature *= alpha / (alpha + 1)
    return new_feature

def Test21(adj, alpha, beta, stored_A=None):
    P = absorption_probability(adj + sp.eye(adj.shape[0]), alpha, stored_A=stored_A)
    P *= alpha
    # P = (P > (beta / alpha)).astype(np.float32)
    lines = np.min([100, P.shape[0]])
    idx = np.arange(P.shape[0])
    np.random.shuffle(idx)
    idx = idx[:lines]
    P_flat = P[idx].flat
    P_index = np.argsort(P_flat)
    P_acc = np.add.accumulate(P_flat[P_index])/lines
    percentage = 1-P_acc[-beta*lines]
    # num = np.sum(P_acc <= (1-beta))
    # gate = P_flat[P_index[np.maximum(num-1, 0)]]

    num = beta*lines
    gate = P_flat[P_index[len(P_index)-num]]
    P = (P > [gate]).astype(np.float32)

    global all_labels
    c = np.argmax(all_labels, axis=1)
    c = c == np.expand_dims(c, 1)
    num = np.sum(P)
    print("neighbor accuracy = ", np.sum(c*P)/num,'average #neighbors = ', num/P.shape[0], 'energy reserved=', percentage)
    # normalize(P, norm='l1', axis=1, copy=False)
    return sp.csr_matrix(P/beta)

def Test27(adj, features, alpha, beta, stored_A=None):
    P = absorption_probability(adj + sp.eye(adj.shape[0]), alpha, stored_A=stored_A)
    # np.sort(P)
    P *= alpha
    # P = np.array([
    #     [0.1, 0.5, 0.4],
    #     [0.2, 0.7, 0.1],
    #     [0.4, 0.3, 0.3]
    # ])
    P_index = np.argsort(P, axis=1)
    P_acc = np.add.accumulate(np.sort(P, axis=1), axis=1)
    # plt.plot(np.add.accumulate(np.sort(P, axis=None)))
    # plt.grid()
    # plt.show()
    num = np.sum(P_acc <= (1-beta), axis=1)
    gate = P[np.arange(P.shape[0]), P_index[np.arange(P.shape[0]), np.maximum(num-1, 0)]]
    # P[P <= [gate]]=0
    P = (P > [gate]).astype(np.float32)
    P=normalize(P, norm='l1', axis=1, copy=False)
    return sp.csr_matrix(P * features)


def Model26(W, t, alpha, y_train, train_mask, stored_A=None):
    A = absorption_probability(W, alpha, stored_A, train_mask)
    W = W.copy()
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    # if not isinstance(features, np.ndarray):
    #     features = features.toarray()
    print("Additional Label:")
    if not hasattr(t, '__getitem__'):
        t = [t for _ in range(y_train.shape[1])]
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]
        neighbors = np.argmax(A, 1)[index]

        # x1 = features[index, :].reshape((-1, 1, features.shape[1]))
        # x2 = features[y_train[:, i].astype(np.bool)].reshape((1, -1, features.shape[1]))
        # D = np.sum((x1 - x2) ** 2, axis=2) ** 0.5
        # D = np.mean(D, axis=1)
        # gate = 100000000 if t[i] >= D.shape[0] else np.sort(D, axis=0)[t[i]]
        # index = index[D<gate]
        # data = np.ones(len(np.where(y)[0])*len(index))
        # # rows = index.repeat(len(np.where(y)[0]))
        # # cols = np.hstack(np.where(y)[0] for i in range(len(index)))
        data = np.ones(len(index))
        rows = index
        cols = neighbors
        W += sp.coo_matrix((data, (rows, cols)), shape=W.shape)
        W += sp.coo_matrix((data, (cols, rows)), shape=W.shape)
        # train_index = np.hstack([train_index, index])
        correct_label_count(index, i)
    return sp.csr_matrix(W)

def Model28(adj, features, dataset, k):
    n = adj.shape[0]
    adj_I = adj + sp.diags(np.ones(n))
    adj_I_sym = normalize_adj(adj_I)

    try:
        # raise Exception('Debug')
        vals, vecs, vec_inv = np.load(dataset+'_I_vals.npy'), \
                              np.load(dataset+'_I_vecs.npy'), \
                              np.load(dataset + '_I_vec_inv.npy')
        print('load vals, vecs, vec_inv from files')
    except:
        vals, vecs = slinalg.eigsh(adj_I_sym, k=adj_I_sym.shape[0]-1)
        vecs = normalize(vecs, norm='l2', axis=0, copy=False)
        # vecs = d_I_inv_sqrt.dot(vecs)
        # vec_inv = np.linalg.inv(vecs)
        vec_inv = vecs.T

        # vals = vals.astype(np.float64)
        # vec_inv = vec_inv.astype(np.float64)
        np.save(dataset+'_I_vals.npy', vals)
        np.save(dataset+'_I_vecs.npy', vecs)
        np.save(dataset+'_I_vec_inv.npy', vec_inv)

    vals = 1-vals
    vecs = vecs[:, -k:]
    vals = vals[-k:]
    # vals_u = np.unique((1e8*vals).astype(np.int64))/1e8
    # sum_matrix = np.expand_dims(vals, axis=1) - np.expand_dims(vals_u, axis=0)
    # sum_matrix = (np.abs(sum_matrix) < 1e-7).astype(np.float32)
    # sum_matrix = np.zeros([300, 30])
    # sum_matrix[np.arange(300),np.arange(30).repeat(10)] = 1
    sum_matrix = np.ones([k, 1])
    return vecs
    features = features.T.dot(vecs)
    features = features.reshape([features.shape[0], 1, features.shape[1]])
    features = vecs.reshape([1]+list(vecs.shape))*features
    features = features.dot(sum_matrix)
    features = np.transpose(features, axes=(1,0,2))
    features = features.reshape([n, -1])
    return sp.csr_matrix(features, dtype=np.float32)

def taubin_smoothing(adj, lam, mu, repeat, features):
    n = adj.shape[0]
    adj = normalize(adj + sp.eye(adj.shape[0]), norm='l1', axis=1)
    smoothor = sp.eye(n) * (1 - lam) + lam * adj
    inflator = sp.eye(n) * (1 - mu) + mu * adj
    step_transformor = smoothor * inflator
    for i in range(repeat):
        features = step_transformor.dot(features)
    if sp.issparse(features):
        features = features.toarray()
    return features

def taubin_smoothor(adj, lam, mu, repeat):
    n = adj.shape[0]
    adj = normalize(adj + sp.eye(adj.shape[0]), norm='l1', axis=1)
    smoothor = sp.eye(n) * (1 - lam) + lam * adj
    inflator = sp.eye(n) * (1 - mu) + mu * adj
    step_transformor = smoothor * inflator
    transformor = sp.eye(n)
    base = step_transformor
    while repeat != 0:
        if repeat % 2:
            transformor *= base
        base *= base
        repeat //= 2
        # print(repeat)
    return transformor


all_labels = None


# dataset = None

def correct_label_count(indicator, i):
    count = np.sum(all_labels[:, i][indicator])
    if indicator.dtype == np.bool:
        total = np.where(indicator)[0].shape[0]
    elif indicator.dtype in [np.int, np.int8, np.int16, np.int32, np.int64]:
        total = indicator.shape[0]
    else:
        raise TypeError('indicator must be of data type np.bool or np.int')
    # print("     for class {}, {}/{} is correct".format(i, count, total))
    print(count, '/', total, sep='', end='\t')


def construct_feed_dict(features, support, labels, labels_mask, triplet, noise_sigma, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['noise_sigma']: noise_sigma})
    if len(triplet):
        feed_dict.update({placeholders['triplet']:triplet})
    return feed_dict


def preprocess_model_config(model_config):
    if model_config['Model'] not in [17, 23]:
        model_config['connection'] = list(model_config['connection'])
        # judge if parameters are legal
        for c in model_config['connection']:
            if c not in ['c', 'd', 'r', 'f', 'C']:
                raise ValueError(
                    'connection string specified by --connection can only contain "c", "d", "r", "f", "C" but "{}" found'.format(
                        c))
        for i in model_config['layer_size']:
            if not isinstance(i, int):
                raise ValueError('layer_size should be a list of int, but found {}'.format(model_config['layer_size']))
            if i <= 0:
                raise ValueError('layer_size must be greater than 0, but found {}' % i)
        if not len(model_config['connection']) == len(model_config['layer_size']) + 1:
            raise ValueError('length of connection string should be equal to length of layer_size list plus 1')

    # Generate name
    if not model_config['name']:
        if model_config['Model'] == 23:
            if model_config['classifier'] == 'svm':
                model_name = 'svm_' + model_config['svm_kernel']
                if model_config['svm_kernel'] == 'rbf':
                    model_name += '_' + str(model_config['gamma'])
                if model_config['svm_kernel'] == 'poly':
                    model_name += '_' + str(model_config['svm_degree'])
            elif model_config['classifier'] == 'tree':
                model_name = 'tree'
                if model_config['tree_depth']:
                    model_name += '_' + str(model_config['tree_depth'])
            elif model_config['classifier'] == 'cnn':
                model_name = 'cnn'
            else:
                raise ValueError('classifier:' + model_config['classifier'])
        else:
            model_name = model_config['connection'][0]
            for char, size in \
                    zip(model_config['connection'][1:], model_config['layer_size']):
                model_name += str(size) + char

            if model_config['conv'] == 'cheby':
                model_name += '_cheby' + str(model_config['max_degree'])
            elif model_config['conv'] == 'taubin':
                model_name += '_conv_taubin' + str(model_config['taubin_lambda']) \
                              + '_' + str(model_config['taubin_mu']) \
                              + '_' + str(model_config['taubin_repeat'])
            elif model_config['conv'] == 'test21':
                model_name += '_' + 'conv_test21' + '_' + str(model_config['alpha']) + '_' + str(model_config['beta'])
            elif model_config['conv'] == 'gcn_unnorm':
                model_name += '_' + 'gcn_unnorm'
            elif model_config['conv'] == 'gcn_noloop':
                model_name += '_' + 'gcn_noloop'
            if model_config['validate']:
                model_name += '_validate'

        if model_config['smoothing'] == 'ap':
            model_name += '_' + 'ap_smoothing' + '_' + str(model_config['smooth_alpha'])
        if model_config['smoothing'] == 'ap_appro':
            model_name += '_' + 'ap_appro' + '_' + str(model_config['smooth_alpha'])
        elif model_config['smoothing'] == 'test21':
            model_name += '_' + 'test21' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['beta'])
        elif model_config['smoothing'] == 'test21_norm':
            model_name += '_' + 'test21_norm' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['beta'])
        elif model_config['smoothing'] == 'test27':
            model_name += '_' + 'test27' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['beta'])
        elif model_config['smoothing'] == 'poly':
            model_name += '_' + 'poly_smoothing'
            for a in model_config['poly_parameters']:
                model_name += '_' + str(a)
        elif model_config['smoothing'] == 'taubin':
            model_name += '_taubin' + str(model_config['taubin_lambda']) \
                          + '_' + str(model_config['taubin_mu']) \
                          + '_' + str(model_config['taubin_repeat'])
        elif model_config['smoothing'] == 'sparse_encoding':
            model_name += '_sparse_encoding'
        elif model_config['smoothing'] is 'manifold_denoising':
            model_name += '_manifold_denoising' + '_' + str(model_config['smooth_alpha']) + '_' + str(model_config['md_repeat'])
        elif model_config['smoothing'] is None:
            pass
        else:
            raise ValueError('invalid smoothing')

        if model_config['smoothing'] is not None and model_config['Model'] == 17:
            model_name += '_' + str(model_config['k'])

        model_name += '_Model' + str(model_config['Model'])
        
        if model_config['Model'] in [1]:
            model_name += '_' + model_config['absorption_type'] + '_alpha_' + str(
                model_config['alpha'])
        if model_config['Model'] in [2, 3, 4, 7, 8]:
            model_name += '_s' + str(model_config['s']) + '_alpha_' + str(
                model_config['alpha'])
        if model_config['Model'] in [5]:
            model_name += '_mu' + str(model_config['mu'])
        if model_config['Model'] in [6]:
            pass
        if model_config['Model'] in [9, 10]:
            model_name += '_alpha_' + str(
                model_config['alpha']) + '_t' + str(model_config['t']).replace('[', '_').replace(']', '_').replace(', ',
                                                                                                                   '_')
        if model_config['Model'] in [11]:
            model_name += '_' + model_config["Model11"]
        if model_config['Model'] in [12]:
            pass
        if model_config['Model'] in [13]:
            model_name += '_' + model_config["Model11"]
            model_name += '_s' + str(model_config['s']) + '_alpha_' + str(
                model_config['alpha']) + '_t' + str(model_config['t']).replace('[', '_').replace(']', '_').replace(', ',
                                                                                                                   '_')
        if model_config['Model'] in [14]:
            pass
        if model_config['Model'] in [15]:
            model_name += '_s' + str(model_config['s']) + '_alpha_' + str(
                model_config['alpha']) + '_t' + str(model_config['t']).replace('[', '_').replace(']', '_').replace(', ',
                                                                                                                   '_')
        if model_config['Model'] in [16]:
            Model_to_add_label = copy.deepcopy(model_config)
            if 'Model_to_add_label' in Model_to_add_label:
                del Model_to_add_label['Model_to_add_label']
            if 'Model_to_predict' in Model_to_add_label:
                del Model_to_add_label['Model_to_predict']
            Model_to_add_label.update(model_config['Model_to_add_label'])
            model_config['Model_to_add_label'] = Model_to_add_label
            preprocess_model_config(model_config['Model_to_add_label'])

            Model_to_predict = copy.deepcopy(model_config)
            if 'Model_to_add_label' in Model_to_predict:
                del Model_to_predict['Model_to_add_label']
            if 'Model_to_predict' in Model_to_predict:
                del Model_to_predict['Model_to_predict']
            Model_to_predict.update(model_config['Model_to_predict'])
            model_config['Model_to_predict'] = Model_to_predict
            preprocess_model_config(model_config['Model_to_predict'])
            model_name = 'Model' + str(model_config['Model']) \
                         + '_{' + model_config['Model_to_add_label']['name'] + '}' \
                         + '_{' + model_config['Model_to_predict']['name'] + '}'
        if model_config['Model'] in [17]:
            model_name += '_alpha_' + str(model_config['alpha'])
        if model_config['Model'] in [18]:
            model_name += '_s' + str(model_config['s']) + '_alpha_' + str(
                model_config['alpha']) + '_t' + str(model_config['t']).replace('[', '_').replace(']', '_').replace(', ',
                                                                                                                   '_')
        if model_config['Model'] in [19]:
            Model_to_add_label = copy.deepcopy(model_config)
            if 'Model_to_add_label' in Model_to_add_label:
                del Model_to_add_label['Model_to_add_label']
            if 'Model_to_predict' in Model_to_add_label:
                del Model_to_add_label['Model_to_predict']
            Model_to_add_label.update(model_config['Model_to_add_label'])
            model_config['Model_to_add_label'] = Model_to_add_label
            preprocess_model_config(model_config['Model_to_add_label'])

            Model_to_predict = copy.deepcopy(model_config)
            if 'Model_to_add_label' in Model_to_predict:
                del Model_to_predict['Model_to_add_label']
            if 'Model_to_predict' in Model_to_predict:
                del Model_to_predict['Model_to_predict']
            Model_to_predict.update(model_config['Model_to_predict'])
            model_config['Model_to_predict'] = Model_to_predict
            preprocess_model_config(model_config['Model_to_predict'])
            model_name = 'Model' + str(model_config['Model']) + '_alpha' + str(model_config['alpha']) \
                         + '__' + model_config['Model_to_add_label']['name'] + '__' \
                         + '__' + model_config['Model_to_predict']['name'] + '__'
        if model_config['Model'] == 20:
            model_name += str(model_config['alpha']) + '_t' + str(model_config['t']).replace('[', '_').replace(']',
                                                                                                               '_').replace(
                ', ', '_')
            model_config['epochs'] *= 2
        if model_config['Model'] == 21:
            model_name += str(model_config['alpha']) + '_t' + str(model_config['t']).replace('[', '_').replace(']',
                                                                                                               '_').replace(
                ', ', '_') \
                          + '_t2' + str(model_config['t2']).replace('[', '_').replace(']', '_').replace(', ', '_')
            model_config['epochs'] *= 2
        if model_config['Model'] == 22:
            model_name += '_alpha_' + str(model_config['alpha'])
            raise ValueError('Please use model 0 with smoothing.')
        if model_config['Model'] == 28:
            model_name += '_k' + str(model_config['k'])
        
        if model_config['loss_func'] == 'imbalance':
            model_name+='_imbalance_beta'+str(model_config['ws_beta'])
        if model_config['loss_func'] == 'triplet':
            model_name+='_triplet_MARGIN'+str(model_config['MARGIN'])+'_lamda'+str(model_config['triplet_lamda'])+'_maxTrip'+ str(model_config['max_triplet'])
            
        model_config['name'] = model_name

    # Generate logdir
    if model_config['logging'] and not model_config['logdir']:
        train_size = '{}_train'.format(model_config['train_size'])
        i = 0
        while True:
            logdir = path.join('log', model_config['dataset'],
                               train_size, model_config['name'], 'run' + str(i))
            i += 1
            if not path.exists(logdir):
                break
        model_config['logdir'] = logdir
        #
        # # Checkpoint path
        # if not model_config.get('ckpt_path', None):
        #     model_config['ckpt_path'] = path.join(model_config['logdir'], 'checkpoint')


if __name__ == '__main__':
    A = sp.csr_matrix(np.array([
        [2., -1.],
        [1., 2.]
    ]))
    B = sp.csr_matrix(np.array([
        [2., 3],
        [6., 9],
    ]))
    X = np.array([
        [1.],
        [1.]
    ])
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora', 20, 20)
    alpha = 1e-5
    adj = normalize_adj(adj)
    adj = adj.copy().astype(np.float32)
    D = adj.sum(1).flat
    L = sp.diags(D) - adj
    L += alpha * sp.eye(adj.shape[0], dtype=L.dtype)

    inv = np.linalg.inv(L.toarray())
    X = gaussian_seidel(L, sp.eye(L.shape[0], dtype=L.dtype).tocsr()[:, train_mask])
    pass


def pow3(a, b):
    ans = 1
    base = a
    while (b != 0):
        if (b % 2):
            ans *= base
        base *= base
        b //= 2
    return ans
