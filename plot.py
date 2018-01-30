import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from sklearn.preprocessing import normalize
from gcn.utils import Test21, absorption_probability

def load_data(dataset_str):
    """Load data."""
    # if dataset_str in ['USPS-Fea', 'CIFAR-Fea', 'Cifar_10000_fea', 'Cifar_R10000_fea']:
    #     data = sio.loadmat('data/{}.mat'.format(dataset_str))
    #     labels = data['labels']
    #     labels = np.zeros([data['labels'].shape[0],np.max(data['labels'])+1])
    #     labels[np.arange(data['labels'].shape[0]),data['labels'].astype(np.int16).flatten()] = 1
    #     features = sp.lil_matrix(data['X']+1)
    #     adj = data['G']

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    g = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    # features = sp.lil_matrix(allx)

    labels = np.vstack((ally, ty))
    return features, labels, adj

def preprocess_features(features, feature_type):
    # """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5)
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # return adj*d_inv_sqrt*d_inv_sqrt.flatten()
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def laplacian_smooth(features, adj, lam, mu, repeat=1):
    n = adj.shape[0]
    smoothor = sp.diags(np.ones(n)) * (1-lam) + lam * adj
    inflator = sp.diags(np.ones(n)) * (1-mu) + mu * adj
    transformor = smoothor*inflator
    for i in range(repeat):
        features = transformor * features
    return features

if __name__ == '__main__':
    dataset = 'cora'
    features, labels, adj = load_data(dataset)
    y=np.argmax(labels, axis=1)
    n = adj.shape[0]
    adj_I = adj + sp.diags(np.ones(n))
    # d_I = np.diag(np.array(adj.sum(1)))

    adj_rw = normalize(adj, norm='l1', axis=1)
    adj_sym = normalize_adj(adj)
    adj_I_rw = normalize(adj, norm='l1', axis=1)
    adj_I_sym = normalize_adj(adj_I) #
    d_I = np.array(adj_I.sum(1)).reshape(-1)
    d_I_inv_sqrt = np.power(d_I, -0.5)
    d_I = np.diag(d_I)
    d_I_inv_sqrt = np.diag(d_I_inv_sqrt)

    try:
        vals, vecs, vec_inv = np.load(dataset+'_I_vals.npy'), \
                              np.load(dataset+'_I_vecs.npy'), \
                              np.load(dataset + '_I_vec_inv.npy')
        print('load vals, vecs, vec_inv from files')
    except:
        vals, vecs = slinalg.eigsh(adj_I_sym, k=adj_I_sym.shape[0]-1)
        vecs = d_I_inv_sqrt.dot(vecs)
        # vec_inv = np.linalg.inv(vecs)
        vec_inv = vecs.T

        # vals = vals.astype(np.float64)
        # vec_inv = vec_inv.astype(np.float64)
        np.save(dataset+'_I_vals.npy', vals)
        np.save(dataset+'_I_vecs.npy', vecs)
        np.save(dataset+'_I_vec_inv.npy', vec_inv)
    x_I = 1-vals

    try:
        vals, vecs, vec_inv = np.load(dataset+'_vals.npy'), \
                              np.load(dataset+'_vecs.npy'), \
                              np.load(dataset + '_vec_inv.npy')
        print('load vals, vecs, vec_inv from files')
    except:
        vals, vecs = slinalg.eigsh(adj_sym, k=adj_I_sym.shape[0]-1)
        vecs = d_I_inv_sqrt.dot(vecs)
        # vec_inv = np.linalg.inv(vecs)
        vec_inv = vecs.T

        # vals = vals.astype(np.float64)
        # vec_inv = vec_inv.astype(np.float64)
        np.save(dataset+'_vals.npy', vals)
        np.save(dataset+'_vecs.npy', vecs)
        np.save(dataset+'_vec_inv.npy', vec_inv)
    x = 1-vals

    import matplotlib.pyplot as plt
    import numpy as np
    dpi = 300

    plt.figure()
    plt.grid()
    axes = plt.gca()
    alpha = 0.1
    A = absorption_probability(adj_sym, alpha, stored_A=dataset+'_A_sym')
    A_acc = np.add.accumulate(-np.sort(-A, axis=None))
    A_acc /= A_acc[-1]
    l = A.shape[0]*A.shape[1]
    plt.plot(np.arange(l)/n, A_acc)
    beta = 20
    y = A_acc[beta*n]
    plt.scatter(beta, y, c='C1')
    axes.text(beta+50, y, "({:.0f}, {:.2f})".format(beta, y))
    beta = 200
    y = A_acc[beta*n]
    plt.scatter(beta, y, c='C2')
    axes.text(beta+50, y, "({:.0f}, {:.2f})".format(beta, y))
    beta = 530
    y = A_acc[beta*n]
    plt.scatter(beta, y, c='C3')
    axes.text(beta+50, y+0.01, "({:.0f}, {:.2f})".format(beta, y))
    plt.xlabel(r'$\beta$')
    plt.ylabel('percentage of sum')
    plt.savefig('image/energy_entries.jpg', dpi=dpi)

    area = 2
    alpha = 1

    plt.figure()
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 / (1 + 5 * x), label=r'$(1+\alpha\lambda)^{-1}, \alpha=5$', alpha=alpha, linewidth=area)
    plt.plot(x, 1 - x, label=r'$1-\lambda$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - x) ** 2, label=r'$(1-\lambda)^2$', alpha=alpha, linewidth=area)
    plt.xlabel(r'$\lambda$')
    legend = plt.legend(loc='lower left', shadow=True)
    plt.savefig('image/response_function.jpg', dpi=dpi)
    # plt.show()


    plt.figure(figsize=(3.5, 3))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1.1, 1.1])
    plt.grid()
    for i in [0.9, 0.7, 0.5, 0.3]:
        plt.plot([0, i, i, 2], [1, 1, 0, 0], label=r'$\lambda_k={}$'.format(i), alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True)
    plt.savefig('image/filters/ideal_filters.jpg', dpi=dpi)


    plt.figure(figsize=(3.5, 3))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 / (1 + 3 * x), label=r'$(1+3\lambda)^{-1}$', alpha=alpha, linewidth=area)
    plt.plot(x, 1 / (1 + 5 * x), label=r'$(1+5\lambda)^{-1}$', alpha=alpha, linewidth=area)
    plt.plot(x, 1 / (1 + 10 * x), label=r'$(1+10\lambda)^{-1}$', alpha=alpha, linewidth=area)
    plt.plot(x, 1 / (1 + 20 * x), label=r'$(1+20\lambda)^{-1}$', alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True)
    plt.savefig('image/filters/lp-like_filters.jpg', dpi=dpi)


    plt.figure(figsize=(3.5, 3))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 - x, label=r'$1-\lambda$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - x) ** 2, label=r'$(1-\lambda)^2$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - x) ** 3, label=r'$(1-\lambda)^3$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - x) ** 4, label=r'$(1-\lambda)^4$', alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True)
    plt.savefig('image/filters/gcn-like_filters_1.jpg', dpi=dpi)


    plt.figure(figsize=(3.5, 3))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 - 0.5*x, label=r'$1-\frac{1}{2}\lambda$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - 0.5*x) ** 2, label=r'$(1-0.5\lambda)^2$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - 0.5*x) ** 3, label=r'$(1-0.5\lambda)^3$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - 0.5*x) ** 4, label=r'$(1-0.5\lambda)^4$', alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True)
    plt.savefig('image/filters/gcn-like_filters_2.jpg',dpi=dpi)


    plt.figure(figsize=(3.5, 3))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, (1 - 0.5*x) ** 2, label=r'$(1-0.5\lambda)^2$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - x) ** 2, label=r'$(1-\lambda)^2$', alpha=alpha, linewidth=area)
    plt.plot(x, (1 - 0.5*x) ** 4, label=r'$(1-0.5\lambda)^4$', alpha=alpha, linewidth=area)
    # plt.plot(x, (1 - x) ** 3, label=r'$(1-\lambda)^3$', alpha=alpha, linewidth=area)
    # plt.plot(x, (1 - 0.5*x) ** 6, label=r'$(1-0.5\lambda)^6$', alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True)
    plt.savefig('image/filters/gcn-like_filters_3.jpg',dpi=dpi)


    sample = np.random.random_sample(x.shape)
    sample_rate=0.2
    x = x[sample<sample_rate]
    x_I = x_I[sample<sample_rate]
    area = 4


    plt.figure(figsize=(4, 3.5))
    plt.scatter(x, 1 - x, alpha=alpha, s=area)
    # plt.title(r'$(\lambda, 1-\lambda)$')
    # plt.ylabel(r'$1-\lambda$', rotation='horizontal')
    # plt.xlabel(r'$\lambda$')
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.yaxis.set_label_coords(-0.1, 1.05)
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig('image/compressing_effect/1_lambda.jpg', dpi=dpi)

    plt.figure(figsize=(4, 3.5))
    plt.scatter(x, (1 - x) ** 2, alpha=alpha, s=area)
    # plt.title(r'$\left(\lambda, (1-\lambda)^2\right)$')
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel(r'$(1-\lambda)^2$', rotation='horizontal')
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.yaxis.set_label_coords(-0.1, 1.05)
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig('image/compressing_effect/1_lambda_2.jpg', dpi=dpi)

    plt.figure(figsize=(4, 3.5))
    plt.scatter(x_I, 1 - x_I, s=area, alpha=alpha)
    # plt.title(r'$(\tilde{\;\lambda}, 1-\tilde{\;\lambda})$')
    # plt.xlabel(r'$\tilde{\;\lambda}$')
    # plt.ylabel(r'$1-\tilde{\;\lambda}$', rotation='horizontal')
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.yaxis.set_label_coords(-0.1, 1.05)
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig('image/compressing_effect/1_tilde_lambda.jpg', dpi=dpi)

    plt.figure(figsize=(4, 3.5))
    plt.scatter(x_I, (1 - x_I) ** 2, s=area, alpha=alpha)
    # plt.title(r'$(\tilde{\;\lambda},(1-\tilde{\;\lambda})^2)$')
    # plt.xlabel(r'$\tilde{\;\lambda}$')
    # plt.ylabel(r'$(1-\tilde{\;\lambda})^2$', rotation='horizontal')
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.yaxis.set_label_coords(-0.1, 1.05)
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig('image/compressing_effect/1_tilde_lambda_2.jpg', dpi=dpi)



    colors = np.array(['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6']).repeat(len(x))
    y = np.hstack([1 - x,
                  1 - x_I,
                  (1 - x) ** 2,
                  (1 - x_I) ** 2,
                  1 / (1 + 5 * x),
                  (2 - x) / 2,
                  ((2 - x) ** 2) / 4
                  ])
    labels = [r'$1-\lambda$',
              r'$1-\lambda$ with self-loop',
              r'$(1-\lambda)^2$',
              r'$(1-\lambda)^2$ with self-loop',
              r'$(1+\alpha\lambda)^{-1}, \alpha=5$',
              r'$\frac{(2-\lambda)}{2}$',
              r'$\frac{(2-\lambda)^2}{4}$']
    x = np.hstack([x, x_I, x, x_I, x, x, x])
    points = np.array([x,y,colors]).T
    np.random.shuffle(points)
    # plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=area, alpha=alpha)
