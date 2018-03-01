import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from numpy import linalg
import scipy.misc
from sklearn.preprocessing import normalize
from gcn.utils import Test21, absorption_probability, smooth, load_data, taubin_smoothing
import gcn.utils
from config import configuration
import matplotlib.pyplot as plt
import numpy as np
import json

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
    # model_config = configuration['model_list'][0]
    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, size_of_each_class, triplet = \
    #     load_data(model_config['dataset'], train_size=model_config['train_size'],
    #               validation_size=model_config['validation_size'],
    #               model_config=model_config, shuffle=model_config['shuffle'])
    #
    # stored = model_config['dataset']
    # n = adj.shape[0]
    # adj_I = adj + sp.diags(np.ones(n))
    # adj_I_sym = normalize_adj(adj_I)
    #
    # try:
    #     # raise Exception('DEBUG')
    #     vals, vecs, vec_inv = np.load(stored+'_I_vals.npy'), \
    #                           np.load(stored+'_I_vecs.npy'), \
    #                           np.load(stored + '_I_vec_inv.npy')
    #     print('load vals, vecs, vec_inv from files')
    # except:
    #     # vals, vecs = slinalg.eigsh(adj_I_sym.toarray(), k=adj_I_sym.shape[0]-1)
    #     vals, vecs = linalg.eigh(adj_I_sym.toarray())
    #     vec_inv = vecs.T
    #
    #     np.save(stored+'_I_vals.npy', vals)
    #     np.save(stored+'_I_vecs.npy', vecs)
    #     np.save(stored+'_I_vec_inv.npy', vec_inv)
    # c = vec_inv.dot(gcn.utils.all_labels)
    # c_abs = np.abs(c)
    # sorted = np.sort(c_abs.flatten())
    # # acc = np.add.accumulate(sorted)
    # import matplotlib.pyplot as plt
    # # plt.plot(acc)
    # for i in range(c.shape[1]):
    #     plt.plot(vals, c_abs[:,i], 'o')
    #     plt.show()



    dpi = 300

    plt.figure(figsize=(3, 4))
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.yticks(np.arange(0.,1.1,0.1))

    plt.grid()
    labels = [r'AP, $\alpha=10$',
              r'RNM, $k=6$',
              r'RW, $k=12$',
              r'Raw features']
    files = ['run-cnn_ap_appro_0.1_Model23_eval-tag-{}.json',
             'run-cnn_taubin1_0_6_Model23_eval-tag-{}.json',
             'run-cnn_taubin0.5_0_12_Model23_eval-tag-{}.json',
             'run-cnn_Model23_eval-tag-{}.json']
    for i in range(4):
        xy = np.array(json.load(open(files[i].format('accuracy'))))[0:100]
        plt.plot(xy[:,1], xy[:,2], label=labels[i])
    legend = plt.legend(loc='lower right', shadow=True, fontsize=10)
    plt.xlabel(r'Train steps', fontsize=14)
    plt.ylabel(r'Test accuracy', rotation='horizontal', fontsize=14)
    axes.yaxis.set_label_coords(0.05,1.05)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/train/test_acc.pdf', dpi=dpi, bbox_inches="tight")


    plt.figure(figsize=(3, 4))
    axes = plt.gca()
    plt.grid()
    for i in range(4):
        xy = np.array(json.load(open(files[i].format('loss'))))[0:100]
        plt.plot(xy[:,1], xy[:,2], label=labels[i])
    legend = plt.legend(loc='upper right', shadow=True, fontsize=10)
    plt.xlabel(r'Train steps', fontsize=14)
    plt.ylabel(r'Test loss', rotation='horizontal', fontsize=14)
    axes.yaxis.set_label_coords(0,1.05)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/train/test_loss.pdf', dpi=dpi, bbox_inches="tight")



    plt.figure(figsize=(2.5, 2.5))
    axes = plt.gca()
    # axes.set_ylim(0.90,0.955)
    axes.set_xlim(0,21)
    plt.grid()
    alpha = [20, 12.5, 10, 5, 3.3, ]
    acc1 = [94.47, 94.7, 94.72, 94.46, 94.44, ]
    plt.plot(alpha, acc1, '-o', label='AP')
    plt.plot([3,20], [92.05, 92.05], '--',label='LP')
    plt.plot([3,20], [91.33, 91.33], '--',label='GCN')
    plt.plot([3,20], [91.60, 91.60], '--',label='CNN')
    legend = plt.legend(loc='lower left', shadow=True, fontsize=12)
    plt.xlabel(r'$\alpha$', fontsize=14)
    plt.ylabel(r'Test accuracy', rotation='horizontal', fontsize=14)
    axes.yaxis.set_label_coords(0.1,1.05)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/parameters/parameters_AP.pdf', dpi=dpi, bbox_inches="tight")

    plt.figure(figsize=(2.5, 2.5))
    axes = plt.gca()
    # axes.set_ylim(0.90,0.955)
    axes.set_xlim(0,21)
    plt.grid()
    k1 = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    acc2 = [94.64, 94.95, 94.95, 94.97, 95.09, 94.65, 94.82, 94.94, 94.96]
    plt.plot(k1, acc2, '-o', label='RW')
    plt.plot([3,20], [92.05, 92.05], '--',label='LP')
    plt.plot([3,20], [91.33, 91.33], '--',label='GCN')
    plt.plot([3,20], [91.60, 91.60], '--',label='CNN')
    legend = plt.legend(loc='lower left', shadow=True, fontsize=12)
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'Test accuracy', rotation='horizontal', fontsize=14)
    axes.yaxis.set_label_coords(0.1,1.05)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/parameters/parameters_RW.pdf', dpi=dpi, bbox_inches="tight")

    plt.figure(figsize=(2.5, 2.5))
    axes = plt.gca()
    # axes.set_ylim(0.90,0.955)
    axes.set_xlim(0,11)
    plt.grid()
    k2 = [3, 4, 5, 6, 7, 8, 9, 10, ]
    acc3 = [94.48, 94.53, 94.94, 94.76, 94.85, 94.95, 95.17, 94.81, ]
    plt.plot(k2, acc3, '-o', label='RNM')
    plt.plot([3,10], [92.05, 92.05], '--',label='LP')
    plt.plot([3,10], [91.33, 91.33], '--',label='GCN')
    plt.plot([3,10], [91.60, 91.60], '--',label='CNN')
    legend = plt.legend(loc='lower left', shadow=True, fontsize=12)
    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel(r'Test accuracy', rotation='horizontal', fontsize=14)
    axes.yaxis.set_label_coords(0.1,1.05)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/parameters/parameters_RNM.pdf', dpi=dpi, bbox_inches="tight")

    model_config = {
        'dataset'           : 'cora',     # 'Dataset string. (cora | citeseer | pubmed | CIFAR-Fea | Cifar_10000_fea | Cifar_R10000_fea | USPS-Fea | MNIST-Fea | MNIST-10000)'
        'shuffle'           : False,
        'train_size'        : 10,         # if train_size is a number, then use TRAIN_SIZE labels per class.
        # 'train_size'        : [20 for i in range(10)], # if train_size is a list of numbers, then it specifies training labels for each class.
        'validation_size'   : 500,           # 'Use VALIDATION_SIZE data to train model'
        'validate'          : False,        # Whether use validation set
        'loss_func'         :'default',     #'imbalance', 'triplet'
        'ws_beta'           : 20,
        'max_triplet':1000,  #for triplet, 1000 for cora to get all tripets
        'feature'           : 'bow',
        'test_size'         : None,

        'smoothing': 'taubin',
        'taubin_lambda': 0.5,
        'taubin_mu': 0,
        'taubin_repeat': 10,
    }
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, size_of_each_class, triplet = \
        load_data(model_config['dataset'], train_size=model_config['train_size'],
                  validation_size=model_config['validation_size'],
                  model_config=model_config, shuffle=model_config['shuffle'])

    # train_size = model_config['train_size']
    # order = np.argsort(labels[train_mask])
    # digits = features[train_mask][order].reshape(10,train_size,28,28)
    # digits = np.vstack([np.hstack([digits[i,j,:,:] for j in range(digits.shape[1])]) for i in range(digits.shape[0])])
    # scipy.misc.imsave('image/digits/digits.pdf', digits)
    #
    # for repeat in [20]:
    #     smoothed_digits = taubin_smoothing(adj, model_config['taubin_lambda'], model_config['taubin_mu'], repeat, features)
    #     smoothed_digits = smoothed_digits[train_mask][order].reshape(10,train_size,28,28)
    #     smoothed_digits = np.vstack([np.hstack([smoothed_digits[i,j,:,:] for j in range(smoothed_digits.shape[1])]) for i in range(smoothed_digits.shape[0])])
    #     scipy.misc.imsave('image/digits/smoothed_digits_{}.pdf'.format(repeat), smoothed_digits)


    dataset = model_config['dataset']

    y=np.argmax(gcn.utils.all_labels, axis=1)
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


    # plt.figure()
    # plt.grid()
    # axes = plt.gca()
    # alpha = 0.1
    # A = absorption_probability(adj_sym, alpha, stored_A=dataset+'_A_sym')
    # A_acc = np.add.accumulate(-np.sort(-A, axis=None))
    # A_acc /= A_acc[-1]
    # l = A.shape[0]*A.shape[1]
    # plt.plot(np.arange(l)/n, A_acc)
    # beta = 20
    # y = A_acc[beta*n]
    # plt.scatter(beta, y, c='C1')
    # axes.text(beta+50, y, "({:.0f}, {:.2f})".format(beta, y))
    # beta = 200
    # y = A_acc[beta*n]
    # plt.scatter(beta, y, c='C2')
    # axes.text(beta+50, y, "({:.0f}, {:.2f})".format(beta, y))
    # beta = 530
    # y = A_acc[beta*n]
    # plt.scatter(beta, y, c='C3')
    # axes.text(beta+50, y+0.01, "({:.0f}, {:.2f})".format(beta, y))
    # plt.xlabel(r'$\beta$')
    # plt.ylabel('percentage of sum')
    # plt.savefig('image/energy_entries.pdf', dpi=dpi)

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
    plt.savefig('image/response_function.pdf', dpi=dpi)
    # plt.show()


    color = ['C0','C4','C2','C3']
    plt.figure(figsize=(3.2, 2.5))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1.1, 1.1])
    plt.grid()
    for i in [0.9, 0.7, 0.5, 0.3]:
        plt.plot([0, i, i, 2], [1, 1, 0, 0], label=r'$\lambda_k={}$'.format(i), alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True, fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/filters/ideal_filters.pdf', dpi=dpi, bbox_inches="tight")

    plt.figure(figsize=(3.2, 2.5))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 / (1 + 3 * x), label=r'$(1+3\lambda)^{-1}$', alpha=alpha, linewidth=area, c=color[0])
    plt.plot(x, 1 / (1 + 5 * x), label=r'$(1+5\lambda)^{-1}$', alpha=alpha, linewidth=area, c=color[1])
    plt.plot(x, 1 / (1 + 10 * x), label=r'$(1+10\lambda)^{-1}$', alpha=alpha, linewidth=area, c=color[2])
    plt.plot(x, 1 / (1 + 20 * x), label=r'$(1+20\lambda)^{-1}$', alpha=alpha, linewidth=area, c=color[3])
    legend = plt.legend(loc='lower left', shadow=True, fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/filters/lp-like_filters.pdf', dpi=dpi, bbox_inches="tight")


    plt.figure(figsize=(3.2, 2.5))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 - x, label=r'$1-\lambda$', alpha=alpha, linewidth=area, c=color[0])
    plt.plot(x, (1 - x) ** 2, label=r'$(1-\lambda)^2$', alpha=alpha, linewidth=area, c=color[1])
    plt.plot(x, (1 - x) ** 3, label=r'$(1-\lambda)^3$', alpha=alpha, linewidth=area, c=color[2])
    plt.plot(x, (1 - x) ** 4, label=r'$(1-\lambda)^4$', alpha=alpha, linewidth=area, c=color[3])
    legend = plt.legend(loc='lower left', shadow=True, fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/filters/gcn-like_filters_1.pdf', dpi=dpi, bbox_inches="tight")


    plt.figure(figsize=(3.2, 2.5))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, 1 - 0.5*x, label=r'$1-\frac{1}{2}\lambda$', alpha=alpha, linewidth=area, c=color[0])
    plt.plot(x, (1 - 0.5*x) ** 2, label=r'$(1-0.5\lambda)^2$', alpha=alpha, linewidth=area, c=color[1])
    plt.plot(x, (1 - 0.5*x) ** 3, label=r'$(1-0.5\lambda)^3$', alpha=alpha, linewidth=area, c=color[2])
    plt.plot(x, (1 - 0.5*x) ** 4, label=r'$(1-0.5\lambda)^4$', alpha=alpha, linewidth=area, c=color[3])
    legend = plt.legend(loc='lower left', shadow=True, fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/filters/gcn-like_filters_2.pdf',dpi=dpi, bbox_inches="tight")


    plt.figure(figsize=(3.2, 2.5))
    plt.plot([-1, 3], [0, 0], c='black', linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0, 2])
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.plot(x, (1 - 0.5*x) ** 2, label=r'$(1-0.5\lambda)^2$', alpha=alpha, linewidth=area, c=color[0])
    plt.plot(x, (1 - x) ** 2, label=r'$(1-\lambda)^2$', alpha=alpha, linewidth=area, c=color[2])
    plt.plot(x, (1 - 0.5*x) ** 4, label=r'$(1-0.5\lambda)^4$', alpha=alpha, linewidth=area, c=color[3])
    # plt.plot(x, 1 / (1 + 6 * x), label=r'$(1+6\lambda)^{-1}$', alpha=alpha, linewidth=area)
    # plt.plot(x, (1 - x) ** 3, label=r'$(1-\lambda)^3$', alpha=alpha, linewidth=area)
    # plt.plot(x, (1 - 0.5*x) ** 6, label=r'$(1-0.5\lambda)^6$', alpha=alpha, linewidth=area)
    legend = plt.legend(loc='lower left', shadow=True, fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/filters/filter-comparison.pdf',dpi=dpi, bbox_inches="tight")


    sample = np.random.random_sample(x.shape)
    sample_rate=0.2
    x = x[sample<sample_rate]
    x_I = x_I[sample<sample_rate]
    area = 4


    plt.figure(figsize=(3.2, 2.5))
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
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/compressing_effect/1_lambda.pdf', dpi=dpi, bbox_inches="tight")

    plt.figure(figsize=(3.2, 2.5))
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
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/compressing_effect/1_lambda_2.pdf', dpi=dpi, bbox_inches="tight")

    plt.figure(figsize=(3.2, 2.5))
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
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/compressing_effect/1_tilde_lambda.pdf', dpi=dpi, bbox_inches="tight")

    plt.figure(figsize=(3.2, 2.5))
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
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig('image/compressing_effect/1_tilde_lambda_2.pdf', dpi=dpi, bbox_inches="tight")



    # colors = np.array(['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6']).repeat(len(x))
    # y = np.hstack([1 - x,
    #               1 - x_I,
    #               (1 - x) ** 2,
    #               (1 - x_I) ** 2,
    #               1 / (1 + 5 * x),
    #               (2 - x) / 2,
    #               ((2 - x) ** 2) / 4
    #               ])
    # labels = [r'$1-\lambda$',
    #           r'$1-\lambda$ with self-loop',
    #           r'$(1-\lambda)^2$',
    #           r'$(1-\lambda)^2$ with self-loop',
    #           r'$(1+\alpha\lambda)^{-1}, \alpha=5$',
    #           r'$\frac{(2-\lambda)}{2}$',
    #           r'$\frac{(2-\lambda)^2}{4}$']
    # x = np.hstack([x, x_I, x, x_I, x, x, x])
    # points = np.array([x,y,colors]).T
    # np.random.shuffle(points)
    # plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=area, alpha=alpha)
