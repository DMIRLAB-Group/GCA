# Some basic helper functions
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score
import pandas as pd
from scipy.stats import entropy


def load_air_dataset(path, domain, train_mode="train"):
    data = pd.read_csv(os.path.join(path, domain + "_%s.csv" % train_mode)).values

    data = data[:, 1:]
    return data


def construct_training_dataset(data, order):
    # Pack the dataset, if it is not in alist already
    if not isinstance(data, list):
        data = [data]

    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0
    for r in range(len(data)):
        data_r = data[r]
        # dataset: T x p
        T_r = data_r.shape[0]
        p_r = data_r.shape[1]
        inds_r = np.arange(order, T_r)
        data_out_r = np.zeros((T_r - order, order, p_r))
        response_r = np.zeros((T_r - order, p_r))
        time_idx_r = np.zeros((T_r - order, ))
        for i in range(T_r - order):
            j = inds_r[i]
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, :]
            time_idx_r[i] = j
        # TODO: just a hack, need a better solution...
        # time_idx_r = time_idx_r + offset + 200 * (66 >= 1)
        time_idx_r = time_idx_r.astype(int)
        if data_out is None:
            data_out = data_out_r
            response = response_r
            time_idx = time_idx_r
        else:
            data_out = np.concatenate((data_out, data_out_r), axis=0)
            response = np.concatenate((response, response_r), axis=0)
            time_idx = np.concatenate((time_idx, time_idx_r))
    return data_out, response, time_idx


def eval_causal_structure(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))]
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))]
        if np.max(a_true_offdiag) == np.min(a_true_offdiag):
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
            auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
    return auroc, auprc


def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
    else:
        precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
    return accuracy, bal_accuracy, precision, recall


def eval_concordance(a_1: np.ndarray, a_2: np.ndarray):
    a_1 = a_1.flatten()
    a_2 = a_2.flatten()
    n_pairs = len(a_1) * (len(a_1) - 1)
    n_concordant_pairs = 0
    n_discordant_pairs = 0

    for i in range(len(a_1)):
        for j in range(len(a_1)):
            if i != j:
                if a_1[i] < a_1[j] and a_2[i] < a_2[j]:
                    n_concordant_pairs += 1
                elif a_1[i] > a_1[j] and a_2[i] > a_2[j]:
                    n_concordant_pairs += 1
                elif a_1[i] < a_1[j] and a_2[i] > a_2[j]:
                    n_discordant_pairs += 1
                elif a_1[i] > a_1[j] and a_2[i] < a_2[j]:
                    n_discordant_pairs += 1
    cindex = (n_concordant_pairs - n_discordant_pairs) / n_pairs

    return cindex


def kl_div_disc(x: np.ndarray, y: np.ndarray, n_bins=16):
    # NOTE: KL divergences are symmetrised!
    # Discretise and approximate using histograms
    h_y, bin_edges = np.histogram(a=y, bins=n_bins, density=False)
    # NOTE: Adding a small constant to avoid division by 0
    h_y = h_y + 1e-6
    h_y = h_y / np.sum(h_y)
    h_x, _ = np.histogram(a=x, bins=bin_edges, density=False)
    h_x = h_x + 1e-6
    h_x = h_x / np.sum(h_x)

    # Compute [D_{KL}(P || Q) + D_{KL}(Q || P)] / 2 for discrete distributions given by histograms
    return (entropy(pk=h_x, qk=h_y, axis=0) + entropy(pk=h_y, qk=h_x, axis=0)) / 2


def kl_div_normal(x: np.ndarray, y: np.ndarray):
    # NOTE: KL divergences are symmetrised!
    # Assumes normality
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    # Add a small positive constants to avoid division by 0
    sigma_x = np.std(x) + 1e-6
    sigma_y = np.std(x) + 1e-6

    kl_xy = np.log(sigma_y / sigma_x) + (sigma_x ** 2 + (mu_x - mu_y) ** 2) / (2 * sigma_y ** 2) - 0.5
    kl_yx = np.log(sigma_x / sigma_y) + (sigma_y ** 2 + (mu_y - mu_x) ** 2) / (2 * sigma_x ** 2) - 0.5

    return (kl_xy + kl_yx) / 2


def absolute_mean_deviation(x: np.ndarray, y: np.ndarray):
    return np.abs(np.mean(x) - np.mean(y))


def absolute_mean_relative_deviation(x: np.ndarray, y: np.ndarray):
    return np.abs(np.mean(x) - np.mean(y)) / (np.abs(np.mean(y)) + 1e-6)


def shuffle_aligned_list(x, y):
    num = x.shape[0]
    p = np.random.permutation(num)
    new_x = x[p]
    new_y = y[p]

    return new_x, new_y


def data_iterator(x, y, batch_size, shuffle=True):

    if shuffle:
        x, y = shuffle_aligned_list(x, y)
    batch_count = 0

    while True:
        if batch_count * batch_size + batch_size > x.shape[0]:
            # print("end")
            batch_count = 0
            if shuffle:
                x, y = shuffle_aligned_list(x, y)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1

        yield x[start: end], y[start: end]


def data_iterator_2(x, y, time_idx, batch_size):
    inds = np.arange(0, x.shape[0])
    np.random.shuffle(inds)
    batch_split = list(np.arange(0, len(inds), batch_size))
    if batch_split[-1] < len(inds):
        batch_split.append(len(inds))

    batch_index_count = 0
    while True:
        if batch_index_count >= len(batch_split)-1:
            # print("end")
            batch_index_count = 0
            inds = np.arange(0, x.shape[0])
            np.random.shuffle(inds)
            batch_split = list(np.arange(0, len(inds), batch_size))
            if batch_split[-1] < len(inds) - 1:
                batch_split.append(len(inds) - 1)

        start = batch_split[batch_index_count]
        end = batch_split[batch_index_count + 1]
        batch_inds = inds[start: end]

        batch_x = x[batch_inds, :, :]
        batch_y = y[batch_inds, :]
        batch_time_idx = time_idx[batch_inds]
        batch_next_time_idx = batch_time_idx + 1

        input_next = x[np.where(np.isin(time_idx, batch_next_time_idx))[0], :, :]

        batch_index_count += 1

        yield batch_x, batch_y, batch_time_idx, batch_next_time_idx, input_next, time_idx, batch_inds


def construct_non_cuasal_train_data(predictor, responses, feature_id):
    new_label_list = list()

    for each_y in responses:

        new_each_y = np.expand_dims(each_y[feature_id], axis=0)
        new_label_list.append(new_each_y)

    return predictor, np.concatenate(new_label_list, axis=0)