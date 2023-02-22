import numpy as np
from scipy.integrate import odeint
from scipy.stats import bernoulli
from copy import deepcopy
import random
import math


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR models to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def random_generate_link_list_with_lag_2(alist, lag):
    # lag_list = [alist]
    # for i in range(lag - 1):
    #     sample_num_upper = int(math.ceil(len(lag_list[-1]) * 0.7))
    #     sample_num_lower = int(math.ceil(len(lag_list[-1]) * 0.5))
    #     sample_num = random.randint(sample_num_lower, sample_num_upper)
    #     next_lag_link_list = random.sample(lag_list[-1], sample_num)
    #     lag_list.append(next_lag_link_list)

    lag_list = [alist]
    for i in range(lag - 1):
        next_lag_link_list = list()
        for j, _ in enumerate(lag_list[-1]):
            sample_num_upper = int(math.ceil(len(lag_list[-1][j][1]) * 0.8))
            sample_num_lower = int(math.ceil(len(lag_list[-1][j][1]) * 0.5))
            sample_num = random.randint(sample_num_lower, sample_num_upper)
            row_lag_link_list = random.sample(list(lag_list[-1][j][1]), sample_num)
            next_lag_link_list.append((j, row_lag_link_list))

        lag_list.append(next_lag_link_list)

    return lag_list


def simulate_three_domain_var_2(p, T, lag, sd_list, sparsity=0.2, beta_value=1.0, scale=0.3, swag_num=5,
                                data_type="nonlinear", sample_type="random", seed=0):

    np.random.seed(seed=seed)

    base_GC = np.eye(p, dtype=int)
    base_beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    base_record_list = []

    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        base_beta[i, choice] = random.uniform(-1, 1)
        base_GC[i, choice] = 1
        base_record_list.append((i, choice))

    domain1_lag_list = random_generate_link_list_with_lag_2(alist=base_record_list, lag=lag)
    domain2_lag_list = random_generate_link_list_with_lag_2(alist=base_record_list, lag=lag)
    domain3_lag_list = random_generate_link_list_with_lag_2(alist=base_record_list, lag=lag)

    domain1_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    domain2_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    domain3_beta_list = [np.eye(p) * beta_value for _ in range(lag)]

    domain1_new_beta_list = list()
    domain2_new_beta_list = list()
    domain3_new_beta_list = list()

    for lag_beta, lag_interaction in zip(domain1_beta_list, domain1_lag_list):
        for x, y in lag_interaction:
            lag_beta[x, y] = beta_value
        domain1_new_beta_list.append(lag_beta)
    domain1_beta = np.hstack(domain1_new_beta_list)
    domain1_beta = make_var_stationary(domain1_beta)

    for lag_beta, lag_interaction in zip(domain2_beta_list, domain2_lag_list):
        for x, y in lag_interaction:
            lag_beta[x, y] = beta_value
        domain2_new_beta_list.append(lag_beta)
    domain2_beta = np.hstack(domain2_new_beta_list)
    domain2_beta = make_var_stationary(domain2_beta)

    for lag_beta, lag_interaction in zip(domain3_beta_list, domain3_lag_list):
        for x, y in lag_interaction:
            lag_beta[x, y] = beta_value
        domain3_new_beta_list.append(lag_beta)
    domain3_beta = np.hstack(domain3_new_beta_list)
    domain3_beta = make_var_stationary(domain3_beta)

    burn_in = 100
    domain1_errors = np.random.normal(scale=sd_list[0], size=(p, T+burn_in))
    domain2_errors = np.random.normal(scale=sd_list[1], size=(p, 2*T+burn_in))
    domain3_errors = np.random.normal(scale=sd_list[2], size=(p, 3*T+burn_in))
    domain1_X = np.zeros((p, T+burn_in))
    domain2_X = np.zeros((p, 2*T+burn_in))
    domain3_X = np.zeros((p, 3*T+burn_in))

    domain1_X[:, :lag] = domain1_errors[:, :lag] * 3 + np.random.randint(-5, 5, size=(p, lag))
    domain2_X[:, :lag] = domain2_errors[:, :lag] * 2 + np.random.randint(-5, 5, size=(p, lag))
    domain3_X[:, :lag] = domain3_errors[:, :lag] * 1 + np.random.randint(-5, 5, size=(p, lag))

    for t in range(lag, T + burn_in):
        if data_type == "nonlinear":
            domain1_X[:, t] = np.dot(domain1_beta, domain1_X[:, (t - lag):t].flatten(order='F') +
                                 0.02 * np.sin(domain1_X[:, (t - lag):t].flatten(order='F'))
                                     * domain1_X[:, (t - lag):t].flatten(order='F'))
        elif data_type == "linear":
            domain1_X[:, t] = np.dot(domain1_beta, domain1_X[:, (t - lag):t].flatten(order="F"))
        else:
            raise ValueError("Unknow data type")

        domain1_X[:, t] += + 5 * scale * domain1_errors[:, t - 1]
        domain1_X[:, t] -= 0.15

    for t in range(lag, 2*T + burn_in):
        if data_type == "nonlinear":
            domain2_X[:, t] = np.dot(domain2_beta, domain2_X[:, (t - lag):t].flatten(order='F') +
                                 0.04 * np.sin(domain2_X[:, (t - lag):t].flatten(order='F'))
                                     * domain2_X[:, (t - lag):t].flatten(order='F'))
        elif data_type == "linear":
            domain2_X[:, t] = np.dot(domain2_beta, domain2_X[:, (t - lag):t].flatten(order="F"))
        else:
            raise ValueError("Unknow data type")

        domain2_X[:, t] += + scale * domain2_errors[:, t - 1]
        # domain2_X[:, t] -= 0.015

    for t in range(lag, 2*T + burn_in):
        if data_type == "nonlinear":
            domain3_X[:, t] = np.dot(domain3_beta, domain3_X[:, (t - lag):t].flatten(order='F') +
                                 0.03 * np.sin(domain3_X[:, (t - lag):t].flatten(order='F'))
                                     * domain3_X[:, (t - lag):t].flatten(order='F'))
        elif data_type == "linear":
            domain3_X[:, t] = np.dot(domain3_beta, domain3_X[:, (t - lag):t].flatten(order="F"))
        else:
            raise ValueError("Unknow data type")

        domain3_X[:, t] += + 10 * scale * domain3_errors[:, t - 1]
        domain3_X[:, t] += 0.15

    new_domain2_data_list = []
    new_domain3_data_list = []
    # for i in list(random.sample(list(range(burn_in, 2 * T+burn_in)), T)):
    for i in range(burn_in, 2 * T+burn_in, 2):
        new_domain2_data_list.append(np.expand_dims(domain2_X.T[i], axis=0))
    new_domain2_data = np.concatenate(new_domain2_data_list, axis=0)

    # for i in list(random.sample(list(range(burn_in, 3 * T + burn_in)), T)):
    for i in range(burn_in, 2 * T+burn_in, 2):
        new_domain3_data_list.append(np.expand_dims(domain3_X.T[i], axis=0))
    new_domain3_data = np.concatenate(new_domain3_data_list, axis=0)

    return domain1_X.T[burn_in:], domain1_beta, base_GC, \
           new_domain2_data, domain2_beta, base_GC, \
           new_domain3_data, domain3_beta, base_GC


def simulate_three_domain_var(p, T, lag, sd_list, sparsity=0.2, beta_value=1.0, seed=0, scale=0.3,
                              swag_num=0, data_type="nonlinear", sample_type="nonrandom"):

    np.random.seed(seed)

    base_GC = np.eye(p, dtype=int)
    # base_GC = np.zeros(shape=[p, p])
    base_beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    base_record_list = []

    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        base_beta[i, choice] = random.uniform(-1, 1)
        base_GC[i, choice] = 1
        base_record_list.append((i, list(choice)))

    domain1_lag_list = random_generate_link_list_with_lag(deepcopy(base_record_list), lag=lag, swag_num=swag_num)
    domain2_lag_list = random_generate_link_list_with_lag(deepcopy(base_record_list), lag=lag, swag_num=swag_num)
    domain3_lag_list = random_generate_link_list_with_lag(deepcopy(base_record_list), lag=lag, swag_num=swag_num)

    # domain1_origin_beta_list = [np.eye(p) * random.uniform(-1, 1) for _ in range(lag)]
    # domain2_origin_beta_list = [np.eye(p) * random.uniform(-1, 1) for _ in range(lag)]
    # domain3_origin_beta_list = [np.eye(p) * random.uniform(-1, 1) for _ in range(lag)]
    domain1_origin_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    domain2_origin_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    domain3_origin_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    domain1_new_beta_list = []
    domain2_new_beta_list = []
    domain3_new_beta_list = []

    for lag_beta, lag_interaction in zip(domain1_origin_beta_list, domain1_lag_list):
        for x, y in lag_interaction:
            lag_beta[x, y] = beta_value
        domain1_new_beta_list.append(lag_beta)
    domain1_beta = np.hstack(domain1_new_beta_list)
    domain1_beta = make_var_stationary(domain1_beta)

    for lag_beta, lag_interaction in zip(domain2_origin_beta_list, domain2_lag_list):
        for x, y in lag_interaction:
            lag_beta[x, y] = random.uniform(-1, 1)
        domain2_new_beta_list.append(lag_beta)
    domain2_beta = np.hstack(domain2_new_beta_list)
    domain2_beta = make_var_stationary(domain2_beta)

    for lag_beta, lag_interaction in zip(domain3_origin_beta_list, domain3_lag_list):
        for x, y in lag_interaction:
            lag_beta[x, y] = random.uniform(-1, 1)
        domain3_new_beta_list.append(lag_beta)
    domain3_beta = np.hstack(domain3_new_beta_list)
    domain3_beta = make_var_stationary(domain3_beta)

    burn_in = 100
    domain1_errors = np.random.normal(scale=sd_list[0], size=(p, T+burn_in))
    domain2_errors = np.random.normal(scale=sd_list[1], size=(p, 2*T+burn_in))
    domain3_errors = np.random.normal(scale=sd_list[2], size=(p, 3*T+burn_in))
    domain1_X = np.zeros((p, T+burn_in))
    domain2_X = np.zeros((p, 2*T + burn_in))
    domain3_X = np.zeros((p, 3*T + burn_in))
    domain1_X[:, :lag] = domain1_errors[:, :lag]
    domain2_X[:, :lag] = domain2_errors[:, :lag]
    domain3_X[:, :lag] = domain3_errors[:, :lag]

    for t in range(lag, T+burn_in):
        if data_type == "linear":
            domain1_X[:, t] = np.dot(domain1_beta, domain1_X[:, (t - lag):t].flatten(order="F"))
        elif data_type == "nonlinear":
            domain1_X[:, t] = np.dot(domain1_beta, domain1_X[:, (t - lag):t].flatten(order='F') +
                                     0.02 * np.sin(domain1_X[:, (t - lag):t].flatten(order='F'))
                                     * domain1_X[:, (t - lag):t].flatten(order='F'))
        else:
            raise ValueError("Unknow data type")
        domain1_X[:, t] += + scale * domain1_errors[:, t - 1]
        domain1_X[:, t] -= 0.015

    for t in range(lag, T*2+burn_in):
        if data_type == "linear":
            domain2_X[:, t] = np.dot(domain2_beta, domain2_X[:, (t - lag):t].flatten(order="F"))
        elif data_type == "nonlinear":
            domain2_X[:, t] = np.dot(domain2_beta, domain2_X[:, (t - lag):t].flatten(order='F') +
                                     0.04 * np.sin(domain2_X[:, (t - lag):t].flatten(order='F'))
                                     * domain2_X[:, (t - lag):t].flatten(order='F'))
        else:
            raise ValueError("Unknow data type")
        domain2_X[:, t] += + scale * domain2_errors[:, t - 1]
        domain2_X[:, t] += 0

    for t in range(lag, T*3+burn_in):
        if data_type == "linear":
            domain3_X[:, t] = np.dot(domain3_beta, domain3_X[:, (t - lag):t].flatten(order="F"))
        elif data_type == "nonlinear":
            domain3_X[:, t] = np.dot(domain3_beta, domain3_X[:, (t - lag):t].flatten(order='F') +
                                     0.06 * np.sin(domain3_X[:, (t - lag):t].flatten(order='F'))
                                     * domain3_X[:, (t - lag):t].flatten(order='F'))
        else:
            raise ValueError("Unknow data type")
        domain3_X[:, t] += + scale * domain3_errors[:, t - 1]
        domain3_X[:, t] += 0.015

    if sample_type == "random":
        domain2_sample_idx = random.sample(list(range(burn_in, burn_in + 2 * T)), T)
        domain3_sample_idx = random.sample(list(range(burn_in, burn_in + 3 * T)), T)
    else:
        domain2_sample_idx = list(range(burn_in, burn_in + 2 * T, 2))
        domain3_sample_idx = list(range(burn_in, burn_in + 3 * T, 3))

    new_domain2_data_list = []
    new_domain3_data_list = []

    for i2, i3 in zip(domain2_sample_idx, domain3_sample_idx):
        new_domain2_data_list.append(np.expand_dims(domain2_X.T[i2], axis=0))
        new_domain3_data_list.append(np.expand_dims(domain3_X.T[i3], axis=0))
    new_domain2_data = np.concatenate(new_domain2_data_list, axis=0)
    new_domain3_data = np.concatenate(new_domain3_data_list, axis=0)

    return domain1_X.T[burn_in:], domain1_beta, base_GC, \
           new_domain2_data, domain2_beta, base_GC, \
           new_domain3_data, domain3_beta, base_GC


def random_generate_link_list_with_lag(alist, lag, swag_num):
    lag_list = [alist]
    for i in range(lag-1):
        next_lag_link_list = list()
        for j, _ in enumerate(lag_list[-1]):
            sample_num_upper = int(math.ceil(len(lag_list[-1][j][1]) * 0.3))
            sample_num_lower = int(math.ceil(len(lag_list[-1][j][1]) * 0.1))
            sample_num = random.randint(sample_num_lower, sample_num_upper)
            row_lag_link_list = random.sample(list(lag_list[-1][j][1]), sample_num)
            next_lag_link_list.append((j, row_lag_link_list))

        lag_list.append(next_lag_link_list)

    for i in range(lag-1):
        for j in range(swag_num):
            first_lag_list = lag_list[i]
            second_lag_list = lag_list[i+1]

            first_lag_link_list_index = random.randint(0, len(first_lag_list) - 1)
            second_lag_link_list_index = random.randint(0, len(second_lag_list) - 1)

            first_lag_link_swag = lag_list[i].pop(first_lag_link_list_index)
            second_lag_link_swag = lag_list[i+1].pop(second_lag_link_list_index)

            lag_list[i].append(second_lag_link_swag)
            lag_list[i+1].append(first_lag_link_swag)

    return lag_list


def simulate_two_domain_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0, scale=0.3,
                            exchange_num=1, src_sd=0.1, tgt_sd=0.3):
    if seed is not None:
        np.random.seed(seed)

    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    src_record_list = []

    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        # beta[i, choice] = beta_value
        beta[i, choice] = random.uniform(-1, 1)
        GC[i, choice] = 1
        src_record_list.append((i, choice))

    # src_lag_list = split_list(alist=src_record_list, group_num=lag)
    # # print(src_lag_list)
    # tgt_lag_list = swag_list(original_list=deepcopy(src_lag_list), exchange_num=exchange_num)

    src_lag_list = random_generate_link_list_with_lag(src_record_list, lag=lag)
    tgt_lag_list = random_generate_link_list_with_lag(src_record_list, lag=lag)

    # print(tgt_lag_list)

    src_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    tgt_beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    src_new_beta_list = list()
    tgt_new_beta_list = list()

    for lag_beta, lags_interaction in zip(src_beta_list, src_lag_list):
        for x, y in lags_interaction:
            lag_beta[x, y] = beta_value
        src_new_beta_list.append(lag_beta)
    src_beta = np.hstack(src_new_beta_list)
    src_beta = make_var_stationary(src_beta)

    for lag_beta, lags_interaction in zip(tgt_beta_list, tgt_lag_list):
        for x, y in lags_interaction:
            lag_beta[x, y] = beta_value
        tgt_new_beta_list.append(lag_beta)
    tgt_beta = np.hstack(tgt_new_beta_list)
    tgt_beta = make_var_stationary(tgt_beta)

    burn_in = 100
    src_errors = np.random.normal(scale=src_sd, size=(p, T + burn_in))
    tgt_errors = np.random.normal(scale=tgt_sd, size=(p, 2*T + burn_in))
    src_X = np.zeros((p, T + burn_in))
    tgt_X = np.zeros((p, 2 * T + burn_in))
    src_X[:, :lag] = src_errors[:, :lag]
    tgt_X[:, :lag] = tgt_errors[:, :lag]

    for t in range(lag, T + burn_in):
        # X[:, t] = \
        #     np.dot(beta, np.sin(X[:, (t - lag):t].flatten(order='F')) * X[:, (t - lag):t].flatten(order='F'))

        # src_X[:, t] = np.dot(src_beta, src_X[:, (t-lag):t].flatten(order="F"))
        src_X[:, t] = np.dot(src_beta, src_X[:, (t - lag):t].flatten(order='F') +
                             0.03 * np.sin(src_X[:, (t - lag):t].flatten(order='F')) * src_X[:, (t - lag):t].flatten(order='F'))
        src_X[:, t] += + scale * src_errors[:, t-1]
        src_X[:, t] -= 0.015
        # src_X[0, t] += 0.5
        # print(src_X[:, t].shape)
        # exit()

    for t in range(lag, T * 2 + burn_in):
        # tgt_X[:, t] = np.dot(tgt_beta, tgt_X[:, (t-lag):t].flatten(order="F"))
        tgt_X[:, t] = np.dot(src_beta, tgt_X[:, (t - lag):t].flatten(order='F') +
                             0.06 * np.sin(tgt_X[:, (t - lag):t].flatten(order='F')) * tgt_X[:, (t - lag):t].flatten(
            order='F'))
        tgt_X[:, t] += + scale * tgt_errors[:, t-1]
        tgt_X[:, t] += 0.015

    new_tgt_data_list = []
    for i in range(1, 2 * T, 2):
        new_tgt_data_list.append(np.expand_dims(tgt_X.T[i], axis=0))
    new_tgt_data = np.concatenate(new_tgt_data_list, axis=0)

    # return src_X.T[burn_in:], src_beta, GC, tgt_X.T[burn_in:], tgt_beta, GC
    return src_X.T[burn_in:], src_beta, GC, new_tgt_data, tgt_beta, GC


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0, scale=0.003):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate dataset.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in)) + 40
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + scale * errors[:, t-1]

    return X.T[burn_in:], beta, GC


def rotate(exchange_list, n):
    return exchange_list[n:] + exchange_list[:n]


def swag_list(original_list, exchange_num):
    exchange_list = list()
    new_list = list()
    for index, lag_list in enumerate(original_list):
        exchange_idx = random.sample(lag_list, exchange_num)
        _ = [lag_list.pop(lag_list.index(i)) for i in exchange_idx]
        exchange_list.append(exchange_idx)
        new_list.append(lag_list)

    exchange_list = rotate(exchange_list=exchange_list, n=1)
    target_lag_list = list()
    for exchange_idx, new_lag in zip(exchange_list, new_list):

        new_lag.extend(exchange_idx)
        target_lag_list.append(new_lag)

    return target_lag_list


def simulate_var_2(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0, scale=0.003):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    record_list = []
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1
        record_list.append((i, choice))

    lag_list = split_list(alist=record_list, group_num=lag)
    beta_list = [np.eye(p) * beta_value for _ in range(lag)]
    new_beta_list = list()
    for lag_beta, lags_interaction in zip(beta_list, lag_list):
        for x, y in lags_interaction:
            lag_beta[x, y] = beta_value
        new_beta_list.append(lag_beta)

    beta = np.hstack(new_beta_list)
    beta = make_var_stationary(beta)

    # Generate dataset.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in)) + 40
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] += + scale * errors[:, t - 1]

    return X.T[burn_in:], beta, GC


def simulate_nonlinear_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0, scale=0.01):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate dataset.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in)) + 10
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = \
            np.dot(beta, np.sin(X[:, (t-lag):t].flatten(order='F')) * X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + scale * errors[:, t-1]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def split_list(alist, group_num, shuffle=True):

    if shuffle:
        random.shuffle(alist)
    lag_list = [[] for _ in range(group_num)]

    for idx, list_idx in enumerate(alist):
        lag_list[idx % group_num].append(list_idx)

    return lag_list
