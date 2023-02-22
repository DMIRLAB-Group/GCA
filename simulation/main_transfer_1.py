import argparse
import os
import numpy as np
import tensorboardX as tb
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import random
from copy import deepcopy
from PIL import Image
import pickle
from model.sernn_with_transfer_simulate import SENNGC
from utils import eval_causal_structure, eval_causal_structure_binary, data_iterator, load_air_dataset
from plot_utils import plot_two_domain_summary_graph, plot_two_domain_data


def normalization(data, num_vara):
    new_data_list = list()

    for i in range(num_vara):
        mean = np.mean(data[:, i])
        stddev = np.std(data[:, i])
        # mm_x = MinMaxScaler()
        # transfrom_data = mm_x.fit_transform(np.expand_dims(data[:, i], axis=1))
        transfrom_data = np.expand_dims((data[:, i] - mean) / (normal_coeff * stddev), axis=1)
        new_data_list.append(transfrom_data)

    new_data = np.concatenate(new_data_list, axis=1)
    return new_data


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def run_epoch(model: SENNGC, optimizer, src_iterator, tgt_iterator, label_tgtx, label_tgty, lmbd, batch_size,
              structure_similarity_restric, device, training_size, feature_id, hard=True):

    model.train()

    src_coeffs_final = torch.zeros([batch_size * training_size, lag, num_vars, num_vars]).to(device)
    tgt_coeffs_final = torch.zeros([batch_size * training_size, lag, num_vars, num_vars]).to(device)

    total_src_rmse = 0.0
    total_src_mae = 0.0
    total_tgt_rmse = 0.0
    total_tgt_mae = 0.0
    mse_fn = torch.nn.MSELoss()

    for steps in range(training_size):
        batch_src_x, batch_src_y = next(src_iterator)

        if steps == 0:
            batch_tgt_x, batch_tgt_y = label_tgtx, label_tgty
            feat_id = None
        else:
            batch_tgt_x, batch_tgt_y = next(tgt_iterator)
            feat_id = feature_id

        batch_src_input = Variable(torch.tensor(batch_src_x, dtype=torch.float)).to(device)
        batch_src_label = Variable(torch.tensor(batch_src_y, dtype=torch.float)).to(device)
        batch_tgt_input = Variable(torch.tensor(batch_tgt_x, dtype=torch.float)).to(device)
        batch_tgt_label = Variable(torch.tensor(batch_tgt_y, dtype=torch.float)).to(device)

        batch_src_pred, batch_src_coeffs, batch_src_all_lag_coeffs = \
            model.forward(inputs=batch_src_input, hard=hard, is_source=True)
        batch_tgt_pred, batch_tgt_coeffs, batch_tgt_all_lag_coeffs = \
            model.forward(inputs=batch_tgt_input, hard=hard, is_source=False)

        src_coeffs_final[steps * batch_size: (steps+1) * batch_size] = batch_src_coeffs
        tgt_coeffs_final[steps * batch_size: (steps+1) * batch_size] = batch_tgt_coeffs

        src_total_loss, src_rmse_loss, src_mae_loss = \
            model.calculate_loss(pred=batch_src_pred, targets=batch_src_label, lambdas=lmbd, feature_id=None,
                                 all_lag_structures=batch_src_all_lag_coeffs, coeffs=batch_src_coeffs)

        tgt_total_loss, tgt_rmse_loss, tgt_mae_loss = \
            model.calculate_loss(pred=batch_tgt_pred, targets=batch_tgt_label, lambdas=lmbd, feature_id=feat_id,
                                 all_lag_structures=batch_tgt_all_lag_coeffs, coeffs=batch_tgt_coeffs)

        src_label_mse_loss = mse_fn(batch_src_pred[:, -1], batch_src_label[:, -1])
        tgt_label_mse_loss = mse_fn(batch_tgt_pred[:, -1], batch_tgt_label[:, -1])

        total_src_rmse += src_rmse_loss.item()
        total_src_mae += src_mae_loss.item()
        total_tgt_rmse += tgt_rmse_loss.item()
        total_tgt_mae += tgt_mae_loss.item()

        alpha = 0.1
        beta = 0.1

        structure_similarity_loss = torch.mean(torch.abs(torch.mean(batch_src_coeffs, dim=1).detach() -
                                                         torch.mean(batch_tgt_coeffs, dim=1)))
        if steps != 0:
            total_loss = src_total_loss + tgt_total_loss + structure_similarity_restric * structure_similarity_loss\
                         + alpha * src_label_mse_loss
        else:
            total_loss = src_total_loss + tgt_total_loss + structure_similarity_restric * structure_similarity_loss\
                         + alpha * (src_label_mse_loss) + beta * tgt_label_mse_loss
        # total_loss = src_total_loss + tgt_total_loss + structure_similarity_restric * structure_similarity_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    total_src_rmse = total_src_rmse / training_size
    total_src_mae = total_src_mae / training_size
    total_tgt_rmse = total_tgt_rmse / training_size
    total_tgt_mae = total_tgt_mae / training_size

    return src_coeffs_final, tgt_coeffs_final, total_src_rmse, total_src_mae, total_tgt_rmse, total_tgt_mae


# def valid(model: SENNGC, test_feature, test_label, device, is_source, feature_id=-1, hard=True):
#
#     model.eval()
#
#     inputs = Variable(torch.tensor(test_feature, dtype=torch.float)).to(device)
#     preds, coeffs, _ = model.forward(inputs=inputs, hard=hard, is_source=is_source)
#     preds = preds.cpu().detach().numpy()
#     rmse = np.sqrt(((preds[:, feature_id] - test_label[:, feature_id]) ** 2).mean())
#     mape = (np.abs((preds[:, feature_id] - test_label[:, feature_id]) / test_label[:, feature_id])).mean()
#
#     return rmse, mape


def valid(model: SENNGC, test_feature, test_label, device, is_source, hard=True, pred_len=5):

    model.eval()
    pred_list = list()
    test_label = test_label[:, :, -1]

    for i in range(pred_len):
        i_input = deepcopy(test_feature[:, i: i + lag, :])
        for idx in range(1, min(len(pred_list), lag)+1):
            i_input[:, -idx, -1] = pred_list[-idx]

        inputs = Variable(torch.tensor(i_input, dtype=torch.float)).float().to(device)
        preds, coeffs, _ = model(inputs=inputs, hard=hard, is_source=is_source)
        pred_list.append(preds[:, -1].cpu().detach().numpy())

    total_pred = np.concatenate([np.expand_dims(pred, axis=1) for pred in pred_list], axis=1)
    # the shape of total_pred is [batch_size, pred_len, num_vars]
    rmse = ((total_pred - test_label) ** 2).mean()
    mape = np.abs((total_pred - test_label)).mean()

    return rmse, mape


def get_test_data(test_data, pred_len, lag):
    # the shape of test_data is [len x num_vars]
    feature_list = []
    label_list = []

    data_size = len(test_data)

    for i in range(data_size - pred_len - lag + 1):
        feature_list.append(np.expand_dims(test_data[i: i+lag+pred_len, :], axis=0))
        # the shape of a sample is [lag, num_vars]
        label_list.append(np.expand_dims(test_data[i+lag: i+lag+pred_len, :], axis=0))
        # the shape of a label is [1, pred_len]

    feature = np.concatenate(feature_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    return feature, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid Search")

    # Simulate dataset parameters
    parser.add_argument("--dataset", type=str, default="linear", help="dataset type")
    parser.add_argument("--p", type=int, default=20, help="Number of variables")
    parser.add_argument("--T", type=int, default=5000, help="Lenght of the time series")
    parser.add_argument("--sparsity", type=float, default=0.20, help="Sparsity of simulated dataset")
    parser.add_argument("--scale", type=float, default=0.01, help="Scale of the noise")
    parser.add_argument("--data_type", type=str, default="linear", help="data_type")
    parser.add_argument("--pred_len", type=int, default=10)
    parser.add_argument("--pred_feature_idx", type=int, default=14)
    # parser.add_argument("--src_sd", type=float, default=1.5)
    parser.add_argument("--src_sd", type=float, default=1.0)
    parser.add_argument("--tgt_sd", type=float, default=5.5)
    parser.add_argument("--is_src2tgt", type=int, default=0)

    # Model specification
    parser.add_argument("--src", type=int, default=2)
    parser.add_argument("--tgt", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=50, help="Number of epoch to train")
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--initial_lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--predictor_lr", type=float, default=0.003, help="Initial learning rate")
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--models", type=str, default="rnn", help="Model to train")
    parser.add_argument("--lag", type=int, default=5, help="Model order")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden_layer_size", type=int, default=5, help="Number of units in the hidden layer")
    # parser.add_argument("--structure_similarity_restric", type=float, default=1.5)
    parser.add_argument("--structure_similarity_restric", type=float, default=0.25)
    parser.add_argument("--domain_dim", type=int, default=3)

    parser.add_argument("--seed", type=int, default=10, help="Random seed")

    # parser.add_argument("--lambdas", type=float, default=0.020)
    # parser.add_argument("--lambdas", type=float, default=0.015)
    # parser.add_argument("--lambdas", type=float, default=0.03)
    # parser.add_argument("--lambdas", type=float, default=0.15)
    parser.add_argument("--lambdas", type=float, default=0.06)
    # parser.add_argument("--lambdas", type=float, default=1.395)
    parser.add_argument("--normal_coeff", type=int, default=1)
    # parser.add_argument("--lambdas", type=float, default=0.0020)
    args = parser.parse_args()
    normal_coeff = args.normal_coeff

    data_path = "dataset/%s/%s_seed_%d_lag_%d_num_para_%d_DOMAIN.pkl" % \
                (args.data_type, args.data_type, 42, args.lag, args.p)
    print("Use generated dataset")
    print(data_path.replace("DOMAIN", str(args.src)))
    print(data_path.replace("DOMAIN", str(args.tgt)))
    train_data_file = open(data_path.replace("DOMAIN", str(args.src)), "rb")
    test_data_file = open(data_path.replace("DOMAIN", str(args.tgt)), "rb")

    train_data_dict = pickle.load(train_data_file)
    test_data_dict = pickle.load(test_data_file)

    train_set = train_data_dict["data"]
    src_full_graph = train_data_dict["full_graph"]
    structures = train_data_dict["summary_graph"]
    test_set = test_data_dict["data"]
    tgt_full_graph = test_data_dict["full_graph"]
    tgt_structure = test_data_dict["summary_graph"]

    train_data_file.close()
    test_data_file.close()

    plot_train_set = deepcopy(train_set)
    plot_test_set = deepcopy(test_set)

    src_total_set = normalization(train_set, args.p)
    tgt_total_set = normalization(test_set, args.p)

    print(src_total_set.shape)
    print(tgt_total_set.shape)

    src_train_size = int(0.5 * len(src_total_set))
    src_val_size = int(0.5 * (len(src_total_set) - src_train_size))
    tgt_train_size = int(0.5 * len(tgt_total_set))
    tgt_val_size = int(0.5 * (len(tgt_total_set) - tgt_train_size))

    src_train_set = src_total_set[: src_train_size, :]
    tgt_train_set = tgt_total_set[: tgt_train_size, :]

    src_valid_set = src_total_set[src_train_size: src_train_size + src_val_size, :]
    tgt_valid_set = tgt_total_set[tgt_train_size: tgt_train_size + tgt_val_size, :]

    src_test_set = src_total_set[src_train_size + src_val_size:, :]
    tgt_test_set = tgt_total_set[tgt_train_size + tgt_val_size:, :]

    direction = "%d->%d" % (args.src, args.tgt)

    lag = args.lag
    num_vars = args.p
    num_hidden_layers = args.num_hidden_layers
    num_epochs = args.epoch
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    seed = args.seed
    hidden_layer_size = args.hidden_layer_size
    lambdas = args.lambdas

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda")

    data_path = os.path.join("%s_lag_%d") % (args.dataset, args.lag)
    exp_path = "beyondsasa"

    if not os.path.exists(data_path):
        os.makedirs(os.path.join("pred_logs", data_path, exp_path, str(args.pred_feature_idx), direction),
                    exist_ok=True)
    else:
        os.remove(os.path.join("pred_logs", data_path, exp_path, str(args.pred_feature_idx), direction))
        os.makedirs(os.path.join("pred_logs", data_path, exp_path, str(args.pred_feature_idx), direction),
                    exist_ok=True)

    src_writer = tb.SummaryWriter(os.path.join("pred_logs", data_path, exp_path,
                                               str(args.pred_feature_idx), direction, "Source"))
    tgt_writer = tb.SummaryWriter(os.path.join("pred_logs", data_path, exp_path,
                                               str(args.pred_feature_idx), direction, "Target"))
    test_writer = tb.SummaryWriter(os.path.join("pred_logs", data_path, exp_path,
                                                str(args.pred_feature_idx), direction, "Test"))

    image_save_path = os.path.join("pred_logs", data_path, exp_path,
                                   str(args.pred_feature_idx), direction)

    data_image_path = os.path.join(image_save_path, "image.png")
    structure_image_path = os.path.join(image_save_path, "domain.png")
    plot_two_domain_summary_graph(src_graph=structures, tgt_graph=tgt_structure, save_path=structure_image_path)
    plot_two_domain_data(src_data=plot_train_set, tgt_data=plot_test_set, save_path=data_image_path)
    data_image = Image.open(data_image_path)
    structure_image = Image.open(structure_image_path)
    # src_writer.add_image("two domain data", np.transpose(np.array(data_image), [2, 0, 1]))
    # src_writer.add_image("two domain structure", np.transpose(np.array(structure_image), [2, 0, 1]))

    src_x, src_y = sliding_windows(src_train_set, lag)
    tgt_x, tgt_y = sliding_windows(tgt_train_set, lag)
    # test_x, test_y = sliding_windows(data=tgt_test_set, seq_length=lag)
    # val_x, val_y = sliding_windows(data=tgt_valid_set, seq_length=lag)

    test_x, test_y = get_test_data(test_data=tgt_test_set, pred_len=args.pred_len, lag=lag)
    val_x, val_y = get_test_data(test_data=tgt_valid_set, pred_len=args.pred_len, lag=lag)

    tgt_x = list(tgt_x.tolist())
    tgt_y = list(tgt_y.tolist())

    src_x = list(src_x.tolist())
    src_y = list(src_y.tolist())

    label_tgtx = list()
    label_tgty = list()

    label_data_size = batch_size
    random_idx = random.sample(list(range(len(tgt_x))), label_data_size)
    for _ in range(label_data_size):
        idx = random.choice(list(range(len(tgt_x))))
        x = tgt_x.pop(idx)
        y = tgt_y.pop(idx)
        label_tgtx.append(x)
        label_tgty.append(y)

    src_x = np.asarray(src_x)
    src_y = np.asarray(src_y)
    label_tgtx = np.asarray(label_tgtx)
    label_tgty = np.asarray(label_tgty)
    tgt_x = np.asarray(tgt_x)
    tgt_y = np.asarray(tgt_y)

    model = SENNGC(num_vars=num_vars, order=lag, hidden_layer_size=hidden_layer_size,
                   num_hidden_layer=num_hidden_layers, device=device, domain_dim=args.domain_dim)

    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=initial_lr, betas=(beta_1, beta_2))

    best_rmse = 1000
    best_mae = 1000
    record_rmse = 1000
    record_mae = 1000

    src_iterator = data_iterator(x=src_x, y=src_y, batch_size=batch_size)
    tgt_iterator = data_iterator(x=tgt_x, y=tgt_y, batch_size=batch_size)
    src_size = (src_x.shape[0] // batch_size) + 1

    for epoch in range(num_epochs):
        src_coeffs, tgt_coeffs, src_rmse, src_mae, tgt_rmse, tgt_mae = \
            run_epoch(model=model, optimizer=optimizer, src_iterator=src_iterator, tgt_iterator=tgt_iterator,
                      label_tgtx=label_tgtx, label_tgty=label_tgty, lmbd=lambdas, batch_size=batch_size,
                      structure_similarity_restric=args.structure_similarity_restric, device=device,
                      training_size=src_size, feature_id=args.pred_feature_idx)

        structure_image_path = os.path.join(image_save_path, "domain_%d.png" % epoch)

        src_coeffs = src_coeffs.cpu().detach()
        src_causal_structure_estimate = \
            torch.max(torch.median(torch.abs(src_coeffs), dim=0)[0], dim=0)[0].cpu().numpy()

        tgt_coeffs = tgt_coeffs.cpu().detach()
        tgt_causal_structure_estimate = \
            torch.max(torch.median(torch.abs(tgt_coeffs), dim=0)[0], dim=0)[0].cpu().numpy()

        plot_two_domain_summary_graph(src_graph=src_causal_structure_estimate,
                                      tgt_graph=tgt_causal_structure_estimate,
                                      save_path=structure_image_path)

        src_acc_l, src_bal_acc_l, src_prec_l, src_rec_l \
            = eval_causal_structure_binary(a_true=structures, a_pred=src_causal_structure_estimate)
        src_auroc_l, src_auprc_l = eval_causal_structure(a_true=structures, a_pred=src_causal_structure_estimate)

        src_writer.add_scalar("rmse", src_rmse, epoch)
        src_writer.add_scalar("mae", src_mae, epoch)
        src_writer.add_scalar("acc", src_acc_l, epoch)
        src_writer.add_scalar("bal_acc", src_bal_acc_l, epoch)
        src_writer.add_scalar("prec", src_prec_l, epoch)
        src_writer.add_scalar("rec", src_rec_l, epoch)
        src_writer.add_scalar("auroc", src_auroc_l, epoch)
        src_writer.add_scalar("auprc", src_auprc_l, epoch)
        # src_writer.add_scalar("structure_similarity_loss", structure_similarity_loss, epoch)

        src_content = "Source Epoch: " + str(epoch) + " RMSE: " + str(src_rmse) + "; Acc.: " + str(
            np.round(src_acc_l, 4)) + "; Bal. Acc.: " + \
                  str(np.round(src_bal_acc_l, 4)) + "; Prec.: " + str(np.round(src_prec_l, 4)) + "; Rec.: " + \
                  str(np.round(src_rec_l, 4)) + "; AUROC: " + str(np.round(src_auroc_l, 4)) + "; AUPRC: " + \
                  str(np.round(src_auprc_l, 4))
        print(src_content)

        tgt_acc_l, tgt_bal_acc_l, tgt_prec_l, tgt_rec_l \
            = eval_causal_structure_binary(a_true=tgt_structure, a_pred=tgt_causal_structure_estimate)
        tgt_auroc_l, tgt_auprc_l = eval_causal_structure(a_true=tgt_structure, a_pred=tgt_causal_structure_estimate)

        tgt_content = "Target Epoch: " + str(epoch) + " RMSE: " + str(tgt_rmse) + "; Acc.: " + str(
            np.round(tgt_acc_l, 4)) + "; Bal. Acc.: " + \
                  str(np.round(tgt_bal_acc_l, 4)) + "; Prec.: " + str(np.round(tgt_prec_l, 4)) + "; Rec.: " + \
                  str(np.round(tgt_rec_l, 4)) + "; AUROC: " + str(np.round(tgt_auroc_l, 4)) + "; AUPRC: " + \
                  str(np.round(tgt_auprc_l, 4))
        print(tgt_content)
        test_rmse, test_mae = \
            valid(model=model, test_feature=test_x, test_label=test_y, device=device,
                  is_source=False, pred_len=args.pred_len)
        # valid(model: SENNGC, test_feature, test_label, device, is_source, feature_id, hard = True):
        tr_test_rmse, tr_test_mae = \
            valid(model=model, test_feature=val_x, test_label=val_y, device=device,
                  is_source=False, pred_len=args.pred_len)
        # test_writer.add_scalar("rmse", test_rmse, epoch)
        # test_writer.add_scalar("mae", test_mae, epoch)

        if best_rmse > tr_test_rmse:
            best_rmse = tr_test_rmse
            best_mae = tr_test_mae
            record_rmse = test_rmse
            record_mae = test_mae

        print("Epoch:%d RMSE: %g\tMAE: %g\tBest RMSE: %g\tBest MAE:%g" %
              (epoch, tr_test_rmse, tr_test_mae, best_rmse, best_mae))
        print("Train RMSE: %g\tTrain MAE: %g" % (record_rmse, record_mae))
        # print(best_rmse)
        print()
    record_file_path = os.path.join("pred_logs", data_path, exp_path, "result_%d.txt" % args.domain_dim)
    record_file = open(record_file_path, mode="a")

    record_hypm = "seed_%d_batch_size_%d_lr_%g_stru_sim_%g_lmbd_%g" % \
                  (seed, batch_size, initial_lr, args.structure_similarity_restric, lambdas)
    # record_content = "%s\tfeature_id:%d\tRMSE:%g\tMAE:%g\n" % (direction, args.pred_feature_idx, best_rmse, best_mae)
    record_content = "%s\tfeature_id:%d\tRMSE:%g\tMAE:%g\n" % (
    direction, args.pred_feature_idx, best_rmse, best_mae)
    print(record_hypm, file=record_file)
    print(record_content, file=record_file)
    print(record_content)
