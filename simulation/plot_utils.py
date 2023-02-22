import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import seaborn as sns


def plot_full_graph(graph):
    fig, axarr = plt.subplots(1, 1, figsize=(16, 5))
    axarr.imshow(graph, cmap='Blues')
    axarr.set_title('GC actual')
    axarr.set_ylabel('Affected series')
    axarr.set_xlabel('Causal series')
    axarr.set_xticks([])
    axarr.set_yticks([])

    plt.show()


def onehot_trans(graph):
    xy = np.nonzero(graph)
    # print(xy)
    # exit()
    for x, y in zip(xy[0], xy[1]):
        graph[x, y] = 1

    return graph


def plot_two_domain_summary_graph(src_graph, tgt_graph, save_path=None):

    src_graph = onehot_trans(src_graph)
    tgt_graph = onehot_trans(tgt_graph)


    fig, axarr = plt.subplots(2, 2, figsize=(16, 10))
    axarr[0][0].imshow(src_graph, cmap="Blues")
    axarr[0][0].set_title('src full graph')
    axarr[0][0].set_ylabel('Affected series')
    axarr[0][0].set_xlabel('Causal series')

    axarr[1][0].imshow(tgt_graph, cmap="Blues")
    axarr[1][0].set_title('src full graph')
    axarr[1][0].set_ylabel('Affected series')
    axarr[1][0].set_xlabel('Causal series')

    # src_graph = onehot_trans(src_graph[:, :15] + src_graph[:, 15:])
    # tgt_graph = onehot_trans(tgt_graph[:, :15] + tgt_graph[:, 15:])
    # # print(src_graph == tgt_graph)
    # axarr[0][1].imshow(src_graph, cmap="Blues")
    # axarr[0][1].set_title('src summary graph')
    # axarr[0][1].set_ylabel('Affected series')
    # axarr[0][1].set_xlabel('Causal series')
    #
    # axarr[1][1].imshow(tgt_graph, cmap="Blues")
    # axarr[1][1].set_title('src summary graph')
    # axarr[1][1].set_ylabel('Affected series')
    # axarr[1][1].set_xlabel('Causal series')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    pass
    plt.close()


def plot_summary_graph(true_graph, pred_graph, content, file_path, is_show=False):
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
    axarr[0].imshow(true_graph, cmap='Blues')
    axarr[0].set_title('GC actual')
    axarr[0].set_ylabel('Affected series')
    axarr[0].set_xlabel('Causal series')
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    axarr[1].imshow(pred_graph, cmap='Blues', vmin=0, vmax=1, extent=(0, len(pred_graph), len(pred_graph), 0))
    axarr[1].set_title('GC estimated')
    axarr[1].set_ylabel('Affected series')
    axarr[1].set_xlabel('Causal series')
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    # Mark disagreements
    for i in range(len(pred_graph)):
        for j in range(len(pred_graph)):
            if pred_graph[i, j] != pred_graph[i, j]:
                rect = plt.Rectangle((j, i - 0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
                axarr[1].add_patch(rect)

    plt.text(0, 0, content)

    # plt.show()
    if is_show is None:
        plt.show()
    else:
        plt.savefig(file_path)


def plot_sns(pt):
    f, ax1 = plt.subplots(figsize=(16, 14), nrows=1)

    sns.heatmap(pt, annot=True, ax=ax1)

    plt.show()
    pass


def plot_data(data):
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
    axarr[0].plot(data)
    axarr[0].set_xlabel('T')
    axarr[0].set_title('Entire time series')
    axarr[1].plot(data[:50])
    axarr[1].set_xlabel('T')
    axarr[1].set_title('First 50 time points')
    plt.tight_layout()
    plt.show()
    pass


def plot_multi_domain_data(data_list, save_path=None):
    fig, axarr = plt.subplots(1, len(data_list), figsize=(8 * len(data_list), 5))
    # plt.ylim((-15, 15))
    for idx, data in enumerate(data_list):
        axarr[idx].plot(data)
        # axarr[idx].set_ylim((-15, 15))
        axarr[idx].set_xlabel('T')
        axarr[0].set_title('%d domain time series' % idx)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_two_domain_data(src_data, tgt_data, save_path=None):
    fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
    # plt.ylim((-10, 10))
    # my_y_ticks = np.arange(-3, 3, 1)
    # plt.yticks(my_y_ticks)
    axarr[0].plot(src_data)
    # axarr[0].set_ylim((-10, 10))
    # axarr[0].set_yticks(my_y_ticks)
    axarr[0].set_xlabel('T')
    axarr[0].set_title('Source domain time series')
    # axarr[1].set_ylim((-10, 10))
    axarr[1].plot(tgt_data)
    axarr[1].set_xlabel('T')
    axarr[1].set_title('Target domain time series')
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

