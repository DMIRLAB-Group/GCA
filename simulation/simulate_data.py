from synthetic import simulate_var, simulate_nonlinear_var, simulate_var_2, simulate_two_domain_var,\
    simulate_three_domain_var, simulate_three_domain_var_2
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import shutil


def plot_data(data_list):
    fig, axarr = plt.subplots(3, 1, figsize=(16, 15))
    plt.rc('font', size=30)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    length = 100
    axarr[0].plot(data_list[0][:length, ])
    axarr[0].set_xlabel('T', fontsize=5)
    axarr[0].tick_params(axis="y", labelsize=30)
    axarr[0].set_xticklabels([], fontsize=12)
    axarr[0].set_ylim(-25, 25)
    axarr[0].set_title('Domain 1 data')

    axarr[1].plot(data_list[1][:length, ])
    axarr[1].set_xlabel('T', fontsize=5)
    axarr[1].tick_params(axis="y", labelsize=30)
    axarr[1].set_xticklabels([], fontsize=12)
    axarr[1].set_title('Domain 2 data')
    axarr[1].set_ylim(-25, 25)
    # axarr[1].set_xticklabels(list(range(-25, 25, 5)), fontsize=12)

    axarr[2].plot(data_list[2][:length, ])
    axarr[2].set_xlabel('T', fontsize=5)
    axarr[2].tick_params(axis="y", labelsize=30)
    axarr[2].set_xticklabels([], fontsize=12)
    axarr[2].set_title('Domain 3 data')
    axarr[2].set_ylim(-25, 25)
    # axarr[2].set_xticklabels(list(range(-25, 25, 5)), fontsize=12)
    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':

    data_type = "linear"
    seed = 42
    lag = 5
    p = 20
    T = 50000

    data_path = "dataset/%s/%s_seed_%d_lag_%d_num_para_%d_DOMAIN.pkl" % \
                (data_type, data_type, seed, lag, p)

    domain1_data, domain1_full, domain1_summary, \
    domain2_data, domain2_full, domain2_summary, \
    domain3_data, domain3_full, domain3_summary \
        = simulate_three_domain_var_2(p=p, T=T, lag=lag, sparsity=0.15, seed=seed,
                                      swag_num=30, sd_list=[1.0, 20.0, 3.0], data_type="nonlinear")
                                      # swag_num=30, sd_list=[1.0, 15.0, 30.0], data_type="nonlinear")

    domain1_data_dict = {"data": domain1_data, "full_graph": domain1_full, "summary_graph": domain1_summary}
    domain2_data_dict = {"data": domain2_data, "full_graph": domain2_full, "summary_graph": domain2_summary}
    domain3_data_dict = {"data": domain3_data, "full_graph": domain3_full, "summary_graph": domain3_summary}
    data_dict = {"1": domain1_data_dict, "2": domain2_data_dict, "3": domain3_data_dict}

    plot_data([domain1_data, domain2_data, domain3_data])
    # exit()

    domain1_file = open(data_path.replace("DOMAIN", "1"), "wb")
    domain2_file = open(data_path.replace("DOMAIN", "2"), "wb")
    domain3_file = open(data_path.replace("DOMAIN", "3"), "wb")

    pickle.dump(domain1_data_dict, domain1_file)
    domain1_file.close()
    pickle.dump(domain2_data_dict, domain2_file)
    domain2_file.close()
    pickle.dump(domain3_data_dict, domain3_file)
    domain3_file.close()

    os.system("python dataset/genearte_no_causal_formate.py")
    src_path = "dataset/linear_sasa"
    tgt_path = "/home/lizijian/workspace/GC/SASA_3_simulate/datasets/linear"
    shutil.rmtree(tgt_path)
    shutil.copytree(src_path, tgt_path)
    # train_set = data_dict[str(args.src)]["data"]
    # src_full_graph = data_dict[str(args.src)]["full_graph"]
    # structures = data_dict[str(args.src)]["summary_graph"]
    #
    # test_set = data_dict[str(args.tgt)]["data"]
    # tgt_full_graph = data_dict[str(args.tgt)]["full_graph"]
    # tgt_structure = data_dict[str(args.tgt)]["summary_graph"]