import os

seed_list = [10, 30, 12, 7, 90]

domain_list = ["1", "2", "3"]

domain_dim_list = [1]

command_format = "python main_transfer_1.py --src %s --tgt %s --seed %d --domain_dim %d"
command_list = list()

for domain_dim in domain_dim_list:
    for seed in seed_list:
        for src in domain_list:
            for tgt in domain_list:
                if src != tgt:
                    command = command_format % (src, tgt, seed, domain_dim)
                    command_list.append(command)
                    os.system(command)

