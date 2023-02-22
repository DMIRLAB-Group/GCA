import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.distributions import kl_divergence


class SENNGC(nn.Module):

    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layer: int,
                 device: torch.device, domain_dim):

        super(SENNGC, self).__init__()

        self.src_pred_list = nn.ModuleList()
        self.src_variables_list = Parameter(torch.Tensor(order, domain_dim))
        self.tgt_variables_list = Parameter(torch.Tensor(order, domain_dim))

        self.src_pred_variables_list = Parameter(torch.Tensor(order, domain_dim))
        self.tgt_pred_variables_list = Parameter(torch.Tensor(order, domain_dim))

        init.normal_(self.src_variables_list, mean=0.0, std=0.05)
        init.normal_(self.tgt_variables_list, mean=0.0, std=0.05)
        init.normal_(self.src_pred_variables_list, mean=0.0, std=0.05)
        init.normal_(self.tgt_pred_variables_list, mean=0.0, std=0.05)

        self.coeff_nets = nn.ModuleList()

        #  Encoder
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_vars * order + domain_dim, hidden_layer_size))]
            if num_hidden_layer > 1:
                for j in range(num_hidden_layer - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size)))
            modules.extend(nn.Sequential(
                # nn.BatchNorm1d(hidden_layer_size),
                # nn.Dropout(0.05),
                # nn.Linear(hidden_layer_size, hidden_layer_size),
                # nn.BatchNorm1d(hidden_layer_size),
                # nn.Dropout(0.05),
                # nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.BatchNorm1d(hidden_layer_size),
                nn.Dropout(0.05),
                nn.Linear(hidden_layer_size, num_vars * num_vars * 2)))
            self.coeff_nets.append(nn.Sequential(*modules))

        #  Decoder
        self.lag_pred_list = nn.ModuleList()
        for k in range(order):
            self.lag_pred_list.append(nn.ModuleList())
            for j in range(num_vars):
                self.lag_pred_list[k].append(nn.Sequential(nn.Linear(num_vars + domain_dim, 2*num_vars),
                                                           # nn.ReLU(),
                                                           # nn.Dropout(0.08),
                                                           # nn.Linear(2*num_vars, num_vars),
                                                           nn.ReLU(),
                                                           nn.Dropout(0.2),
                                                           nn.Linear(2 * num_vars, 1)
                                                           ))
                init.normal_(self.lag_pred_list[k][j][0].weight, mean=0.0, std=0.01)
                init.normal_(self.lag_pred_list[k][j][-1].weight, mean=0.0, std=0.01)
                # init.kaiming_normal_(self.lag_pred_list[k][j][0].weight)
                # init.kaiming_normal_(self.lag_pred_list[k][j][-1].weight)


        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layer
        self.device = device
        self.base_loss = MSELoss()
        self.mae_loss = torch.nn.L1Loss()

    def forward(self, inputs: torch.Tensor, hard, is_source):

        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")
        batch_size = inputs.shape[0]

        coeffs = torch.zeros(size=[batch_size, self.order, self.num_vars, self.num_vars]).to(self.device)
        all_lag_structures = torch.zeros([batch_size, self.order, self.num_vars, self.num_vars, 2]).to(self.device)
        preds = torch.zeros([batch_size, self.num_vars]).to(self.device)

        pred_k_structure_list = list()

        if is_source:
            domain_variable_list = self.src_variables_list
            domain_pred_variable_list = self.src_pred_variables_list
        else:
            domain_variable_list = self.tgt_variables_list
            domain_pred_variable_list = self.tgt_pred_variables_list

        for k in range(self.order):

            modified_input_list = list()
            for i in range(self.order):
                if i < len(pred_k_structure_list):
                    modified_input = torch.matmul(pred_k_structure_list[i], inputs[:, i, :].unsqueeze(-1)).squeeze()
                else:
                    modified_input = inputs[:, i, :]

                modified_input_list.append(modified_input)

            total_modified_input = torch.cat(modified_input_list, dim=-1)
            domain_variable = domain_variable_list[k, :].unsqueeze(0).repeat([batch_size, 1])
            domain_predict_variable = domain_pred_variable_list[k, :].unsqueeze(0).repeat([batch_size, 1])

            total_modified_input = torch.cat([total_modified_input, domain_variable], dim=-1)

            coeff_net_k = self.coeff_nets[k]
            coeff_k = coeff_net_k(total_modified_input)

            coeff_k = torch.reshape(coeff_k, [batch_size, self.num_vars, self.num_vars, 2])
            k_lag_structure = coeff_k
            k_lag_structure = gumbel_softmax(logits=k_lag_structure, latent_dim=self.num_vars * self.num_vars,
                                             temperature=0.05, hard=hard)
            k_lag_structure = torch.reshape(k_lag_structure, [batch_size, self.num_vars, self.num_vars, 2])
            all_lag_structures[:, k, :, :, :] = k_lag_structure
            coeffs[:, k, :, :] = k_lag_structure[:, :, :, 0]
            pred_k_structure_list.append(k_lag_structure[:, :, :, 0])

            lag_pred_layer_list = self.lag_pred_list[k]
            mask_input = k_lag_structure[:, :, :, 0] * inputs[:, k, :].unsqueeze(1).repeat([1, self.num_vars, 1])
            for var_idx in range(self.num_vars):

                lag_pred = lag_pred_layer_list[var_idx](torch.cat([mask_input[:, var_idx, :],
                                                                   domain_predict_variable], dim=-1)).squeeze()
                preds[:, var_idx] += lag_pred

        return preds, coeffs, all_lag_structures

    def calculate_kld(self, all_lag_structures: torch.tensor):
        posterior_dist = torch.distributions.Categorical(logits=all_lag_structures)
        prior_dist = torch.distributions.Categorical(probs=torch.ones_like(all_lag_structures) * 0.1)

        KLD = kl_divergence(posterior_dist, prior_dist).mean()

        return KLD

    def get_parameter_num(self):
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_num

    def calculate_loss(self, pred, targets, coeffs, lambdas, all_lag_structures, feature_id=None):
        if feature_id is not None:
            pred = torch.cat([pred[:, :feature_id], pred[:, feature_id+1:]], dim=-1)
            targets = torch.cat([targets[:, :feature_id], targets[:, feature_id+1:]], dim=-1)

        mse_loss = self.base_loss(pred, targets)
        mae_loss = self.mae_loss(pred, targets)

        sparsity_penalty = 0.5 * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                           0.5 * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

        KLD = self.calculate_kld(all_lag_structures=all_lag_structures)

        total_loss = mse_loss + lambdas * sparsity_penalty + KLD

        return total_loss, torch.sqrt(mse_loss), mae_loss


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim=2, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    latent_dim: var_num * var_num
    categorical_dim=1
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)



