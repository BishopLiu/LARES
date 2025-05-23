import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender
from layers import BSARecEncoder, FMLPEncoder, MoEAdaptorLayer
import copy


class ContrastiveLoss(nn.Module):
    def __init__(self, tau, sem_func):
        super().__init__()
        self.tau = tau
        self.sem_func = sem_func

    def forward(self, x, y):
        if self.sem_func == 'cos':
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        
        B = x.shape[0]
        
        logits = torch.matmul(x, y.transpose(0, 1)) / self.tau
        labels = torch.arange(B, device=x.device, dtype=torch.long)
        
        loss = F.cross_entropy(logits, labels)

        return loss
    

class BaseRecommender(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']

        # load parameters info
        self.n_pre_layers = config['n_pre_layers']
        self.n_core_layers = config['n_core_layers']
        self.sampling_scheme = config['sampling_scheme']
        self.mean_recurrence = config['mean_recurrence']
        self.state_init_method = config['state_init_method']
        self.state_std = config['state_std']
        self.state_scale = config['state_scale']
        self.adapter_type = config['adapter_type']
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.aug_types = config['aug_types']
        self.sem_func = config['sem_func']  # [dot, cos]
        self.same_step = config['same_step']
        
        self.AUG_ITEM_SEQ = "sem_aug"
        self.AGU_ITEM_SEQ_LEN = "sem_aug_length"

        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = self.hidden_dropout_prob
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        if self.adapter_type == "concat":
            self.adapter = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)
        elif self.adapter_type == "add":
            self.adapter = lambda x: x
        elif self.adapter_type == "linear":
            self.adapter = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        else:
            raise ValueError(f"Unknown adapter type: {self.adapter_type}")

        self.pre_encoder = None
        self.core_encoder = None

        self.layernorm_1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layernorm_2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.test_recurrence = self.mean_recurrence

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @torch.no_grad()
    def randomized_iteration_sampler(self):

        if self.training:
            if "uniform" in self.sampling_scheme:
                t = torch.randint(low=1, high=1 + self.mean_recurrence * 2, size=(1,), )
            elif "poisson-lognormal" in self.sampling_scheme:
                sigma = 0.5
                mu = math.log(self.mean_recurrence) - (sigma ** 2 / 2)
                rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, )
                t = torch.poisson(torch.tensor([rate], dtype=torch.float), ) + 1
                t = torch.minimum(t, torch.as_tensor(3 * self.mean_recurrence))
            elif "poisson-unbounded" in self.sampling_scheme:
                t = torch.poisson(torch.tensor([self.mean_recurrence], dtype=torch.float), )
            elif "poisson-bounded" in self.sampling_scheme:
                t = torch.minimum(
                    torch.poisson(torch.tensor([self.mean_recurrence], dtype=torch.float), ),
                    torch.as_tensor(2 * self.mean_recurrence),
                )
            elif "non-recurrent" in self.sampling_scheme:
                t = 1
            elif "constant" in self.sampling_scheme:
                t = torch.as_tensor(self.mean_recurrence)
        else:
            t = torch.as_tensor(self.mean_recurrence)

        return t.squeeze().to(dtype=torch.long)

    def initialize_state(self, hidden_states):
        # zero / normal_zero / normal
        x = torch.zeros_like(hidden_states)

        if self.state_init_method == "normal":
            torch.nn.init.trunc_normal_(x, mean=0.0, std=self.state_std, a=-3 * self.state_std, b=3 * self.state_std)
        elif self.state_init_method == "normal_zero":
            if self.training:
                torch.nn.init.trunc_normal_(x, mean=0.0, std=self.state_std, a=-3 * self.state_std, b=3 * self.state_std)

        return self.state_scale * x
            
    def forward(self, item_seq, item_seq_len, input_states=None, return_all_states=False, num_steps=-1):

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.layernorm_1(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        all_states = []
        pre_output = self.pre_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1]

        if input_states is None:
            states = self.initialize_state(pre_output)
        else:
            states = input_states

        
        if num_steps < 0:
            if self.training:
                    num_steps = self.randomized_iteration_sampler()
            else:
                num_steps = torch.as_tensor(self.test_recurrence)


        for step in range(num_steps):
            if self.adapter_type == "concat":
                states = self.adapter(torch.cat([states, pre_output], dim=-1))
            elif self.adapter_type == "add":
                states = (states + pre_output) / 2
            elif self.adapter_type == "linear":
                states = F.sigmoid(self.adapter) * states + (1 - F.sigmoid(self.adapter)) * pre_output
            states = self.layernorm_2(states)
            states = self.core_encoder(states, extended_attention_mask, output_all_encoded_layers=True)[-1]
            all_states.append(states)

        all_outputs = []
        per_step_logps = None
        last_output = self.gather_indexes(states, item_seq_len - 1)  # [B H]

        if return_all_states:
            for states in all_states:
                out = self.gather_indexes(states, item_seq_len - 1)
                all_outputs.append(out)

            all_outputs = torch.stack(all_outputs, dim=1)  # [B, T, H]
            per_step_logits = torch.matmul(all_outputs, self.item_embedding.weight.transpose(0, 1))  # [B, T, N]
            per_step_logps = torch.log_softmax(per_step_logits, dim=-1)  # [B, T, N]

        return last_output, per_step_logps, all_outputs

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        aug_item_seq = interaction[self.AUG_ITEM_SEQ]
        aug_item_seq_len = interaction[self.AGU_ITEM_SEQ_LEN]

        seq_output, _, _ = self.forward(item_seq, item_seq_len)
        
        test_item_emb = self.item_embedding.weight
        if self.sem_func == 'cos':
            test_item_emb = F.normalize(test_item_emb, dim=-1)
            seq_output = F.normalize(seq_output, dim=-1)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.tau
        else:
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        n_step = self.randomized_iteration_sampler() if self.same_step else -1

        aug_seq_output1, _, all_step_output = self.forward(item_seq, item_seq_len, return_all_states=True, num_steps=n_step)
        aug_seq_output2, _, _ = self.forward(aug_item_seq, aug_item_seq_len, num_steps=n_step)

        cl_func = ContrastiveLoss(tau=self.tau, sem_func=self.sem_func)

        loss += self.alpha*(cl_func(aug_seq_output1, aug_seq_output2)+cl_func(aug_seq_output2, aug_seq_output1))/2

        B, T = all_step_output.shape[0], all_step_output.shape[1]
        if T > 1:
            idx = torch.randint(0, T, (1,), device=self.device)[0]
            selected_output = all_step_output[:, idx]
            loss += self.gamma*(cl_func(aug_seq_output1, selected_output)+cl_func(selected_output, aug_seq_output1))/2

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        input_states = interaction["input_states"] if "input_states" in interaction else None
        test_item = interaction[self.ITEM_ID]
        seq_output, _, _ = self.forward(item_seq, item_seq_len, input_states)
        test_item_emb = self.item_embedding(test_item)
        if self.sem_func == 'cos':
            test_item_emb = F.normalize(test_item_emb, dim=-1)
            seq_output = F.normalize(seq_output, dim=-1)
            scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.tau
        else:
            scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        input_states = interaction["input_states"] if "input_states" in interaction else None
        seq_output, _, _ = self.forward(item_seq, item_seq_len, input_states)
        test_items_emb = self.item_embedding.weight
        if self.sem_func == 'cos':
            test_items_emb = F.normalize(test_items_emb, dim=-1)
            seq_output = F.normalize(seq_output, dim=-1)
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) / self.tau
        else:
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class LARES(BaseRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.n_heads = config['n_heads']
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer

        self.pre_encoder = TransformerEncoder(
            n_layers=self.n_pre_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.core_encoder = TransformerEncoder(
            n_layers=self.n_core_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len, input_states=None, return_all_states=False, num_steps=-1):
        return super().forward(item_seq, item_seq_len, input_states, return_all_states, num_steps)


class BSARec_plus(BaseRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.n_heads = config['n_heads']
        self.inner_size = config['inner_size']
        self.c = config['c']
        self.a = config['a']
        self.pre_encoder = BSARecEncoder(
            n_layers=self.n_pre_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            alpha=self.a,
            c=self.c,
        )

        self.core_encoder = BSARecEncoder(
            n_layers=self.n_core_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            alpha=self.a,
            c=self.c,
        )
        
        self.apply(self._init_weights)
    
    def forward(self, item_seq, item_seq_len, input_states=None, return_all_states=False, num_steps=-1):
        return super().forward(item_seq, item_seq_len, input_states, return_all_states, num_steps)


class FMLPRec_plus(BaseRecommender):
    def __init__(self, config, dataset):
        super(FMLPRec_plus, self).__init__(config, dataset)
        self.pre_encoder = FMLPEncoder(
            n_layers=self.n_pre_layers,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_his_len=self.max_seq_length,
        )

        self.core_encoder = FMLPEncoder(
            n_layers=self.n_core_layers,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_his_len=self.max_seq_length,
        )
        
        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len, input_states=None, return_all_states=False, num_steps=-1):
        return super().forward(item_seq, item_seq_len, input_states, return_all_states, num_steps)


class TedRec_plus(LARES):
    def __init__(self, config, dataset):
        super(TedRec_plus, self).__init__(config, dataset)
        self.temperature = config['temperature']
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        self.item_gating = nn.Linear(self.hidden_size, 1)
        self.fusion_gating = nn.Linear(self.hidden_size, 1)

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
            self.max_seq_length
        )

        self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)

        self.item_gating.weight.data.normal_(mean = 0, std = 0.02)
        self.fusion_gating.weight.data.normal_(mean = 0, std = 0.02)

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion
        """
        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm='ortho')
        item_fft = torch.fft.rfft(item_emb, dim=1, norm='ortho')

        complext_weight = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * complext_weight, n = feature_emb.shape[1], dim = 1, norm = 'ortho')
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n = feature_emb.shape[1], dim = 1, norm = 'ortho')

        item_gate_w = self.item_gating(item_conv)
        fusion_gate_w = self.fusion_gating(fusion_conv)

        contextual_emb = 2 * (item_conv * torch.sigmoid(item_gate_w) + fusion_conv * torch.sigmoid(fusion_gate_w))
        return contextual_emb
    
    def forward(self, item_seq, item_seq_len, input_states=None, return_all_states=False, num_steps=-1):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_text_emb = self.moe_adaptor(self.plm_embedding(item_seq))
        input_emb = self.contextual_convolution(self.item_embedding(item_seq), item_text_emb)
        input_emb = input_emb + position_embedding
        input_emb = self.layernorm_1(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        all_states = []
        pre_output = self.pre_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1]

        if input_states is None:
            states = self.initialize_state(pre_output)
        else:
            states = input_states

        
        if num_steps < 0:
            if self.training:
                    num_steps = self.randomized_iteration_sampler()
            else:
                num_steps = torch.as_tensor(self.test_recurrence)


        for step in range(num_steps):
            if self.adapter_type == "concat":
                states = self.adapter(torch.cat([states, pre_output], dim=-1))
            elif self.adapter_type == "add":
                states = (states + pre_output) / 2
            elif self.adapter_type == "linear":
                states = F.sigmoid(self.adapter) * states + (1 - F.sigmoid(self.adapter)) * pre_output
            states = self.layernorm_2(states)
            states = self.core_encoder(states, extended_attention_mask, output_all_encoded_layers=True)[-1]
            all_states.append(states)

        all_outputs = []
        per_step_logps = None
        last_output = self.gather_indexes(states, item_seq_len - 1)  # [B H]

        if return_all_states:
            for states in all_states:
                out = self.gather_indexes(states, item_seq_len - 1)
                all_outputs.append(out)

            all_outputs = torch.stack(all_outputs, dim=1)  # [B, T, H]
            all_outputs = F.normalize(all_outputs, dim=-1)
            item_embedding = F.normalize(self.item_embedding.weight, dim=-1)
            per_step_logits = torch.matmul(all_outputs, item_embedding.transpose(0, 1))  # [B, T, N]
            per_step_logps = torch.log_softmax(per_step_logits, dim=-1)  # [B, T, N]

        return last_output, per_step_logps, all_outputs
