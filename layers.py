import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from recbole.model.layers import FeedForward, MultiHeadAttention
import math


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FilterLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps, max_his_len):
        super(FilterLayer, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_length = max_his_len

        self.complex_weight = nn.Parameter(torch.randn(1, self.max_seq_length//2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=self.layer_norm_eps )


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps, hidden_act, max_his_len):
        super(Intermediate, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.max_seq_length = max_his_len

        self.dense_1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.intermediate_act_fn = gelu

        self.dense_2 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FMLPLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, hidden_act, layer_norm_eps, max_his_len):
        super(FMLPLayer, self).__init__()
        self.filterlayer = FilterLayer(hidden_size, hidden_dropout_prob, layer_norm_eps, max_his_len)
        self.intermediate = Intermediate(hidden_size, hidden_dropout_prob, layer_norm_eps, hidden_act, max_his_len)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.filterlayer(hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class FMLPEncoder(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, hidden_act, layer_norm_eps, max_his_len, n_layers):
        super(FMLPEncoder, self).__init__()
        layer = FMLPLayer(hidden_size, hidden_dropout_prob, hidden_act, layer_norm_eps, max_his_len)
        self.n_layers = n_layers
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(self.n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BSARecEncoder(nn.Module):
    def __init__(self, hidden_size, inner_size, n_heads, hidden_act, hidden_dropout_prob,
                 attn_dropout_prob, layer_norm_eps, n_layers, alpha, c):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(hidden_size=hidden_size, inner_size=inner_size,
                            n_heads=n_heads, hidden_act=hidden_act,
                            hidden_dropout_prob=hidden_dropout_prob, attn_dropout_prob=attn_dropout_prob, layer_norm_eps=layer_norm_eps, alpha=alpha, c=c)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, hidden_size, inner_size, n_heads, hidden_act, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, alpha, c):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(hidden_size, n_heads, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, alpha, c)
        self.feed_forward = FeedForward(hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, alpha, c):
        super(BSARecLayer, self).__init__()
        self.filter_layer = FrequencyLayer(hidden_size, hidden_dropout_prob, c)
        self.attention_layer = MultiHeadAttention(n_heads,
                                                  hidden_size,
                                                  hidden_dropout_prob,
                                                  attn_dropout_prob,
                                                  layer_norm_eps,)
        self.alpha = alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class FrequencyLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class DTRLayer(nn.Module):
    """Distinguishable Textual Representations Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0, max_seq_length=50):
        super(DTRLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(1, max_seq_length, input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, max_seq_length=50, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([DTRLayer(layers[0], layers[1], dropout, max_seq_length) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)
