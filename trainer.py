import os
from time import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.trainer import Trainer
from recbole.utils import early_stopping, dict2str, set_color, get_gpu_usage, EvaluatorType
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
import torch.amp as amp
import copy
from recbole.utils import get_local_time


def interleave_and_pad(tensors_list):
    
    G = len(tensors_list)
    B = tensors_list[0].shape[1]
    # 将所有张量的第二维填充到最大长度
    padded_tensors = pad_sequence(tensors_list).permute(2, 1, 0, 3) # [T, G, B, N]->[B, G, T, N]
    T, N = padded_tensors.shape[2], padded_tensors.shape[3]

    # 按照第一维进行交错拼接
    interleaved_tensor = padded_tensors.reshape(-1, T, N)
    
    return interleaved_tensor


class PTTrainer(Trainer):
    def __init__(self, config, model):
        super(PTTrainer, self).__init__(config, model)
        saved_model_file = "{}-{}-{}.pth".format(self.config["model"], self.config['train_stage'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

    @torch.no_grad()
    def evaluate(
            self, eval_data, load_best_model=True, model_file=None, show_progress=False,
            test_recurrence_ratios=None
    ):
        r"""Evaluate the model based on the eval data.

		Args:
			eval_data (DataLoader): the eval data
			load_best_model (bool, optional): whether load the best model in the training process, default: True.
											  It should be set True, if users want to test the model after training.
			model_file (str, optional): the saved model file, default: None. If users want to test the previously
										trained model file, they can set this parameter.
			show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

		Returns:
			collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
		"""
        if test_recurrence_ratios is None:
            return super().evaluate(eval_data, load_best_model, model_file, show_progress)

        recurrence_list = [1]
        for recurrence_ratio in test_recurrence_ratios:
            r = max(1, int(self.config["mean_recurrence"] * recurrence_ratio) )
            recurrence_list.append(r)

        recurrence_list = sorted(list(set(recurrence_list)))
        results = {}
        for recurrence in recurrence_list:
            self.model.test_recurrence = recurrence
            # self.logger.info(f"Testing with recurrence {recurrence}")
            results[recurrence] = super().evaluate(eval_data, load_best_model, model_file, show_progress)

        return results


class RLTrainer(PTTrainer):
    def __init__(self, config, model):
        super(RLTrainer, self).__init__(config, model)
        self.k = config['k']
        self.beta = config['beta']
        self.group_num = config['group_num']
        self.ref_model = copy.deepcopy(model)
        self.progress_step = config['progress_step']
        self.reward_metric = config['reward_metric']

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        self.model.train()
        total_loss = None
        total_reward = 0
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        scaler = amp.GradScaler("cuda", enabled=self.enable_scaler)
        n_samples = 0

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            
            losses, reward = self.compute_loss(interaction, epoch_idx)
            total_reward += reward
            n_samples += interaction[self.model.ITEM_SEQ].shape[0]
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)

            scaler.scale(loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
        self.logger.info("epoch %d training [rewards: %.2f]" % (epoch_idx, total_reward / (n_samples*self.group_num)))

        torch.cuda.empty_cache()

        return total_loss
    
    def compute_loss(self, interaction, epoch_idx):
        # Compute the per-token log probabilities for the model
        self.ref_model.eval()

        item_seq = interaction[self.model.ITEM_SEQ]
        item_seq_len = interaction[self.model.ITEM_SEQ_LEN]
        pos_items = interaction[self.model.POS_ITEM_ID]
        pos_items = pos_items.repeat_interleave(self.group_num)

        per_step_logps = []
        ref_per_step_logps = []
        scores = []
        for _ in range(self.group_num):
            final_output, step_logps, _ = self.model(item_seq, item_seq_len, return_all_states=True)
            per_step_logps.append(step_logps.transpose(0, 1))
            scores.append(torch.matmul(final_output, self.model.item_embedding.weight.transpose(0, 1)))

            num_steps = step_logps.shape[1]
            _, ref_step_logps, _ = self.ref_model(item_seq, item_seq_len, return_all_states=True, num_steps=num_steps)
            ref_per_step_logps.append(ref_step_logps.transpose(0, 1))
        
        scores = torch.stack(scores, dim=1).flatten(0, 1)

        per_step_logps = interleave_and_pad(per_step_logps)
        ref_per_step_logps = interleave_and_pad(ref_per_step_logps)
        action_mask = (per_step_logps[:,:,0]!=0).long()  # [B*G, T]
        BG, T = action_mask.shape[0], action_mask.shape[1]

        action_len = action_mask.sum(dim=1)
        last_step_logps = self.model.gather_indexes(per_step_logps, action_len-1)

        advantages, rewards = self.compute_advantages(last_step_logps, pos_items, self.k)
        
        # per_step_logps = per_step_logps.max(dim=-1).values
        x_index = torch.arange(BG).repeat_interleave(T)
        y_index = torch.arange(T).repeat(BG)
        z_index = pos_items.repeat_interleave(T)

        per_step_logps = per_step_logps[x_index, y_index, z_index].view(-1, T)
        ref_per_step_logps = ref_per_step_logps[x_index, y_index, z_index].view(-1, T)

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_step_logps - per_step_logps) - (ref_per_step_logps - per_step_logps) - 1
        # per_token_kl = 0
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_step_logps - per_step_logps.detach()) * advantages.unsqueeze(1)

        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * action_mask).sum(dim=1) / action_len).mean()

        return loss, rewards
    
    def compute_advantages(self, last_step_logps, pos_items, k):
        batch_size = last_step_logps.size(0)
        k = min(k, last_step_logps.size(-1))  # 防止k超过类别总数
        topk_idx = torch.topk(last_step_logps, k, dim=-1).indices

        if self.reward_metric.lower() == 'recall':
            rewards = torch.any(topk_idx == pos_items.view(batch_size, 1), dim=1).float()
        elif self.reward_metric.lower() == 'ndcg':
            mask = (topk_idx == pos_items.unsqueeze(1))  # [B, K]
            ranks = mask.int().argmax(dim=1)  # [B]
            in_top_k = mask.sum(dim=1) > 0

            rewards = torch.where(
                in_top_k,
                1.0 / torch.log2(ranks.float() + 2.0),
                torch.tensor(0.0, device=self.device)
            )
        else:
            raise ValueError(f"Unsupported reward metric: {self.reward_metric}")

        mean_grouped_rewards = rewards.view(-1, self.group_num).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.group_num).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.group_num, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.group_num, dim=0)
        
        advantages = (rewards-mean_grouped_rewards) / (std_grouped_rewards+1e-4)

        return advantages, rewards.sum()


