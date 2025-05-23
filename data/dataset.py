from recbole.data.dataset import SequentialDataset
import numpy as np
import pickle
import os
import os.path as osp
import torch.nn as nn
import json
import torch


class LARESDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def semantic_augmentation(self, dataset):
        aug_path = self.config['data_path'] + '/semantic_augmentation.pkl'
        if os.path.exists(aug_path):
            same_target_index = pickle.load(open(aug_path, 'rb'))
            return same_target_index

        same_target_index = []
        target_item = dataset.inter_feat[dataset.iid_field]
        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]  # all index of a specific item id with self item
            delete_index = np.argwhere(all_index_same_id == index)
            all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
            same_target_index.append(all_index_same_id_wo_self.tolist())
        pickle.dump(same_target_index, open(aug_path, 'wb'))

        return same_target_index
    
    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config-old.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        assert self.benchmark_filename_list is not None

        self._drop_unused_col()
        cumsum = list(np.cumsum(self.file_size_list))
        datasets = [
            self.copy(self.inter_feat[start:end])
            for start, end in zip([0] + cumsum[:-1], cumsum)
        ]
        train_dataset = datasets[0]
        same_target_index = self.semantic_augmentation(train_dataset)
        setattr(train_dataset, 'same_target_index', same_target_index)
        datasets[0] = train_dataset

        return datasets


class TedRecDataset(LARESDataset):
    def __init__(self, config):
        super().__init__(config)
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        # loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
        loaded_feat = np.load(feat_path).reshape(-1, self.plm_size).astype(np.float32)
        token2id = json.load(open(osp.join(self.config['data_path'], f'{self.dataset_name}.emb_map.json'), 'r'))

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token2id[token])-1]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding
    
