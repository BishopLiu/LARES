from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction
import numpy as np


class PTTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.same_target_index = dataset.same_target_index
        self.static_item_id_list = dataset.inter_feat[dataset.item_id_list_field]
        self.static_item_length = dataset.inter_feat[dataset.item_list_length_field]
        self.iid_field = dataset.iid_field
        self.item_id_list_field = dataset.item_id_list_field
        self.item_list_length_field = dataset.item_list_length_field
    
    def _aug_(self, cur_data, index):
        null_index = []
        sample_pos = []
        for i, idx in enumerate(index):
            targets = self.same_target_index[idx]
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                sample_pos.append(np.random.choice(targets))
        sem_pos_seqs = self.static_item_id_list[sample_pos]
        sem_pos_lengths = self.static_item_length[sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = cur_data[self.item_id_list_field][null_index]
            sem_pos_lengths[null_index] = cur_data[self.item_list_length_field][null_index]
        
        cur_data.update(Interaction({'sem_aug': sem_pos_seqs, 'sem_aug_length': sem_pos_lengths}))
        
        return cur_data
    
    def collate_fn(self, index):
        data = self._dataset[np.array(index)]
        data = self._aug_(data, index)
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)
    
