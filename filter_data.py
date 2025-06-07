import argparse
import torch
import os
import json
from recbole.utils import init_seed, set_color
from model import LARES, FMLPRec_plus, BSARec_plus, TedRec_plus
from data.dataset import LARESDataset, TedRecDataset
from tqdm import tqdm
from recbole.data.dataloader import TrainDataLoader


def main(ckpt_path, device, n_iter):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt['config-old']
    config['device'] = device
    m = config['model']
    d = config['dataset']

    init_seed(config['seed'], config['reproducibility'])

    # dataset filtering
    if m == 'TedRec':
        dataset = TedRecDataset(config)
    else:
        dataset = LARESDataset(config)

    # dataset splitting
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets

    train_data = TrainDataLoader(config, train_dataset, None, shuffle=False)

    # model loading and initialization
    if m == 'LARES':
        model = LARES(config, train_data.dataset).to(config['device'])
    elif m == 'FMLPRec':
        model = FMLPRec_plus(config, train_data.dataset).to(config['device'])
    elif m == 'BSARec':
        model = BSARec_plus(config, train_data.dataset).to(config['device'])
    elif m == 'TedRec':
        model = TedRec_plus(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError(f'Unknown model name: {m}')

    pretrain_model_weight = ckpt['state_dict']
    model.load_state_dict(pretrain_model_weight)

    model.eval()
    selected_idx = []

    for _ in range(n_iter):
        num_sample = 0
        iter_data = (
                tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Evaluate   ", "pink"),
                )
            )
        for batch_idx, batched_data in enumerate(iter_data):
            batched_data = batched_data.to(model.device)
            scores = model.full_sort_predict(batched_data)
            labels = batched_data[model.POS_ITEM_ID]
            topk_idx = torch.topk(scores, 100, dim=-1).indices
            mask = (topk_idx == labels.unsqueeze(1))  # [B, K]
            in_top_k = mask.sum(dim=1) > 0
            idx = (torch.argwhere(in_top_k) + num_sample).detach().cpu().flatten().tolist()
            selected_idx.extend(idx)
            num_sample += len(batched_data)
    selected_idx = list(set(selected_idx))
    selected_idx.sort()

    with open(os.path.join(config['data_path'], f'{d}.{m}.select_idx.json'), 'w') as f:
        json.dump(selected_idx, f)

    print(f"Selected {len(selected_idx)} items after {n_iter} iterations of filtering.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on.')
    parser.add_argument('--n_iter', type=int, default=3, help='Number of iterations for filtering.')

    args = parser.parse_args()

    main(args.ckpt_path, args.device, args.n_iter)
