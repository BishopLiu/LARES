import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, set_color
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader
from model import LARES, FMLPRec_plus, BSARec_plus, TedRec_plus
from utils import create_dataset
from trainer import PTTrainer
from data.dataloader import PTTrainDataLoader
from data.dataset import LARESDataset, TedRecDataset
from utils import init_logger


def main(d, m, **kwargs):
    # configurations initialization
    props = [f'props/{m}.yaml', 'props/sl_train.yaml']
    print(props)

    # configurations initialization
    config = Config(model=LARES, dataset=d, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    if m == 'TedRec':
        dataset = TedRecDataset(config)
    else:
        dataset = LARESDataset(config)
    logger.info(dataset)

    # dataset splitting
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets

    train_data = PTTrainDataLoader(config, train_dataset, None, shuffle=True)
    valid_data = FullSortEvalDataLoader(config, valid_dataset, None, shuffle=False)
    test_data = FullSortEvalDataLoader(config, test_dataset, None, shuffle=False)

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

    logger.info(model)

    # trainer loading and initialization
    trainer = PTTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_results = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'],
                                    test_recurrence_ratios = config['test_recurrence_ratios'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    for recurrence, test_result in test_results.items():
        logger.info(set_color(f'recurrence_{recurrence} test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Instruments', help='dataset name')
    parser.add_argument('-m', type=str, default='LARES', help='model name')
    args, unparsed = parser.parse_known_args()
    print(args)

    main(args.d, args.m)
