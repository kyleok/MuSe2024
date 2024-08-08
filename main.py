import argparse
import os
import random
import sys
from datetime import datetime

import numpy
import torch
import wandb
from dateutil import tz
from torch import nn

import config
from config import TASKS, PERCEPTION, HUMOR
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from eval import evaluate, calc_auc, calc_pearsons
# from model import Model, xLSTM, TFModel
from models.rnn_model import Model
from models.transformer_model import TFModel
from models.xlstm_model import xLSTMModel as xLSTM
from train import train_model
from utils import Logger, seed_worker, log_results


def parse_args():
    def int_list(arg):
        return [int(x) for x in arg.split(',')]

    parser = argparse.ArgumentParser(description='MuSe 2024.')

    parser.add_argument('--task', type=str, required=True, choices=TASKS,
                        help=f'Specify the task from {TASKS}.')
    parser.add_argument('--feature', required=True,
                        help='Specify the features used (only one).')
    parser.add_argument('--label_dim', default="assertiv", choices=config.PERCEPTION_LABELS)
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--use_gpu', action='store_false',
                        help='Specify whether to use CPU for training (default: use GPU).')  # changed from store_true, always use gpu
    parser.add_argument('--cache', action='store_false',
                        help='Specify whether to cache data as pickle file (default: True).')  # changed from store_true, always cache
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    # evaluation only arguments
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--eval_seed', type=str, default=None,
                        help='Specify seed to be evaluated; only considered when --eval_model is given.')
    parser.add_argument('--model_type', type=str, default='RNN', choices=['RNN', 'XLSTM', 'TF'],
                        help='Specify the model to use.')
    # parser.add_argument('--lstm_ratio', type=str, default=None, help='Ratio of mLSTM to sLSTM blocks (e.g., "7:1")')
    parser.add_argument('--num_blocks', type=int, default=1, help='Number of xLSTM blocks')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension for xLSTM')
    parser.add_argument('--context_length', type=int, default=256, help='context length for xLSTM')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for xLSTM,both m and s')
    parser.add_argument('--kernel_size', type=int, default=4, help='conv1d kernel size for xLSTM,both m and s')
    parser.add_argument('--proj_factor', type=float, default=1.3, help='Number of heads for xLSTM,both m and s')
    parser.add_argument('--slstm_at', type=int_list, default=[1],
                        help='Comma-separated list of integers specifying SLSTM block positions')

    args = parser.parse_args()
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    if args.eval_model:
        assert args.eval_seed
    return args


def get_loss_fn(task):
    if task == HUMOR:
        return nn.BCELoss(), 'Binary Crossentropy'
    elif task == PERCEPTION:
        return nn.MSELoss(reduction='mean'), 'MSE'


def get_eval_fn(task):
    if task == PERCEPTION:
        return calc_pearsons, 'Pearson'
    elif task == HUMOR:
        return calc_auc, 'AUC'


def main(args):
    # ensure reproducibility
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    wandb.init(project='MuSe2024_final_fixed', entity='feelsgood_muse',config={
            "task": args.task,
            "feature": args.feature,
            "label_dim": args.label_dim,
            "normalize": args.normalize,
            "model_dim": args.model_dim,
            "rnn_n_layers": args.rnn_n_layers,
            "rnn_bi": args.rnn_bi,
            "d_fc_out": args.d_fc_out,
            "rnn_dropout": args.rnn_dropout,
            "linear_dropout": args.linear_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "early_stopping_patience": args.early_stopping_patience,
            "regularization": args.regularization,
            "model_type": args.model_type,
            "num_blocks": args.num_blocks,
            "embedding_dim": args.embedding_dim,
            "context_length": args.context_length,
            "num_heads": args.num_heads,
            "kernel_size": args.kernel_size,
            "proj_factor": args.proj_factor,
            "slstm_at": args.slstm_at,
        })
    wandb.run.name = args.log_file_name + '_' + str(args.seed)

    # emo_dim only relevant for stress/personalisation
    args.label_dim = args.label_dim if args.task == PERCEPTION else ''
    print('Loading data ...')
    args.paths['partition'] = os.path.join(config.PATH_TO_METADATA[args.task], f'partition.csv')
    args.normalize = True if args.feature == "egemaps" else False
    if args.normalize:
        print('Normalizing features...')

    data = load_data(args.task, args.paths, args.feature, args.label_dim, args.normalize, save=args.cache)
    datasets = {partition: MuSeDataset(data, partition) for partition in data.keys()}

    args.d_in = datasets['train'].get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    if args.eval_model is None:  # Train and validate on single seed
        seed = args.seed

        torch.manual_seed(seed)
        data_loader = {}
        for partition, dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=4,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=custom_collate_fn)

        model_dict = {"RNN": Model, "XLSTM": xLSTM, "TF": TFModel}
        model = model_dict[args.model_type](args)

        print('=' * 50)
        print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')

        val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                           args.lr, args.paths['model'], seed, use_gpu=args.use_gpu,
                                                           loss_fn=loss_fn, eval_fn=eval_fn,
                                                           eval_metric_str=eval_str,
                                                           regularization=args.regularization,
                                                           early_stopping_patience=args.early_stopping_patience)
        remove = True if val_score < 0.35 else False
        print(f"model will be removed: {remove}")
        # restore best model encountered during training
        if args.model_type == 'XLSTM':
            pass
        else:
            model = torch.load(best_model_file)

            if not args.predict:  # run evaluation only if test labels are available
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                 eval_fn=eval_fn, use_gpu=args.use_gpu)
                print(f'[Test {eval_str}]:  {test_score:7.4f}')
            if remove:
                os.remove(best_model_file)


    else:  # Evaluate existing model (No training)
        model_file = os.path.join(args.paths['model'], f'model_{args.eval_seed}.pth')
        model = torch.load(model_file, map_location=torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
        data_loader = {}
        for partition, dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=4,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=custom_collate_fn)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                                  use_gpu=args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     use_gpu=args.use_gpu)
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting devel and test samples...')
        best_model = torch.load(model_file, map_location=config.device)
        evaluate(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_devel.csv')
        evaluate(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_test.csv')
        print(f'Find predictions in {os.path.join(args.paths["predict"])}')

    print('Done.')
    wandb.finish()


if __name__ == '__main__':
    print("Start", flush=True)
    wandb.require('core')
    args = parse_args()

    args.log_file_name = '{}_{}_[{}]_[{}_{}_{}_{}]_[{}_{}]'.format(args.model_type,
                                                                   datetime.now(tz=tz.gettz()).strftime(
                                                                       "%Y-%m-%d-%H-%M"),
                                                                   args.feature.replace(os.path.sep, "-"),
                                                                   args.model_dim, args.rnn_n_layers, args.rnn_bi,
                                                                   args.d_fc_out, args.lr, args.batch_size)

    # adjust your paths in config.py
    task_id = args.task if args.task != PERCEPTION else os.path.join(args.task, args.label_dim)
    args.paths = {
        'log': os.path.join(config.LOG_FOLDER, task_id) if not args.predict else os.path.join(config.LOG_FOLDER,
                                                                                              task_id, 'prediction'),
        'data': os.path.join(config.DATA_FOLDER, task_id),
        'model': os.path.join(config.MODEL_FOLDER, task_id,
                              args.log_file_name if not args.eval_model else args.eval_model)}
    if args.predict:
        if args.eval_model:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, task_id, args.eval_model, args.eval_seed)
        else:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, task_id, args.log_file_name)

    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))

    main(args)

    # os.system(f"rm -r {config.OUTPUT_PATH}")
