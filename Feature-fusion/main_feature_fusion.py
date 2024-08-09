import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import random
from datetime import datetime
import pathlib

import numpy
import torch
import wandb
from dateutil import tz
import pandas as pd

import config
from config import TASKS, PERCEPTION, HUMOR
from main import get_loss_fn, get_eval_fn
from data_parser import load_data
from dataset import MuSeDataset, custom_collate_fn
from utils import Logger, seed_worker


from models.feature_fusion_model import FeatureFusionModel
from data_parser_feature_fusion import load_feature_fusion_data
from dataset_feature_fusion import MuSeFeatureDataset, custom_feature_collate_fn
from train_feature_fusion import train_feature_fusion_model
from eval_feature_fusion import evaluate_feature_fusion


def parse_args():

    parser = argparse.ArgumentParser(description='MuSe 2024.')

    parser.add_argument('--task', type=str, required=True, choices=TASKS,
                        help=f'Specify the task from {TASKS}.')
    parser.add_argument('--pt_vision_model_name', type=str, default=None,
                        help='Specify the vision features used (only one).')
    parser.add_argument('--pt_audio_model_name', type=str, default=None,
                        help='Specify the audio features used (only one).')
    parser.add_argument('--pt_text_model_name', type=str, default=None,
                        help='Specify the text features used (only one).')
    parser.add_argument('--label_dim', default="assertiv", choices=config.PERCEPTION_LABELS)
    parser.add_argument('--model_dim', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')

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
                        help='Specify model which is to be evaluated; no training with this  (default: False).')
    parser.add_argument('--eval_seed', type=str, default=None,
                        help='Specify seed to be evaluated; only considered when --eval_model is given.')
    parser.add_argument('--fusion_type', type=str, default='feature_fusion',
                        help='')
    parser.add_argument('--feature_level', type=str, default='encoder',
                        help='')
    
    args = parser.parse_args()
    if not (args.result_csv is None) and args.predict:
        print("--result_csv is not compatible with --predict")
        sys.exit(-1)
    if args.eval_model:
        assert args.eval_seed
    return args

def get_model_info(model_path):
    info = dict()

    model_path = model_path.lstrip()
    if model_path.startswith("MultiAtt_"):
        info['sim_att'] = True
        info['encoder'] = model_path.split('_')[1]
    else:
        info['sim_att'] = False
        info['encoder'] = model_path.split('_')[0]

    sub_parts = [p.rstrip(']_') for p in model_path.split('[')[1:]]

    info['feature'] = sub_parts[0]
    parameters = sub_parts[1].split('_')
    info['model_dim'] = int(parameters[0])
    info['encoder_n_layers'] = int(parameters[1])
    info['args_rnn_bi'] = bool(parameters[2])
    info['d_fc_out'] = int(parameters[3])

    info['batch_size'] = int(sub_parts[2].split('_')[1].rstrip(']'))

    return info


def save_extracted_feature(model, model_device, data, feature_path):
    datasets = {partition:MuSeDataset(data, partition) for partition in data.keys()}

    data_loader = {}
    df = None
    for partition, dataset in datasets.items():  # one DataLoader for each partition
        data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                                                num_workers=4,
                                                                worker_init_fn=seed_worker,
                                                                collate_fn=custom_collate_fn)
        
        # print(f'Extracting feature from {model_file}:')
        features, full_metas = extract_feature('perception', model, data_loader[partition], device=model_device)

        flat_subj_ids = numpy.array([arr[0] for arr in full_metas[0]]).reshape(-1, 1)
        flat_features = features.reshape(features.shape[0], -1).to('cpu')

        df_id = pd.DataFrame(flat_subj_ids, columns=['subj_id'])
        df_feature = pd.DataFrame(flat_features, columns=[f'feature_{id}' for id in range(1, flat_features.shape[1]+1)])
        df_new = pd.concat([df_id, df_feature], axis=1)

        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new])

    df = df.sort_values(by='subj_id', key=lambda col: col.astype(int))

    # make sure the directory exists
    csv_dir = pathlib.Path(feature_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    # write back
    if os.path.exists(csv_dir):
        df.to_csv(feature_path, index=False)
        
def extract_feature(task, model, data_loader, predict=True, prediction_path=None,
            filename=None, device='cpu'):
    losses, sizes = 0, 0
    full_preds = []
    full_labels = []
    if predict:
        full_metas = []
    else:
        full_metas = None
        
    hooked_features = []

    def hook_input_fn(module, input, output):
        hooked_features.append(input[0])
    def hook_output_fn(module, input, output):
        hooked_features.append(output)

    layer_dict = dict(model.named_modules())

    if 'out' in layer_dict.keys():
        layer_dict['out'].register_forward_hook(hook_input_fn)

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data

            model = model.to(device)
            features = features.to(device)
            feature_lens = feature_lens.to(device)
            labels = labels.to(device)

            _, _= model(features, feature_lens)
            
            full_metas.append(metas.tolist())
            
        output = hooked_features[0]

        if len(output.shape) == 3:
            output = output[:, -1, :]
        return output, full_metas


def main(args):
    # ensure reproducibility
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    wandb.init(project='MuSe2024_final_fixed', entity='feelsgood_muse',config={
            "task": args.task,
            "label_dim": args.label_dim,
            "model_dim": args.model_dim,
            "d_fc_out": args.d_fc_out,
            "linear_dropout": args.linear_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "early_stopping_patience": args.early_stopping_patience,
            "regularization": args.regularization,
        })
    wandb.run.name = args.log_file_name + '_' + str(args.seed)

    # emo_dim only relevant for stress/personalisation
    args.label_dim = args.label_dim if args.task==PERCEPTION else ''
    print('Loading data ...')
    args.paths['partition'] = os.path.join(config.PATH_TO_METADATA[args.task], f'partition.csv')
    args.normalize = None

    for pt_model_name in args.pt_model_names.values():
        pt_model_info = get_model_info(pt_model_name)


        pt_feature_path = os.path.join(config.DATA_FOLDER, task_id, f'{pt_model_name}.csv')
        if not os.path.exists(pt_feature_path):
            pt_model_dir = os.path.join(config.BEST_MODEL_FOLDER, task_id, pt_model_name)
            pt_model_name = os.listdir(pt_model_dir)[0]
            pt_model_path = os.path.join(pt_model_dir, pt_model_name)
            print(f"Target pretrained model: {pt_model_path}")
            model = torch.load(pt_model_path, map_location=config.device)
            model.n_to_1 = True                

            data = load_data(args.task, args.paths, pt_model_info['feature'], args.label_dim)
            
            save_extracted_feature(model, config.device, data, pt_feature_path)

            print("-"*50)



    data = load_feature_fusion_data(args.task, args.paths, args.pt_model_names, args.label_dim, args.normalize, save=args.cache)
    datasets = {partition:MuSeFeatureDataset(data, partition) for partition in data.keys()}

    args.d_in = datasets['train'].get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = get_loss_fn(args.task)
    eval_fn, eval_str = get_eval_fn(args.task)

    if args.eval_model is None:  # Train and validate for each seed
        seed = args.seed

        torch.manual_seed(seed)
        data_loader = {}
        for partition,dataset in datasets.items():  # one DataLoader for each partition
            batch_size = args.batch_size if partition == 'train' else 2 * args.batch_size
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                                    num_workers=4,
                                                                    worker_init_fn=seed_worker,
                                                                    collate_fn=custom_feature_collate_fn)
        
        
        args.model_dim = dataset.get_feature_dim()

        model = FeatureFusionModel(args)

        print('=' * 50)
        print(f'Training model... [seed {seed}] for at most {args.epochs} epochs')

        val_loss, val_score, best_model_file = train_feature_fusion_model(args.task, model, data_loader, args.epochs,
                                                            args.lr, args.paths['model'], seed, use_gpu=args.use_gpu,
                                                            loss_fn=loss_fn, eval_fn=eval_fn,
                                                            eval_metric_str=eval_str,
                                                            regularization=args.regularization,
                                                            early_stopping_patience=args.early_stopping_patience)
        # restore best model encountered during training
        model = torch.load(best_model_file)

        if not args.predict:  # run evaluation only if test labels are available
            test_loss, test_score = evaluate_feature_fusion(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                eval_fn=eval_fn, use_gpu=args.use_gpu)
            print(f'[Test {eval_str}]:  {test_score:7.4f}')


        model_file = best_model_file

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
                                                                 collate_fn=custom_feature_collate_fn)
        _, valid_score = evaluate_feature_fusion(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                                  use_gpu=args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate_feature_fusion(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     use_gpu=args.use_gpu)
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting devel and test samples...')
        best_model = torch.load(model_file, map_location=config.device)
        evaluate_feature_fusion(args.task, best_model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'],
                 filename='predictions_devel.csv')
        evaluate_feature_fusion(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'], 
                 filename='predictions_test.csv')
        print(f'Find predictions in {os.path.join(args.paths["predict"])}')

    print('Done.')
    wandb.finish()


if __name__ == '__main__':
    print("Start", flush=True)
    wandb.require('core')
    args = parse_args()

    args.pt_model_names = {}

    args.pt_vision_model_name is not None and args.pt_model_names.update({'vision': args.pt_vision_model_name})
    args.pt_audio_model_name is not None and args.pt_model_names.update({'audio': args.pt_audio_model_name})
    args.pt_text_model_name is not None and args.pt_model_names.update({'text': args.pt_text_model_name})


    feature_names = []

    for pt_model_name in args.pt_model_names.values():
        pt_model_info = get_model_info(pt_model_name)

        feature_names.append('multiatt-' if pt_model_info['sim_att'] else '' + '{}-{}'.format(pt_model_info['encoder'], pt_model_info['feature']))

    
    args.log_file_name =  'feature-fusion_{}_[{}]_[{}]_[{}_{}]'.format(datetime.now(tz=tz.gettz()).strftime(
                                                                                    "%Y-%m-%d-%H-%M"), '_'.join(feature_names),
                                                                                args.d_fc_out, args.lr,args.batch_size)


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
