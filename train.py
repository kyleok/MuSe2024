import os
import numpy as np
import torch
import torch.optim as optim
from eval import evaluate, get_predictions
import wandb
import pandas as pd
import dill
import copy


def train(model, train_loader, optimizer, loss_fn, use_gpu=False):
    train_loss_list = []

    model.train()
    if use_gpu:
        model.cuda()

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        batch_size = features.size(0)

        if use_gpu:
            features = features.cuda()
            feature_lens = feature_lens.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        preds, _ = model(features, feature_lens)
        # pdb.set_trace()

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    train_loss = np.mean(train_loss_list)
    return train_loss


def save_model(model, model_folder, id):
    model_file_name = f'model_{id}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    torch.save(model, model_file, pickle_module=dill)
    return model_file


def write_predictions(task, full_metas, full_preds, full_labels, prediction_path, filename):
    assert prediction_path != ''

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    metas_flat = [item for sublist in full_metas for item in sublist]
    preds_flat = [item for sublist in full_preds for item in sublist]
    labels_flat = [item for sublist in full_labels for item in sublist]

    num_meta_cols = len(metas_flat[0]) if isinstance(metas_flat[0], list) else metas_flat[0].shape[0]
    prediction_df = pd.DataFrame(columns=[f'meta_col_{i}' for i in range(num_meta_cols)])
    for i in range(num_meta_cols):
        prediction_df[f'meta_col_{i}'] = [m[i] for m in metas_flat]
    prediction_df['prediction'] = preds_flat
    prediction_df['label'] = labels_flat
    prediction_df.to_csv(os.path.join(prediction_path, filename), index=False)
    return prediction_df


def train_model(task, model, data_loader, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, regularization=0.0):
    print(
        f"settings are: {task=}, {epochs=}, {lr=}, {model_path=}, {identifier=}, {use_gpu=}, {loss_fn=}, {eval_fn=}, {eval_metric_str=}, {early_stopping_patience=}, {regularization=}")
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    best_model_state = None
    early_stop = 0

    for epoch in range(1, epochs + 1):
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu)
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu)

        if np.isnan(val_score):
            val_score = -0.99
            print(f'Warning: val score is nan, replacing it with -0.99')

        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            # best_model_state = model.state_dict().copy() this was bad move
            best_model_state = copy.deepcopy(model.state_dict())
            try:
                best_model_file = save_model(model, model_path, identifier)
                print(f'ID/Seed {identifier} | Model saved at {best_model_file}')
            except Exception as e:
                print(f'Warning: Model could not be saved due to {e}')
        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 
                   'val_loss': val_loss, 'val_score': val_score,
                   'best_val_score': best_val_score, 'best_val_loss': best_val_loss})
                   
    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Generate predictions for train, dev, and test sets
    for split, loader in [('train', train_loader), ('dev', val_loader), ('test', test_loader)]:
        full_labels, full_preds, full_metas = get_predictions(model, task, loader, use_gpu=use_gpu)
        csv_filename = f'{identifier}_{split}_predictions.csv'
        csv_path = os.path.join(model_path, csv_filename)
        prediction_df = write_predictions(task, full_metas, full_preds, full_labels, model_path, csv_filename)

        # Log the CSV file to wandb as an artifact
        artifact = wandb.Artifact(f"{split}_predictions", type="predictions")
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)

        # Wait for the artifact to be uploaded
        print(f"Waiting for {csv_filename} to sync with wandb...")
        artifact.wait()

        # Remove the local CSV file
        try:
            os.remove(csv_path)
            print(f"Local file {csv_filename} removed successfully.")
        except Exception as e:
            print(f"Error removing local file {csv_filename}: {e}")

    return best_val_loss, best_val_score, best_model_file
