import numpy as np 
import torch


def get_feature_predictions(model, task, data_loader, use_gpu=False):
    full_preds = []
    full_labels = []
    full_metas = []
    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, labels, metas = batch_data

            if use_gpu:
                model.cuda()
                features = features.cuda()
                labels = labels.cuda()

            preds, _ = model(features)

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist())
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist())
            full_metas.append(metas.tolist())

    return full_labels, full_preds, full_metas

def evaluate_feature_fusion(task, model, data_loader, loss_fn, eval_fn, use_gpu=False):
    losses, sizes = 0, 0
    full_preds = []
    full_labels = []

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, labels, metas = batch_data
            if torch.any(torch.isnan(labels)):
                print('No labels available, no evaluation')
                return np.nan, np.nan

            batch_size = features.size(0)

            if use_gpu:
                model.cuda()
                features = features.cuda()
                labels = labels.cuda()

            preds,_ = model(features)

            loss = loss_fn(torch.flatten(preds), torch.flatten(labels))

            losses += loss.item() * batch_size
            sizes += batch_size

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist())
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist())

        score = eval_fn(full_preds, full_labels)
        total_loss = losses / sizes
        return total_loss, score