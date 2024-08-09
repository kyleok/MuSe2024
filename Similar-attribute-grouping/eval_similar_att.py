import numpy as np 
import torch


def get_sim_att_predictions(model, task, data_loader, label_dim, use_gpu=False):
    full_preds = []
    full_labels = []
    full_metas = []
    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data

            if use_gpu:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                labels = labels[label_dim].cuda()

            preds, _ = model(features, feature_lens)
            preds = preds[label_dim]
            metas = metas[label_dim]

            full_labels.append(labels.cpu().detach().squeeze().numpy().tolist())
            full_preds.append(preds.cpu().detach().squeeze().numpy().tolist())
            full_metas.append(metas.tolist())

    return full_labels, full_preds, full_metas

def evaluate_sim_att(task, model, data_loader, label_dims, loss_fn, eval_fn, use_gpu=False):
    losses, sizes = 0, 0
    full_labels = {label_dim: [] for label_dim in label_dims}
    full_preds = {label_dim: [] for label_dim in label_dims}

    model.eval()
    with torch.no_grad():
        for batch, batch_data in enumerate(data_loader, 1):
            features, feature_lens, labels, metas = batch_data
            for label_dim in label_dims:
                if torch.any(torch.isnan(labels[label_dim])):
                    print('No labels available, no evaluation')
                    return np.nan, np.nan

            batch_size = features.size(0)

            if use_gpu:
                model.cuda()
                features = features.cuda()
                feature_lens = feature_lens.cuda()
                for label_dim in label_dims:
                    labels[label_dim] = labels[label_dim].cuda()

            preds, _ = model(features, feature_lens)

            loss = 0
            for label_dim in label_dims:
                loss += loss_fn(preds[label_dim].squeeze(-1), labels[label_dim].squeeze(-1))
            loss /= len(label_dims)

            losses += loss.item() * batch_size
            sizes += batch_size

            for label_dim in label_dims:
                full_labels[label_dim].append(labels[label_dim].cpu().detach().squeeze().numpy().tolist())
                full_preds[label_dim].append(preds[label_dim].cpu().detach().squeeze().numpy().tolist())

        score = {}
        for label_dim in label_dims:
            score[label_dim] = eval_fn(full_preds[label_dim], full_labels[label_dim])
        total_loss = losses / sizes
        return total_loss, score