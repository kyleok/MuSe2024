import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import joblib
import wandb
import os
import string
import random
from scipy import stats

LABELS_FILE = "labels.csv"
PARTITION_FILE = "partition.csv"
TEST_FILE = "best.csv"
NEW_PREDICTION_FILE = "best_devel.csv"  # New file for the dataset to predict on


def generate_random_string(length=10):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def load_data():
    labels = pd.read_csv(LABELS_FILE, index_col='subj_id')
    partition = pd.read_csv(PARTITION_FILE, index_col='Id')
    data = labels.join(partition, how='inner')
    data = data.dropna()
    return data


def load_test_data():
    """Load test data from best.csv"""
    test_data = pd.read_csv(TEST_FILE, index_col='subj_id')
    return test_data


def load_new_prediction_data():
    """Load new data to predict on"""
    new_data = pd.read_csv(NEW_PREDICTION_FILE, index_col='subj_id')
    return new_data


def prepare_data(data, target_characteristic):
    X = data.drop(['Partition', target_characteristic], axis=1)
    y = data[target_characteristic]

    X_train = X[data['Partition'] == 'train']
    y_train = y[data['Partition'] == 'train']
    X_val = X[data['Partition'] == 'devel']
    y_val = y[data['Partition'] == 'devel']
    return X_train, X_val, y_train, y_val


def get_model(model_type, config):
    if model_type == 'linear':
        return LinearRegression()
    elif model_type == 'lasso':
        return Lasso(alpha=config['alpha'])
    elif model_type == 'ridge':
        return Ridge(alpha=config['alpha'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def calc_pearsons(preds, labels):
    if isinstance(preds, pd.Series):
        preds = preds.values
    elif not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    if isinstance(labels, pd.Series):
        labels = labels.values
    elif not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    r = stats.pearsonr(preds, labels)
    return r[0]


def train_and_predict(pipeline, X_train, y_train, X_val, X_test, X_new):
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    test_predictions = pipeline.predict(X_test)
    new_predictions = pipeline.predict(X_new)
    return pipeline, val_predictions, test_predictions, new_predictions


def get_feature_importance(pipeline, feature_names):
    if hasattr(pipeline.named_steps['model'], 'coef_'):
        importances = np.abs(pipeline.named_steps['model'].coef_)
        return dict(zip(feature_names, importances))
    else:
        wandb.termlog("Model does not have coefficients for feature importance.")
        return None


def main(args):
    wandb.init(config=args, entity='feelsgood_muse', project='15prediction_final_DEVEL')
    config = wandb.config

    wandb.termlog(f"Loading training data from {LABELS_FILE} and {PARTITION_FILE}")
    data = load_data()

    wandb.termlog(f"Preparing data for characteristic: {config.characteristic}")
    X_train, X_val, y_train, y_val = prepare_data(data, config.characteristic)

    wandb.termlog(f"Setting up model: {config.model_type}")
    model = get_model(config.model_type, config)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(Lasso(alpha=0.01))),
        ('model', model)
    ])

    wandb.termlog("Loading test data")
    test_data = load_test_data()
    X_test = test_data.drop([config.characteristic], axis=1)
    y_test = test_data[config.characteristic]

    wandb.termlog("Loading new prediction data")
    new_data = load_new_prediction_data()
    X_new = new_data.drop([config.characteristic], axis=1) if config.characteristic in new_data.columns else new_data

    wandb.termlog("Training model and making predictions")
    best_pipeline, val_predictions, test_predictions, new_predictions = train_and_predict(pipeline, X_train, y_train,
                                                                                          X_val, X_test, X_new)

    # Calculate Pearson correlation for validation set
    val_pearson_r = calc_pearsons(val_predictions, y_val)
    wandb.log({"validation_pearson_r": val_pearson_r})

    # Calculate Pearson correlation for new predictions using validation labels
    new_pearson_r = calc_pearsons(new_predictions[:len(y_val)], y_val)
    wandb.log({"new_data_pearson_r": new_pearson_r})

    # Create DataFrames for validation, test, and new predictions
    val_results_df = pd.DataFrame({
        'subj_id': X_val.index,
        'ground_truth': y_val,
        'predicted': val_predictions
    })

    test_results_df = pd.DataFrame({
        'subj_id': X_test.index,
        'predicted': test_predictions
    })

    new_results_df = pd.DataFrame({
        'subj_id': X_new.index,
        'predicted': new_predictions
    })

    # Save predictions to CSV files
    val_file = f'validation_predictions_{config.characteristic}.csv'
    test_file = f'test_predictions_{config.characteristic}.csv'
    new_file = f'new_data_predictions_{config.characteristic}.csv'
    val_results_df.to_csv(val_file, index=False)
    test_results_df.to_csv(test_file, index=False)
    new_results_df.to_csv(new_file, index=False)

    wandb.log({
        "validation_results": wandb.Table(dataframe=val_results_df),
        "test_results": wandb.Table(dataframe=test_results_df),
        "new_data_results": wandb.Table(dataframe=new_results_df)
    })

    wandb.termlog(f"Saved validation predictions to {val_file}")
    wandb.termlog(f"Saved test predictions to {test_file}")
    wandb.termlog(f"Saved new data predictions to {new_file}")

    wandb.termlog("Getting feature importances")
    feature_importances = get_feature_importance(best_pipeline, X_train.columns)
    if feature_importances:
        wandb.log({"feature_importances": wandb.Table(data=[[k, v] for k, v in feature_importances.items()],
                                                      columns=["feature", "importance"])})
    else:
        wandb.termlog("Skipping feature importances as they're not available for this model")

    # Generate a unique filename
    random_string = generate_random_string()
    filename = f'{config.model_type}_{config.characteristic}_{random_string}_model.joblib'

    # Save the model locally
    joblib.dump(best_pipeline, filename)

    # Save the model to wandb
    artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
    artifact.add_file(filename)
    wandb.log_artifact(artifact)

    wandb.termlog("Waiting for model to be uploaded...")
    artifact.wait()
    # Remove the local file
    if os.path.exists(filename):
        os.remove(filename)
        wandb.termlog(f"Model removed from local storage: {filename}")
    else:
        wandb.termlog(f"File {filename} not found. It may have been already removed.")

    wandb.termlog("Run completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run characteristic prediction model")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha parameter for Lasso and Ridge")
    parser.add_argument("--characteristic", type=str, required=True, help="Characteristic to predict")
    parser.add_argument("--model_type", type=str, required=True, choices=['linear', 'lasso', 'ridge'],
                        help="Type of model to use")

    args = parser.parse_args()

    main(args)
