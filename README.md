# Team FeelsGood: MuSe-Perception 2024

[Homepage](https://www.muse-challenge.org) || [Baseline Paper](https://www.researchgate.net/publication/380664467_The_MuSe_2024_Multimodal_Sentiment_Analysis_Challenge_Social_Perception_and_Humor_Recognition)

## Introduction

This repository contains the code and methodology for Team FeelsGood's approach to the [MuSe-Perception challenge](https://www.muse-challenge.org/challenge/sub-challenges), part of the 2024 Multimodal Sentiment Analysis Challenge (MuSe). We extended the [Baseline Approach](https://github.com/amirip/MuSe-2024) with additional models and techniques to enhance social attribute prediction.

If you would like to see our approach and its results please see [Our Paper](https://willbeupdated.com)


## Key Features

- **Basic Approach**: Optimized RNN and Transformer encoder models using two-pronged hyperparameter tuning
- **xLSTM Model**: Implementation of xLSTM for capturing long-term dependencies
- **Text Feature Extraction**: Utilization of various pre-trained Transformer models for linguistic analysis
- **Similar Attribute Grouping**: Joint learning of correlated social attributes
- **Novel Fusion Methods**: Exploration of attribute-level and uni-modal feature fusion techniques

## Dataset

We used the LMU Munich Executive Leadership Perception (LMU-ELP) dataset, which includes audio-visual recordings of CEOs presenting their companies.

## Results

Our approach achieved significant improvements over the baseline for several attributes:

- Best performance for 'aggressive' attribute using xLSTM
- Improved prediction of Agentive attributes using text features
- Overall best results achieved through various fusion methods

## Installation

We recommend using a Python virtual environment. Follow these steps:

1. Create a conda environment using the yaml file `environment_pt220cu121.yaml` (from [xLSTM](https://github.com/NX-AI/xlstm)'s repo)
2. Activate the environment:
   ```bash
   conda activate your_env_here
   ```
3. Install additional dependencies:
   ```bash
   pip install ninja
   conda install cccl
   pip install -r requirements.txt
   pip install wandb
   pip uninstall numpy
   pip install numpy==1.26.4
   pip install xlstm==1.0.3
   pip install dill
   ```
Detailed versions used are available in `environment.yaml`

## Settings

Use the `main.py` script for training and evaluating models. Key changes from baseline code include:

- `--model_type`: Choose `RNN`, `TF`, or `XLSTM`
- `--normalize`: eGeMAPS are always normalized
- `--use_gpu` and `--cache`: Both are on by default

Additional changes:
- WandB sweep compatibility
- Single seed per run
- Predictions saved after each training as xLSTM checkpoints can't be saved or loaded
- Model checkpoints saved for baseline and Transformer models

For more details, see the `parse_args()` method in `main.py`.

## External Resources

- [Our Paper](https://willbeupdated.com): Detailed methodology and findings
- [Trained Models](https://willbeupdated.com): Best performing model weights

## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@inproceedings{{will be updated,
author = {will be updated},
title = {Enhancing Social Attribute Prediction: A Comprehensive
Approach to the MuSe-Perception Task},
year = {2024},
isbn = {{will be updated},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {{will be updated},
doi = {{will be updated},
booktitle = {{will be updated},
pages = {{will be updated},
numpages = {{will be updated},
keywords = {will be updated},
location = {{will be updated},
series = {MuSe' 24}
}
```
