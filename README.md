# Pretraining Language Models with Text-Attributed Heterogeneous Graphs
## Data Preparation
Please download the datasets from [DatasetsForTHLM](https://drive.google.com/drive/folders/1LtxOPUNOS14jiFc1sLd8i0rXFPOhXCbV?usp=sharing) , and put it into `./Data`

## Model Pretraining

Example of training THLM on Patents dataset:

`python main.py --dataset_name Patents`

## Get node embeddings
Obtain node embeddings for Patents, GoodReads and OAG_Venue in `./Downstream/preprocess_data`

Example of obtaining node embeddings for Patents:

`python Patent_features.py`

## Model Evaluation
* Link Prediction for OAG_Venue: `./Downstream/Link-Train-OAG`
* Link Prediction for Patents/GoodReads: `./Downstream/Link-Train-Patent`

* Node Classification for OAG_Venue: `./Downstream/train-OAG`
* Node Classification for Patents/GoodReads: `./Downstream/train-Patent`

## Pre-trained Language Models
We also provide the pre-trained language models on these three datasets at [HuggingFace](https://huggingface.co/Xiaoqii/THLM).

## Environments:
* PyTorch 2.0.0
* transformers 4.23.1
* dgl 0.9.1
* tqdm
* numpy

