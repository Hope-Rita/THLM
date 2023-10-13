import logging
from pathlib import Path

import torch
import torch.nn as nn
import warnings
import copy
import os
import json
import sys
import shutil
from tqdm import tqdm
import time
import numpy as np

from Downstream.MLP.MLP import MLP
from Downstream.utils.Classifier import Classifier
from Downstream.utils.EarlyStopping import EarlyStopping
from Downstream.utils.metrics import get_metric
from Downstream.utils.utils import set_random_seed, load_dataset, convert_to_gpu, get_optimizer_and_lr_scheduler, \
    get_node_data_loader, get_n_params, load_patent_dataset

args = {
    'dataset': 'book',
    'model_name': 'MLP',
    'embedding_name': 'ours_rgcn',
    'predict_category': 'patent',
    'seed': 0,
    'mode': 'train',
    'cuda': 1,
    'learning_rate': 0.001,
    'hidden_units': [256, 256],
    'dropout': 0.0,
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'epochs': 5000,
    'patience': 50
}
if args['dataset'] == 'patent':
    args["truth_path"] = "../../Data/Patents/downstream/com_real_truth.json"
    args['data_path'] = f"../../Data/Patents/model/{args['embedding_name']}.pkl"
    args['data_split_idx_path'] = f'../../Data/Patents/model/split_idx.pkl'
else:
    args['data_split_idx_path'] = "../../Data/GoodReads/downstream/split_idx.pkl"
    args["truth_path"] = "../../Data/GoodReads/downstream/com_real_truth.json"
    args['data_path'] = f"../../Data/GoodReads/model/{args['embedding_name']}.pkl"
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
torch.cuda.set_device('cuda:{}'.format(args["cuda"]))


def load_truth(truth_path):
    with open(truth_path, "r") as f:
        truths = json.load(f)
    max_id = -1
    for i, codes in truths.items():
        max_id = max(max_id, max(codes))
    return truths, max_id + 1


def return_truth(ids: torch.Tensor, Truth, codes):
    all_truths = []
    ids = ids.numpy().tolist()
    for id in ids:
        code_real = torch.tensor(Truth[str(id)])
        truth = torch.nn.functional.one_hot(code_real, num_classes=codes).sum(dim=0)
        assert (truth < 2).all() and truth.sum() > 0
        all_truths.append(truth.float())
    return torch.stack(all_truths, dim=0)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(args['seed'])
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"../log/{args['model_name']}/{args['dataset']}/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"../log/{args['model_name']}/{args['dataset']}/"
                             f"emb_{args['embedding_name']}_hidden_units_{args['hidden_units']}_seed_{args['seed']}_lr_{args['learning_rate']}_dp_{args['dropout']}-{time.time()}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)


    print(f'loading dataset for {args["dataset"]}')

    graph, train_idx, valid_idx, test_idx = load_patent_dataset(data_path=args['data_path'],
                                                                predict_category=args['predict_category'],
                                                                data_split_idx_path=args[
                                                                    'data_split_idx_path'])
    train_idx, valid_idx, test_idx = torch.from_numpy(train_idx), torch.from_numpy(valid_idx), torch.from_numpy(test_idx)
    truth, codes = load_truth(args["truth_path"])

    labels = return_truth(torch.arange(graph.num_nodes(args['predict_category'])), truth, codes)
    labels = convert_to_gpu(labels, device=args['device'])

    input_features = convert_to_gpu(graph.nodes[args['predict_category']].data['feat'][..., -768:], device=args['device'])
    mlp = MLP(input_features.shape[-1], args['hidden_units'], dropout=args['dropout'])

    classifier = Classifier(n_hid=args['hidden_units'][-1], n_out=codes)

    model = nn.Sequential(mlp, classifier)

    # params = torch.load("../save_model/patent/MLP/bert_hgnnI_0/MLP.pkl", map_location='cpu')
    # model.load_state_dict(params)

    model = convert_to_gpu(model, device=args['device'])

    print(model)
    # print(f'the size of MLP parameters is {count_parameters_in_KB(model[0])} KB.')

    print(f'configuration is {args}')


    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=1, epochs=args['epochs'])

    save_model_folder = f"../save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}"

    # shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = nn.CrossEntropyLoss()

    train_steps = 0

    data_tqdm = tqdm(range(args['epochs']), ncols=300)
    if args['mode'] == 'train':
        for epoch in data_tqdm:
            epoch_start_time = time.time()

            model.train()

            nodes_representation = model[0](copy.deepcopy(input_features))
            train_y_predicts = model[1](nodes_representation[train_idx])
            # labels = return_truth(train_idx, truth, codes)
            # train_y_true = convert_to_gpu(labels[train_idx], device=args['device'])
            train_y_true = labels[train_idx]

            loss = loss_func(train_y_predicts, train_y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)
            if epoch % 10 == 0:
                scores = get_metric(y_true=train_y_true.detach().cpu(), y_pred=train_y_predicts.detach().cpu(),
                                idx=1, method='macro', stage='train')

            else:
                scores = 0
            model.eval()

            with torch.no_grad():
                nodes_representation = model[0](copy.deepcopy(input_features))
                val_y_predicts = model[1](nodes_representation[valid_idx])
                # labels = return_truth(valid_idx, truth, codes)
                # val_y_trues = convert_to_gpu(labels[valid_idx], device=args['device'])
                val_y_trues = labels[valid_idx]
                val_loss = loss_func(val_y_predicts, val_y_trues)

                val_scores = get_metric(y_true=val_y_trues.detach().cpu(), y_pred=val_y_predicts.detach().cpu(),
                                        method='macro', stage='valid')

                test_y_predicts = model[1](nodes_representation[test_idx])
                # labels = return_truth(test_idx, truth, codes)
                # test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])
                test_y_trues = labels[test_idx]
                test_loss = loss_func(test_y_predicts, test_y_trues)

                test_scores = get_metric(y_true=test_y_trues.detach().cpu(), y_pred=test_y_predicts.detach().cpu(),
                                        method='macro', stage='test')
            epoch_time = time.time() - epoch_start_time
            print()
            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {loss:.4f}, '
                f'train metric: {scores}, \n'
                f'valid loss: {val_loss:.4f}, '
                f'valid metric: {val_scores} \n'
                f'test loss: {test_loss:.4f}, '
                f'test metric: {test_scores}')
            validate_ndcg_list, validate_pre_list = [], []
            for key in val_scores:
                if key.startswith('ndcg_'):
                    # if key.startswith('recall_'):
                    validate_ndcg_list.append(val_scores[key])
                elif key.startswith('recall_'):
                    validate_pre_list.append(val_scores[key])
            validate_ndcg = np.mean(validate_ndcg_list)
            validate_prec = np.mean(validate_pre_list)

            early_stop = early_stopping.step([('ndcg', validate_ndcg, True)], model)

            if early_stop:
                break
        early_stopping.load_checkpoint(model)
    else:
        params = torch.load(f"../save_model/{args['dataset']}/MLP/{args['embedding_name']}/MLP.pkl", map_location='cpu')
        model.load_state_dict(params)
        model = convert_to_gpu(model, device=args['device'])
    # load best model
    # evaluate the best model
    model.eval()
    with torch.no_grad():
        nodes_representation = model[0](copy.deepcopy(input_features))
        embed_path = f"../save_emb/{args['dataset']}/"
        if not os.path.exists(embed_path):
            os.makedirs(embed_path, exist_ok=True)
        torch.save(nodes_representation,
                   f"../save_emb/{args['dataset']}/{args['embedding_name']}_{args['model_name']}.pkl")
        # clus_res = clustering(labels, nodes_representation.cpu())
        # config_saver.record(clus_res, 2)

        train_y_predicts = model[1](nodes_representation[train_idx])

        # labels_train = return_truth(torch.tensor(train_idx), truth, codes)

        # train_y_trues = convert_to_gpu(labels[train_idx], device=args['device'])
        train_y_trues = labels[train_idx]
        train_scores = get_metric(y_true=train_y_trues.cpu(), y_pred=train_y_predicts.detach().cpu(),
                                   idx=0, method='micro', stage='valid')
        logger.info(f'final train metric: {train_scores}')

        val_y_predicts = model[1](nodes_representation[valid_idx])
        # labels_valid = return_truth(torch.tensor(valid_idx), truth, codes)
        # val_y_trues = convert_to_gpu(labels[valid_idx], device=args['device'])
        val_y_trues = labels[valid_idx]
        val_scores = get_metric(y_true=val_y_trues.cpu(), y_pred=val_y_predicts.detach().cpu(),
                                   method='micro', stage='valid')
        logger.info(f'final valid metric: {val_scores}')

        test_y_predicts = model[1](nodes_representation[test_idx])
        # labels_test = return_truth(torch.tensor(test_idx), truth, codes)
        # test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])
        test_y_trues = labels[test_idx]
        test_scores = get_metric(y_true=test_y_trues.cpu(), y_pred=test_y_predicts.detach().cpu(),
                                   method='micro', stage='valid')
        logger.info(f'final test metric: {test_scores}')

        # save model result
        result_json = {
            "train accuracy": train_scores,
            "validate accuracy": val_scores,
            "test accuracy": test_scores
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"../results/{args['dataset']}/{args['model_name']}/{args['embedding_name']}"
        if not os.path.exists(save_result_folder):
            os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"emb_{args['embedding_name']}_hidden_units_{args['hidden_units']}_seed_{args['seed']}_lr_{args['learning_rate']}_dp_{args['dropout']}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)
        sys.exit()
