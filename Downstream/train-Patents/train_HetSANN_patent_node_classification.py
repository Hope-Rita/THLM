import logging
import time
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
import dgl
import numpy as np

from Downstream.HetSANN.HetSANN import HetSANN
from Downstream.utils.Classifier import Classifier
from Downstream.utils.EarlyStopping import EarlyStopping
from Downstream.utils.metrics import get_metric
from Downstream.utils.utils import set_random_seed, load_dataset, convert_to_gpu, get_optimizer_and_lr_scheduler, \
    get_node_data_loader, get_n_params, load_patent_dataset

args = {'dataset': 'book',
        'model_name': 'HetSANN',
        'embedding_name': 'bert_THLM',
        'predict_category': 'patent',
        'seed': 0,
        'cuda': 3,
        'mode': 'train',
        'learning_rate': 0.001,
        'num_heads': 8,    # Number of attention heads
        'hidden_units': 32,
        'dropout': 0.3,
        'n_layers': 2,
        'residual': True,
        'batch_size': 2560,  # the number of graphs to train in each batch
        'node_neighbors_min_num': 10,  # number of sampled edges for each type for each GNN layer
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'epochs': 500,
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
device = torch.cuda.set_device('cuda:{}'.format(args["cuda"]))


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


def evaluate(model: nn.Module, loader: dgl.dataloading.NodeDataLoader, loss_func: nn.Module,
             truths: torch.Tensor, codes, predict_category: str, device: str, mode: str):
    """

    :param model: model
    :param loader: data loader (validate or test)
    :param loss_func: loss function
    :param labels: node labels
    :param predict_category: str
    :param device: device str
    :param mode: str, evaluation mode, validate or test
    :return:
    """
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_predicts = []
        total_loss = 0.0
        loader_tqdm = tqdm(loader, ncols=120)
        for batch, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            # nodes representation with all types in the heterogeneous graph
            input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}

            # dictionary of all types of nodes representation
            nodes_representation = model[0](blocks, copy.deepcopy(input_features))

            # Tensor, (samples_num, )
            y_predict = model[1](nodes_representation[predict_category])

            # Tensor, (samples_num, )
            # labels = return_truth(output_nodes[args['predict_category']], truth, codes)
            val_labels = truths[output_nodes[args['predict_category']]]
            y_true = convert_to_gpu(val_labels, device=device)
            loss = loss_func(y_predict, y_true)

            total_loss += loss.item()
            y_trues.append(y_true.detach().cpu())
            y_predicts.append(y_predict.detach().cpu())

            loader_tqdm.set_description(f'{mode} for the {batch}-th batch, {mode} loss: {loss.item()}')

        total_loss /= (batch + 1)
        y_trues = torch.cat(y_trues, dim=0)
        y_predicts = torch.cat(y_predicts, dim=0)

    return total_loss, y_trues, y_predicts


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(args['seed'])
    torch.set_num_threads(1)

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

    print(f'loading dataset {args["dataset"]}...')

    graph, train_idx, valid_idx, test_idx = load_patent_dataset(data_path=args['data_path'],
                                                                predict_category=args['predict_category'],
                                                                data_split_idx_path=args[
                                                                    'data_split_idx_path'])

    print(f'get node data loader...')
    train_loader, val_loader, test_loader = get_node_data_loader(args['node_neighbors_min_num'], args['n_layers'],
                                                                 graph,
                                                                 batch_size=args['batch_size'],
                                                                 sampled_node_type=args['predict_category'],
                                                                 train_idx=train_idx, valid_idx=valid_idx,
                                                                 test_idx=test_idx)
    truth, codes = load_truth(args["truth_path"])
    id_list = torch.arange(graph.num_nodes(args['predict_category']))
    labels = return_truth(id_list, truth, codes)

    hetSANN = HetSANN(graph=graph,
                      input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                      hidden_dim=args['hidden_units'], num_layers=args['n_layers'],
                      n_heads=args['num_heads'], dropout=args['dropout'], residual=args['residual'])

    classifier = Classifier(n_hid=args['hidden_units'] * args['num_heads'], n_out=codes)

    model = nn.Sequential(hetSANN, classifier)

    run_dir = Path(
        f"../results/{args['model_name']}/{args['dataset']}/emb_{args['embedding_name']}_hidden_units_{args['hidden_units']}_seed_{args['seed']}_lr_{args['learning_rate']}_dp_{args['dropout']}")
    if not run_dir.exists():
        os.makedirs(str(run_dir))


    model = convert_to_gpu(model, device=args['device'])
    logger.info(model)

    logger.info(f'Model #Params: {get_n_params(model)}.')

    logger.info(f'configuration is {args}')

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=len(train_loader), epochs=args['epochs'])

    save_model_folder = f"../save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}"

    # shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = nn.BCEWithLogitsLoss()

    train_steps = 0
    if args['mode'] == 'train':
        for epoch in range(args['epochs']):
            model.train()

            train_y_trues = []
            train_y_predicts = []
            train_total_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, ncols=120)
            for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
                blocks = [convert_to_gpu(b, device=args['device']) for b in blocks]
                # nodes representation with all types in the heterogeneous graph
                input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}

                # dictionary of all types of nodes representation
                nodes_representation = model[0](blocks, copy.deepcopy(input_features))

                train_y_predict = model[1](nodes_representation[args['predict_category']])

                # Tensor, (samples_num, )
                # labels = return_truth(output_nodes[args['predict_category']], truth, codes)
                train_labels = labels[output_nodes[args['predict_category']]]
                train_y_true = convert_to_gpu(train_labels, device=args['device'])

                loss = loss_func(train_y_predict, train_y_true)


                train_total_loss += loss.item()
                train_y_trues.append(train_y_true.detach().cpu())
                train_y_predicts.append(train_y_predict.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loader_tqdm.set_description(f'training for the {batch}-th batch, train loss: {loss.item()}')

                # step should be called after a batch has been used for training.
                train_steps += 1
                scheduler.step(train_steps)

            train_total_loss /= (batch + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            scores = get_metric(y_true=train_y_trues, y_pred=train_y_predicts.detach().cpu(),
                                idx=1, method='macro', stage='train')

            model.eval()

            val_total_loss, val_y_trues, val_y_predicts = evaluate(model, loader=val_loader, loss_func=loss_func,
                                                                   truths=labels, codes=codes,
                                                                   predict_category=args['predict_category'],
                                                                   device=args['device'],
                                                                   mode='validate')
            val_scores = get_metric(y_true=val_y_trues, y_pred=val_y_predicts.detach().cpu(),
                                    method='macro', stage='valid')


            test_total_loss, test_y_trues, test_y_predicts = evaluate(model, loader=test_loader, loss_func=loss_func,
                                                                      truths=labels, codes=codes,
                                                                      predict_category=args['predict_category'],
                                                                      device=args['device'],
                                                                      mode='test')
            test_scores = get_metric(y_true=test_y_trues, y_pred=test_y_predicts.detach().cpu(),
                                     method='macro', stage='valid')


            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, '
                f'train loss: {train_total_loss:.4f}, '
                f'train metric: {scores}, \n'
                f'valid loss: {val_total_loss:.4f}, '
                f'valid metric: {val_scores} \n'
                f'test loss: {test_total_loss:.4f}, '
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
        params = torch.load(f"../save_model/{args['dataset']}/HetSANN/{args['embedding_name']}/HetSANN.pkl", map_location='cpu')
        model.load_state_dict(params)
        model = convert_to_gpu(model, device=args['device'])
    # evaluate the best model
    model.eval()

    nodes_representation = model[0].inference(graph, copy.deepcopy(
        {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}), device=args['device'])
    torch.save(nodes_representation, f"../save_emb/{args['dataset']}/{args['embedding_name']}_{args['model_name']}.pkl")

    # clus_res = clustering(labels, nodes_representation["patent"].cpu())
    # config_saver.record(clus_res, 2)

    train_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        train_idx]
    labels_train = labels[train_idx.tolist()]
    train_scores = get_metric(y_true=labels_train, y_pred=train_y_predicts.detach().cpu(),
                              method='micro', stage='valid')
    logger.info(f'final train metric: {train_scores}')

    val_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        valid_idx]
    labels_valid = labels[valid_idx.tolist()]
    val_scores = get_metric(y_true=labels_valid, y_pred=val_y_predicts.detach().cpu(),
                            method='micro', stage='valid')
    logger.info(f'final valid metric: {val_scores}')

    test_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        test_idx]
    labels_test = labels[test_idx.tolist()]
    test_scores = get_metric(y_true=labels_test, y_pred=test_y_predicts.detach().cpu(),
                             method='micro', stage='test')
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
    save_result_path = os.path.join(save_result_folder,
                                    f"emb_{args['embedding_name']}_hidden_units_{args['hidden_units']}_seed_{args['seed']}_lr_{args['learning_rate']}_dp_{args['dropout']}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)
    sys.exit()
