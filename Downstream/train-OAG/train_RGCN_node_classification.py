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

from Downstream.RGCN.RGCN import RGCN


from Downstream.utils.Classifier import Classifier
from Downstream.utils.EarlyStopping import EarlyStopping
from Downstream.utils.metrics import get_metric
from Downstream.utils.utils import set_random_seed, load_dataset, convert_to_gpu, get_optimizer_and_lr_scheduler, \
    get_node_data_loader, get_n_params

args = {'dataset': 'OAG',
        'model_name': 'RGCN',
        'embedding_name': 'bert_THLM',
        'mode': 'test',
        'seed': 0,
        'predict_category': 'paper',
        'cuda': 2,
        'learning_rate': 0.001,
        'hidden_units': [256, 256],
        'n_layers': 2,
        'dropout': 0.5,
        'n_bases': -1,
        'use_self_loop': True,
        'batch_size': 4096,
        'node_neighbors_min_num': 10,
        'optimizer': 'adam', 'weight_decay': 0, 'epochs': 500, 'patience': 30}

if args['dataset'] == 'OAG':
    args['data_path'] = f"../../Data/OAG_Venue/model/{args['embedding_name']}.pkl"
    args['data_split_idx_path'] = "../../Data/OAG_Venue/downstream/OAG_CS_Venue_Text_split_idx.pkl"
    args['predict_category'] = 'paper'

args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
device = torch.cuda.set_device('cuda:{}'.format(args["cuda"]))
# device = 'cpu'


def evaluate(model: nn.Module, loader: dgl.dataloading.NodeDataLoader, loss_func: nn.Module,
             labels: torch.Tensor, predict_category: str, device: str, mode: str):
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
            # target node relation representation in the heterogeneous graph
            input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}

            nodes_representation = model[0](blocks, copy.deepcopy(input_features))

            y_predict = model[1](nodes_representation[predict_category])

            # Tensor, (samples_num, )
            y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

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
    torch.set_num_threads(1)

    set_random_seed(args['seed'])

    print(f'loading dataset {args["dataset"]}...')

    graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(data_path=args['data_path'],
                                                                              predict_category=args['predict_category'],
                                                                              data_split_idx_path=args[
                                                                                  'data_split_idx_path'])
    if args['dataset'] == 'amazon':
        labels = labels - 1
        print(len(labels), graph.num_nodes(args['predict_category']))
    print(f'get node data loader...')

    # for ntype in graph.ntypes:
    #     graph.nodes[ntype].data['feat'] = torch.rand(graph.num_nodes(ntype), 768)

    train_loader, val_loader, test_loader = get_node_data_loader(args['node_neighbors_min_num'], args['n_layers'],
                                                                 graph,
                                                                 batch_size=args['batch_size'],
                                                                 sampled_node_type=args['predict_category'],
                                                                 train_idx=train_idx, valid_idx=valid_idx,
                                                                 test_idx=test_idx)

    rgcn = RGCN(graph=graph, input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                hidden_sizes=args['hidden_units'], num_bases=args['n_bases'], dropout=args['dropout'],
                use_self_loop=args['use_self_loop'])

    classifier = Classifier(n_hid=args['hidden_units'][-1], n_out=num_classes)

    model = nn.Sequential(rgcn, classifier)

    model = convert_to_gpu(model, device=args['device'])
    print(model)

    print(f'Model #Params: {get_n_params(model)}.')

    print(f'configuration is {args}')

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=len(train_loader), epochs=args['epochs'])

    save_model_folder = f"../save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}"

    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = nn.CrossEntropyLoss()

    train_steps = 0
    num_classes = labels.max().tolist()+1
    if args['mode'] == 'train':
        for epoch in range(args['epochs']):
            model.train()

            train_y_trues = []
            train_y_predicts = []
            train_total_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, ncols=120)
            for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
                blocks = [convert_to_gpu(b, device=args['device']) for b in blocks]
                # target node relation representation in the heterogeneous graph
                input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}

                nodes_representation= model[0](blocks, copy.deepcopy(input_features))

                train_y_predict = model[1](nodes_representation[args['predict_category']])

                # Tensor, (samples_num, )
                train_y_true = convert_to_gpu(labels[output_nodes[args['predict_category']]], device=args['device'])
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

            # train_accuracy, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1),
            #                                                              labels=train_y_trues)
            # config_saver.record({"acc": train_accuracy, "macroF1": train_macro_f1}, 0)
            scores = get_metric(y_true=torch.nn.functional.one_hot(train_y_trues.detach().cpu(), num_classes=num_classes), y_pred=train_y_predicts.detach().cpu(),
                                idx=-1, method='macro', stage='train')                                                              

            model.eval()

            val_total_loss, val_y_trues, val_y_predicts = evaluate(model, loader=val_loader, loss_func=loss_func,
                                                                   labels=labels,
                                                                   predict_category=args['predict_category'],
                                                                   device=args['device'],
                                                                   mode='validate')

            # val_accuracy, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
            #                                                          labels=val_y_trues)
            # config_saver.record({"acc": val_accuracy, "macroF1": val_macro_f1}, 1)
            
            scores = get_metric(y_true=torch.nn.functional.one_hot(val_y_trues.detach().cpu(), num_classes=num_classes), y_pred=val_y_predicts.detach().cpu(),
                                idx=-1, method='macro', stage='valid')                                                              


            test_total_loss, test_y_trues, test_y_predicts = evaluate(model, loader=test_loader, loss_func=loss_func,
                                                                      labels=labels,
                                                                      predict_category=args['predict_category'],
                                                                      device=args['device'],
                                                                      mode='test')

            # test_accuracy, test_macro_f1 = evaluate_node_classification(predicts=test_y_predicts.argmax(dim=1),
            #                                                             labels=test_y_trues)
            # config_saver.record({"acc": test_accuracy, "macroF1": test_macro_f1}, 2)
            
            scores = get_metric(y_true=torch.nn.functional.one_hot(test_y_trues.detach().cpu(), num_classes=num_classes), y_pred=test_y_predicts.detach().cpu(),
                                idx=-1, method='macro', stage='test')                                                              



            print(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}, '
                # f'accuracy {train_accuracy:.4f}, macro f1 {train_macro_f1:.4f}, \n'
                f'valid loss: {val_total_loss:.4f}, '
                # f'accuracy {val_accuracy:.4f}, macro f1 {val_macro_f1:.4f} \n'
                f'test loss: {test_total_loss:.4f}, ')
                # f'accuracy {test_accuracy:.4f}, macro f1 {test_macro_f1:.4f}')

            early_stop = early_stopping.step([('ndcg_1', scores['ndcg_1'], True)], model)

            if early_stop:
                break
    else:
        param_path = f"../save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}/RGCN.pkl"
        params = torch.load(param_path, map_location='cpu')
        model.load_state_dict(params)
        model = convert_to_gpu(model, device=args['device'])
    early_stopping.load_checkpoint(model)
    # model.load_state_dict(torch.load("../save_model/OGB_MAG/R_HGNN_lr0.001_dropout0.5_seed_0_bert/R_HGNN_lr0.001_dropout0.5_seed_0_bert.pkl"))
    # evaluate the best model
    model.eval()

    nodes_representation = model[0].inference(graph, copy.deepcopy(
        {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}), device=args['device'])
    embed_path = f"../save_emb/{args['dataset']}/"
    if not os.path.exists(embed_path):
        os.makedirs(embed_path, exist_ok=True)
    torch.save(nodes_representation,
               f"../save_emb/{args['dataset']}/{args['embedding_name']}_{args['model_name']}.pkl")


    train_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        train_idx]
    train_y_trues = convert_to_gpu(labels[train_idx], device=args['device'])
    # train_accuracy, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1),
    #                                                               labels=train_y_trues)

    # print(f'final train accuracy: {train_accuracy:.4f}, macro f1 {train_macro_f1:.4f}')
    # config_saver.record({"acc": train_accuracy, "macroF1": train_macro_f1}, 0)
    train_scores = get_metric(y_true=torch.nn.functional.one_hot(train_y_trues.detach().cpu(), num_classes=num_classes), y_pred=train_y_predicts.detach().cpu(),
                                idx=-1, method='micro', stage='valid')                                                              

    
    val_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        valid_idx]
    val_y_trues = convert_to_gpu(labels[valid_idx], device=args['device'])
    # val_accuracy, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
        #                                                          labels=val_y_trues)

        # print(f'final valid accuracy {val_accuracy:.4f}, macro f1 {val_macro_f1:.4f}')
        # config_saver.record({"acc": val_accuracy, "macroF1": val_macro_f1}, 1)
        
    val_scores = get_metric(y_true=torch.nn.functional.one_hot(val_y_trues.detach().cpu(), num_classes=num_classes), y_pred=val_y_predicts.detach().cpu(),
                                idx=-1, method='micro', stage='valid')                                                              


    test_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        test_idx]
    test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])
    # test_accuracy, test_macro_f1 = evaluate_node_classification(predicts=test_y_predicts.argmax(dim=1),
        #                                                             labels=test_y_trues)
        # print(f'final test accuracy {test_accuracy:.4f}, macro f1 {test_macro_f1:.4f}')
        # config_saver.record({"acc": test_accuracy, "macroF1": test_macro_f1}, 2)
        
    test_scores = get_metric(y_true=torch.nn.functional.one_hot(test_y_trues.detach().cpu(), num_classes=num_classes), y_pred=test_y_predicts.detach().cpu(),
                                idx=-1, method='micro', stage='test')                                                              


    # save model result
    result_json = {
            "train accuracy": train_scores,
            "validate accuracy": val_scores,
            "test accuracy": test_scores
        }
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"../results/{args['dataset']}/{args['model_name']}"
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args['embedding_name']}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)

    print(result_json)
    sys.exit()
