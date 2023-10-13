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

from Downstream.MLP.MLP import MLP
from Downstream.utils.Classifier import Classifier
from Downstream.utils.EarlyStopping import EarlyStopping
from Downstream.utils.metrics import get_metric
from Downstream.utils.utils import set_random_seed, load_dataset, convert_to_gpu, get_optimizer_and_lr_scheduler

args = {
    'dataset': 'OAG',  # OGB_MAG, OAG_CS_Field_F1, OAG_CS_Field_F2, OAG_CS_Venue, Amazon
    'model_name': 'MLP',
    'embedding_name': 'bert_THLM',
    'predict_category': 'review',
    'seed': 0,
    'mode': 'train',
    'cuda': 0,
    'learning_rate': 0.001,
    'hidden_units': [256, 256],
    'dropout': 0.3,
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'epochs': 5000,
    'patience': 200
}
if args['dataset'] == 'OAG':
    args['data_path'] = f"../../Data/OAG_Venue/model/{args['embedding_name']}.pkl"
    args['data_split_idx_path'] = "../../Data/OAG_Venue/downstream/OAG_CS_Venue_Text_split_idx.pkl"
    args['predict_category'] = 'paper'

args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
# args['device'] = 'cpu'


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.set_num_threads(2)

    set_random_seed(args['seed'])

    print(f'loading dataset for {args["dataset"]}')

    graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(data_path=args['data_path'],
                                                                              predict_category=args['predict_category'],
                                                                              data_split_idx_path=args[
                                                                                  'data_split_idx_path'])
    if args['dataset'] == 'amazon':
        labels = labels - 1
    input_features = graph.nodes[args['predict_category']].data['feat'][..., -768:]

    mlp = MLP(input_features.shape[-1], args['hidden_units'], dropout=args['dropout'])

    classifier = Classifier(n_hid=args['hidden_units'][-1], n_out=num_classes)

    model = nn.Sequential(mlp, classifier)


    print(model)

    # print(f'the size of MLP parameters is {count_parameters_in_KB(model[0])} KB.')

    print(f'configuration is {args}')

    model, input_features, labels = convert_to_gpu(model, input_features, labels, device=args['device'])

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=1, epochs=args['epochs'])

    save_model_folder = f"../save_model/{args['dataset']}/{args['model_name']}/{args['embedding_name']}"

    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = nn.CrossEntropyLoss()

    train_steps = 0
    num_classes = labels.max().tolist()+1
    data_tqdm = tqdm(range(args['epochs']), ncols=300)
    if args['mode'] == 'train':
        for epoch in data_tqdm:
            epoch_start_time = time.time()

            model.train()

            nodes_representation = model[0](copy.deepcopy(input_features))
            train_y_predicts = model[1](nodes_representation[train_idx])
            train_y_trues = labels[train_idx]
            loss = loss_func(train_y_predicts, train_y_trues)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)

            # train_accuracy, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1),
            #                                                              labels=train_y_trues)
            # config_saver.record({"acc": train_accuracy, "macroF1": train_macro_f1}, 0)
            scores = get_metric(y_true=torch.nn.functional.one_hot(train_y_trues.detach().cpu(), num_classes=num_classes), y_pred=train_y_predicts.detach().cpu(),
                                idx=-1, method='macro', stage='train')                                                              

            model.eval()

            with torch.no_grad():
                nodes_representation = model[0](copy.deepcopy(input_features))
                val_y_predicts = model[1](nodes_representation[valid_idx])
                val_y_trues = labels[valid_idx]
                val_loss = loss_func(val_y_predicts, val_y_trues)

                # val_accuracy, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
                #                                                          labels=val_y_trues)
                # config_saver.record({"acc": val_accuracy, "macroF1": val_macro_f1}, 1)
            
                scores = get_metric(y_true=torch.nn.functional.one_hot(val_y_trues.detach().cpu(), num_classes=num_classes), y_pred=val_y_predicts.detach().cpu(),
                                idx=-1, method='macro', stage='valid')                                                              


                test_y_predicts = model[1](nodes_representation[test_idx])
                test_y_trues = labels[test_idx]
                test_loss = loss_func(test_y_predicts, test_y_trues)

                # test_accuracy, test_macro_f1 = evaluate_node_classification(predicts=test_y_predicts.argmax(dim=1),
                #                                                             labels=test_y_trues)
                # config_saver.record({"acc": test_accuracy, "macroF1": test_macro_f1}, 2)
            
                scores = get_metric(y_true=torch.nn.functional.one_hot(test_y_trues.detach().cpu(), num_classes=num_classes), y_pred=test_y_predicts.detach().cpu(),
                                idx=-1, method='macro', stage='test')                                                              


            epoch_time = time.time() - epoch_start_time
            data_tqdm.set_description(
            f'Epoch: {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}, cost seconds {epoch_time}, '
            f'train loss: {loss.item():.4f}, '
            #f'accuracy {train_accuracy:.4f}, macro f1 {train_macro_f1:.4f}, \n'
            f'valid loss: {val_loss.item():.4f}, '
            #f'accuracy {val_accuracy:.4f}, macro f1 {val_macro_f1:.4f}, \n'
            f'test loss: {test_loss.item():.4f}, ')
            # f'accuracy {test_accuracy:.4f}, macro f1 {test_macro_f1:.4f}')

            # early_stop = early_stopping.step([('accuracy', val_accuracy, True), ('macro_f1', val_macro_f1, True)], model)
            early_stop = early_stopping.step([('ndcg_1', scores['ndcg_1'], True)], model)

            if early_stop:
                break

    # load best model
        early_stopping.load_checkpoint(model)
    # evaluate the best model
    else:
        params = torch.load(f"../save_model/OAG/MLP/{args['embedding_name']}/MLP.pkl", map_location='cpu')
        model.load_state_dict(params)
        model = convert_to_gpu(model, device=args['device'])
    model.eval()
    with torch.no_grad():
        nodes_representation = model[0](copy.deepcopy(input_features))
        embed_path = f"../save_emb/{args['dataset']}/"
        if not os.path.exists(embed_path):
            os.makedirs(embed_path, exist_ok=True)
        torch.save(nodes_representation,
                   f"../save_emb/{args['dataset']}/{args['embedding_name']}_{args['model_name']}.pkl")

        train_y_predicts = model[1](nodes_representation[train_idx])
        train_y_trues = labels[train_idx]

        # train_accuracy, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1),
        #                                                               labels=train_y_trues)

        # print(f'final train accuracy: {train_accuracy:.4f}, macro f1 {train_macro_f1:.4f}')
        # config_saver.record({"acc": train_accuracy, "macroF1": train_macro_f1}, 0)
        train_scores = get_metric(y_true=torch.nn.functional.one_hot(train_y_trues.detach().cpu(), num_classes=num_classes), y_pred=train_y_predicts.detach().cpu(),
                                idx=-1, method='micro', stage='valid')                                                              

        val_y_predicts = model[1](nodes_representation[valid_idx])
        val_y_trues = labels[valid_idx]
        # val_accuracy, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
        #                                                          labels=val_y_trues)

        # print(f'final valid accuracy {val_accuracy:.4f}, macro f1 {val_macro_f1:.4f}')
        # config_saver.record({"acc": val_accuracy, "macroF1": val_macro_f1}, 1)
        
        val_scores = get_metric(y_true=torch.nn.functional.one_hot(val_y_trues.detach().cpu(), num_classes=num_classes), y_pred=val_y_predicts.detach().cpu(),
                                idx=-1, method='micro', stage='valid')                                                              


        test_y_predicts = model[1](nodes_representation[test_idx])
        test_y_trues = labels[test_idx]
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
    sys.exit()
