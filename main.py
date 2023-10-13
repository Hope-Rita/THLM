import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm

import time
import os
import json

from transformers import get_linear_schedule_with_warmup
from utils.load_data import Data, Data_OAG, Data_GoodReads
from utils.load_model import Model_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def filterParameter(name, data):
    print(name, data.requires_grad)
    return True


def save_results(path, data):
    pd.DataFrame(data, index=[0]).to_csv(path)


# save parameters of the model
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def reload_model(model, model_path):
    model.load_state_dict(torch.load(model_path))


get_checkpoint_path = lambda \
        epoch: f'./{config["model_folder"]}/THLM.pth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining on TAHGs')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='Patents',
                        choices=['Patents', 'GoodReads', 'OAG_Venue'])
    args = parser.parse_args()
    with open(f"./config/{args.dataset_name}.json", "r") as f:
        config = json.load(f)
    setup_seed(config['seed'])

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"log/{config['pre_text']}/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}/{}.log'.format(config['pre_text'], str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(config)
    gpus = [i for i in range(torch.cuda.device_count())]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    torch.set_num_threads(1)

    if args.dataset_name == 'Patents':
        data_generator = Data(applicant_path=config['applicant_path'], company_path=config['company_path'],
                              abstract_path=config['text_path'], title_path=config['title_path'],
                              filter_patent=config['patent_filter_path'], filter_company=config['company_filter_path'],
                              filter_applicant=config['applicant_filter_path'], graph_path=config['graph_path'],
                              sample_graph_num=config['sample_num'],
                              bert_tokenizer=config['bertName'], batch_size=config['batch_size'],
                              has_mask=config['has_mlm'], sample_title=config["sample_num"],
                              has_neighbor=config['has_neighbor'],
                              pred_layer=config['pred_layer'])
    elif args.dataset_name == 'OAG_Venue':
        data_generator = Data_OAG(author_path=config['author_path'], batch_size=config['batch_size'],
                                  field_path=config['field_path'],
                                  affiliation_path=config['affiliation_path'], paper_path=config['text_path'],
                                  sample_graph_num=config['sample_num'],
                                  bert_tokenizer=config['bertName'],
                                  has_mask=config['has_mlm'],
                                  has_neighbor=config['has_neighbor'], title_path=config['title_path'])
    elif args.dataset_name == 'GoodReads':

        data_generator = Data_GoodReads(author_path=config['applicant_path'], publisher_path=config['company_path'],
                                        abstract_path=config['text_path'], title_path=config['title_path'],
                                        filter_book=config['patent_filter_path'],
                                        filter_publisher=config['company_filter_path'],
                                        filter_author=config['applicant_filter_path'], graph_path=config['graph_path'],
                                        sample_graph_num=config['sample_num'],
                                        bert_tokenizer=config['bertName'], batch_size=config['batch_size'],
                                        has_mask=config['has_mlm'], sample_title=config["sample_num"],
                                        has_neighbor=config['has_neighbor'],
                                        pred_layer=config['pred_layer'])

    # Step 1: Prepare graph data and device ================================================================= #
    if config['gpu'] >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(gpus[0])
        # pass
    else:
        device = 'cpu'
    model_folder = f"{config['model_folder']}/{args.dataset_name}"
    os.makedirs(model_folder, exist_ok=True)

    # Step 2: Create model and training components=========================================================== #
    model_loader = Model_loader(config, data_generator, device, vocab_num=data_generator.total_vocab,
                                )
    model = model_loader.model
    model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    if not config['pre_path'].startswith("None"):
        model.load_state_dict(torch.load(config['pre_path']))
    no_decay = ["bias", "LayerNorm.weight"]
    gcn_weights = "module.predNeig"

    optimizer_weight_dc = []
    optimizer_no_weight_dc = []
    optimizer_gcn_weight_dc = []
    optimizer_gcn_no_weight_dc = []

    for n, p in model.named_parameters():
        if not n.startswith(gcn_weights):
            if not any(nd in n for nd in no_decay):
                optimizer_weight_dc.append(p)
            else:
                optimizer_no_weight_dc.append(p)
        else:
            if not any(nd in n for nd in no_decay):
                optimizer_gcn_weight_dc.append(p)
            else:
                optimizer_gcn_no_weight_dc.append(p)

    optimizer_grouped_parameters = [
        {
            "params": optimizer_weight_dc,
            "weight_decay": config["weight_decay"],
        },
        {
            "params": optimizer_no_weight_dc,
            "weight_decay": 0.0,
        },
    ]

    optimizer_gcn_parameters = [
        {
            "params": optimizer_gcn_weight_dc,
            "weight_decay": config["weight_decay"],
        },
        {
            "params": optimizer_gcn_no_weight_dc,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['lr'])
    optimizerGCN = torch.optim.AdamW(optimizer_gcn_parameters, lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()
    bce_func = torch.nn.BCEWithLogitsLoss()
    len_dataset = data_generator.total_num
    batch_size, epoch = config['batch_size'], config['epoch']
    if len_dataset % batch_size == 0:
        total_steps = (len_dataset // batch_size) * epoch
    else:
        total_steps = (len_dataset // batch_size + 1) * epoch

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.08 * total_steps,
                                                num_training_steps=total_steps)

    dataloaders = data_generator.dataloader
    # Step 3: training epoches ============================================================================== #
    n_batch = data_generator.total_num // config['batch_size'] + 1
    t0 = time.time()
    loss_all, learning_rates = [], []

    for epoch in range(config['epoch']):

        loss, cott, batch = 0., 0, 0
        model.train()

        train_loader_tqdm = tqdm(data_generator.dataloader, ncols=150)
        for input_nodes, input_types in train_loader_tqdm:
            t1 = time.time()
            raw_texts, mask_texts, mask_ids, token_ids, labels, new_inputs = data_generator.MaskToken(
                [input_nodes, input_types])
            t2 = time.time()
            mask_texts = mask_texts.to(device)
            token_ids = token_ids.to(device)
            select_ids, neigh_labels, masks_neigh = data_generator.return_neiLabels(
                [new_inputs[0], new_inputs[1]])
            for ntype, nlist in select_ids.items():
                select_ids[ntype] = nlist.to(device)
            for ntype, nlist in neigh_labels.items():
                neigh_labels[ntype] = nlist.to(device)
            for ntype, nlist in masks_neigh.items():
                masks_neigh[ntype] = nlist.to(device)

            mlm_pred, neigh_pred_list = model(input_nodes, data_generator.graph, mask_texts,
                                              token_ids,
                                              select_ids, data_generator.nodeEmb)

            batch_loss = 0
            if data_generator.has_msk:
                y_pred = mlm_pred[mask_ids]
                truth = raw_texts.flatten()[mask_ids].to(device)
                mlm_loss = loss_func(y_pred, truth)

                batch_loss = mlm_loss

            if data_generator.has_neighbor:
                for ntype, nmask in masks_neigh.items():
                    loss_ntype = bce_func(neigh_pred_list[ntype] * nmask,
                                          neigh_labels[ntype])

                    batch_loss = batch_loss + loss_ntype

            t3 = time.time()
            optimizer.zero_grad()
            optimizerGCN.zero_grad()
            batch_loss.backward()

            optimizerGCN.step()
            optimizer.step()
            scheduler.step()
            loss += batch_loss.item()
            learning_rates.append(optimizerGCN.state_dict()['param_groups'][0]['lr'])

            t4 = time.time()
            cott += len(raw_texts)
            train_loader_tqdm.set_description(
                'training for the {}-th batch, '
                'BERT lr:{:.8f}, GCN lr:{:.5f}, '
                'train loss: {}'.format(batch, optimizer.state_dict()["param_groups"][0]["lr"],
                                        optimizerGCN.state_dict()["param_groups"][0]["lr"],
                                        batch_loss.item()))

            batch += 1
        torch.save(
            model.state_dict(),
            get_checkpoint_path(epoch))

