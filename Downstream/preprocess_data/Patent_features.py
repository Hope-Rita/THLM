import random

import torch
import dgl
import transformers
from dgl import load_graphs
from dgl.data.utils import save_graphs
import os
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

"""
    First generate features of nodes that are not associated features in the original graphs
    Then add reverse relations in the original graph, assign the node features
    Finally, store the processed graph
"""


class MagDataset():
    def __init__(self, graph_path, patent_path, abstract_path, title_path, company_path, applicant_path,
                 company_filter, applicant_filter,
                 bert_tokenizer, embedding_path):
        self.data_dict, self.total_nodes = {}, {}
        self.data_dict["patent"], self.total_nodes["patent"] = self.load_dict(abstract_path)  # 读取paper信息
        self.data_dict["title"], self.total_nodes["title"] = self.load_dict(title_path)  # 读取paper信息
        self.load_bertTokenizer(bert_tokenizer)
        self.graph = self.load_graph(graph_path)
        self.virtual_info = {}
        self.map_reIndex(abstract_path, patent_path, type="patent")
        self.map_reIndex(title_path, patent_path, type="title")
        self.map_reIndex(company_path, company_filter, type="company")
        self.map_reIndex(applicant_path, applicant_filter, type="applicant")
        self.embedding_dict = {}

    def map_reIndex(self, text_path, patent_path, type='patent'):
        data_dict = pd.read_csv(text_path).dropna().drop_duplicates()  # paper的id信息和paper的摘要信息 N*2
        print(data_dict.head(5))

        data = pd.read_csv(patent_path)  # remap的表格[filter]
        merge_data = pd.merge(data, data_dict, how='left').sort_values(by=data.columns[-1], ascending=True)

        self.total_nodes[type] = merge_data[[data.columns[-1], data_dict.columns[-1]]].dropna().values[:, 0]
        print(self.total_nodes[type])
        print('===================================================')

        texts = []
        for x in tqdm(merge_data.values[:, -1]):
            if pd.isnull(x):
                texts.append(torch.tensor([101]))
                continue
            x_ = str(x).replace(" ", "")[1:-1].split(",")
            texts.append(torch.tensor([int(i) for i in x_]))
        # texts = [[int(i) for i in x[2:-2].split(",')] for x in merge_data.values[:, -1]]
        self.virtual_info[type] = pad_sequence(texts, batch_first=True, padding_value=0).long()

    def load_dict(self, text_path):
        data_dict = pd.read_csv(text_path).dropna().drop_duplicates() 
        total_nodes = data_dict.values[:, 0]
        return data_dict, total_nodes

    def load_bertTokenizer(self, bert_tokenizer):
        self.Bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.total_vocab = self.Bert_tokenizer.vocab_size
        self.cls_id, self.mask_id = self.Bert_tokenizer.cls_token_id, self.Bert_tokenizer.mask_token_id
        self.pad_id, self.sep_id = self.Bert_tokenizer.pad_token_id, self.Bert_tokenizer.sep_token_id

    def load_graph(self, graph_path):
        graph_list, labels = load_graphs(graph_path)
        original_graph = graph_list[0]
        self.labels = labels
        return original_graph

    def reconstructEmbed(self, new_feat):
        original_graph = self.graph
        data_dict = {}
        for src_type, etype, dst_type in original_graph.canonical_etypes:
            src, dst = original_graph.edges(etype=etype)
            data_dict[(src_type, etype, dst_type)] = (src, dst)

        self.new_graph = dgl.heterograph(data_dict=data_dict,
                                         num_nodes_dict={ntype: original_graph.number_of_nodes(ntype) for ntype in
                                                         original_graph.ntypes})

        # concat the content and structural feature of paper
        for key, feats in new_feat.items():
            self.new_graph.nodes[key].data['feat'] = feats

    def rebuildEmbedding(self, new_feats):
        for type, new_feat in new_feats.items():
            self.embedding_dict[type] = new_feat

    def getTextInGraph(self, type, ids): 
        return self.virtual_info[type][ids]  

def reload_model(model, model_path):
    # Use when some parts of pretrained model are not needed
    pretrained_dict = torch.load(model_path, map_location='cpu')
    model_dict = model.state_dict()
    print(model_dict.keys())
    pretrained_dict = {k.replace("module.encoder.", "") if "module.encoder." in k else k: v for k, v in
                       pretrained_dict.items()}
    print(pretrained_dict.keys())

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict}
    print(pretrained_dict.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def concateData(datas: list, bertModel: transformers.BertTokenizer):
    res = []
    token_type = []
    for com_ids, text_ids in zip(datas[0], datas[1]):
        com_id_filter = com_ids[com_ids != 0][1:-1]
        text_id_filter = text_ids[text_ids != 0][1:-1]
        if len(com_id_filter) == 0:
            dt_list = [torch.tensor([bertModel.cls_token_id]), text_id_filter, torch.tensor([bertModel.sep_token_id])]
            token_id = torch.tensor([0] * (len(text_id_filter) + 2))
        else:
            dt_list = [torch.tensor([bertModel.cls_token_id]), com_id_filter, torch.tensor([bertModel.sep_token_id]),
                       text_id_filter, torch.tensor([bertModel.sep_token_id])]
            first_token = torch.tensor([0] * (len(com_id_filter) + 2))
            second_token = torch.tensor([1] * (len(text_id_filter) + 1))
            token_id = torch.cat([first_token, second_token])
        res.append(torch.cat(dt_list))
        token_type.append(token_id)
    text_ids = pad_sequence(res, batch_first=True, padding_value=0)[:, :511]
    token_ids = pad_sequence(token_type, batch_first=True, padding_value=1)[:, :511]
    return text_ids, token_ids


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def constructEmbNode(data_sets, ntype, encoder, device, args):
    text_feat = torch.rand(data_sets.virtual_info[ntype].shape[0], 768)
    dataloader = DataLoader(data_sets.total_nodes[ntype], batch_size=args["batch_size"], shuffle=False)
    for nodes in tqdm(dataloader):  # 专利的表征
        feat_inputs = data_sets.getTextInGraph(ntype, nodes)[..., :280].to(device)
        token_type_ids = torch.zeros(feat_inputs.shape, dtype=torch.long).to(device)
        attn_mask = feat_inputs.new_zeros(feat_inputs.shape, dtype=torch.long).to(device)
        attn_mask[feat_inputs != 0] = 1
        lengths = attn_mask.sum(dim=-1).to(device)
        outputs = encoder(input_ids=feat_inputs,
                          token_type_ids=token_type_ids,
                          attention_mask=attn_mask,
                          return_dict=True)
        word_emb = outputs.last_hidden_state
        aver_feats = torch.sum(word_emb, dim=1) / lengths.unsqueeze(dim=1)
        text_feat[nodes] = aver_feats.cpu()
    return text_feat


if __name__ == "__main__":
    args = {
        'cuda': 3,
        'embedding_dim': 128,
        'seed': 3,
        "batch_size": 4,
        'graph_path': "../../Data/Patents/com_patents-graph.pkl",
        "abstract_path": "../../Data/Patents/abstract.csv",
        "title_path": "../../Data/Patents/title.csv",
        "patent_path": "../../Data/Patents/patent-filter.csv",
        "bert_tokenizer": "../../Data/Bert-base-cased/",
        "embedding_path": "None",
        "company_path": "../../Data/Patents/Assignees-token.csv",
        "applicant_path": "../../Data/Patents/Applicants-token.csv",
        "company_filter_path": "../../Data/Patents/company-filter.csv",
        "applicant_filter_path": "../../Data/Patents/applicant-filter.csv",
        "pre_path": "../../save_models/THLM.pth"
    }

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
    torch.cuda.set_device('cuda:{}'.format(args["cuda"]))
    torch.set_num_threads(1)
    setup_seed(args['seed'])
    # os.makedirs('../../Data/ogbn_mag/', exist_ok=True)

    encoder = BertModel.from_pretrained(args["bert_tokenizer"]).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()

    if not args["pre_path"].startswith("None"):
        reload_model(encoder, args['pre_path'])

    data_sets = MagDataset(graph_path=args['graph_path'], abstract_path=args['abstract_path'],
                           title_path=args['title_path'], bert_tokenizer=args['bert_tokenizer'],
                           embedding_path=args['embedding_path'], patent_path=args["patent_path"],
                           company_path=args["company_path"], applicant_path=args["applicant_path"],
                           company_filter=args["company_filter_path"], applicant_filter=args["applicant_filter_path"],
                           )
    pat_feat = torch.zeros(data_sets.virtual_info["patent"].shape[0], 768)
    text_length = torch.zeros(data_sets.virtual_info["patent"].shape[0], dtype=torch.long)
    # text_feat[..., :128] = data_sets.embedding_dict['patent'][..., :128]
    if args['pretrain']:
        pat_feat = constructEmbNode(data_sets, ntype='patent', encoder=encoder, device=device, args=args)
        com_feat = constructEmbNode(data_sets, ntype='company', encoder=encoder, device=device, args=args)
        applicant_feat = constructEmbNode(data_sets, ntype='applicant', encoder=encoder, device=device, args=args)
    else:
        dataloader = DataLoader(data_sets.total_nodes['patent'], batch_size=args["batch_size"], shuffle=False)
        for nodes in tqdm(dataloader):  # 专利的表征
            feat_inputs = data_sets.getTextInGraph('patent', nodes)[..., :280].to(device)
            token_type_ids = torch.zeros(feat_inputs.shape, dtype=torch.long).to(device)
            attn_mask = feat_inputs.new_zeros(feat_inputs.shape, dtype=torch.long).to(device)
            attn_mask[feat_inputs != 0] = 1
            lengths = attn_mask.sum(dim=-1).to(device)
            outputs = encoder(input_ids=feat_inputs,
                              token_type_ids=token_type_ids,
                              attention_mask=attn_mask,
                              return_dict=True)
            word_emb = outputs.last_hidden_state
            aver_feats = torch.sum(word_emb, dim=1) / lengths.unsqueeze(dim=1)
            pat_feat[nodes] = aver_feats.cpu()
            text_length[nodes] = lengths.cpu()

        # 给出公司的表征
        com_feat = torch.zeros(data_sets.virtual_info["company"].shape[0], 768)
        edges = data_sets.graph.edges(etype='company_patent')
        edges = torch.stack(edges, dim=1).numpy().tolist()
        dataloader = DataLoader(edges, batch_size=args["batch_size"], shuffle=False)
        for nodes in tqdm(dataloader):  # 遍历公司对
            feat_com = data_sets.getTextInGraph('company', nodes[0])[..., :100]
            feat_text = data_sets.getTextInGraph('patent', nodes[1])[..., :280]
            embed_input, type_ids = concateData([feat_com, feat_text], data_sets.Bert_tokenizer)
            attn_mask = embed_input.new_zeros(embed_input.shape, dtype=torch.long)
            attn_mask[embed_input != 0] = 1
            lengths = attn_mask.sum(dim=-1)
            outputs = encoder(input_ids=embed_input.to(device),
                              token_type_ids=type_ids.to(device),
                              attention_mask=attn_mask.to(device),
                              return_dict=True)
            word_emb = outputs.last_hidden_state.cpu()
            aver_feats = torch.sum(word_emb, dim=1) / lengths.unsqueeze(dim=1)
            com_feat[nodes[0]] = com_feat[nodes[0]] + aver_feats.cpu()
        out_degree = data_sets.graph.out_degrees(range(com_feat.shape[0]), etype='company_patent')
        com_feat = com_feat / out_degree.unsqueeze(dim=1)

        # 给出作者的表征
        applicant_feat = torch.zeros(data_sets.virtual_info["applicant"].shape[0], 768)
        edges = data_sets.graph.edges(etype='applicant_patent')
        edges = torch.stack(edges, dim=1).numpy().tolist()
        dataloader = DataLoader(edges, batch_size=args["batch_size"], shuffle=False)
        for nodes in tqdm(dataloader):  # 遍历公司对
            feat_com = data_sets.getTextInGraph('applicant', nodes[0])[..., :100]
            feat_text = data_sets.getTextInGraph('patent', nodes[1])[..., :280]
            embed_input, type_ids = concateData([feat_com, feat_text], data_sets.Bert_tokenizer)
            attn_mask = embed_input.new_zeros(embed_input.shape, dtype=torch.long)
            attn_mask[embed_input != 0] = 1
            lengths = attn_mask.sum(dim=-1)
            outputs = encoder(input_ids=embed_input.to(device),
                              token_type_ids=type_ids.to(device),
                              attention_mask=attn_mask.to(device),
                              return_dict=True)
            word_emb = outputs.last_hidden_state.cpu()
            aver_feats = torch.sum(word_emb, dim=1) / lengths.unsqueeze(dim=1)

            applicant_feat[nodes[0]] = applicant_feat[nodes[0]] + aver_feats.cpu()
        out_degree = data_sets.graph.out_degrees(range(applicant_feat.shape[0]), etype='applicant_patent')
        applicant_feat = applicant_feat / out_degree.unsqueeze(dim=1)

    data_sets.rebuildEmbedding({"patent": pat_feat, "company": com_feat, "applicant": applicant_feat})
    data_sets.reconstructEmbed({"patent": pat_feat, "company": com_feat, "applicant": applicant_feat})

    graph_output_path = '../../Data/Patents/model/bert_THLM.pkl'
    save_graphs(graph_output_path, data_sets.new_graph, data_sets.labels)

    print("=========================finish================================")
