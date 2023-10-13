import sys

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


class MagDataset:
    def __init__(self, graph_path, title_path, abstract_path, field_path, author_path, affiliation_path,
                 bert_tokenizer):
        self.data_dict, self.total_nodes = {}, {}
        self.data_dict["title"], self.total_nodes["title"] = self.load_dict(title_path)  # 读取paper信息
        self.data_dict["paper"], self.total_nodes["paper"] = self.load_dict(abstract_path)  # 读取paper信息
        self.data_dict["field_of_study"], self.total_nodes["field_of_study"] = self.load_dict(field_path)  # 读取paper信息
        self.data_dict["author"], self.total_nodes["author"] = self.load_dict(author_path)  # 读取paper信息
        self.data_dict["institution"], self.total_nodes["institution"] = self.load_dict(affiliation_path)  # 读取paper信息
        self.load_bertTokenizer(bert_tokenizer)
        self.graph = self.load_graph(graph_path)
        self.map_reIndex()

    def map_reIndex(self):
        self.types = ["paper", "field_of_study", "author", "institution"]
        token_list = ["title"] + self.types
        self.virtual_dict = {}
        self.raw_node, self.real_node = {}, {}
        for type in token_list:
            virtual_num = self.graph.nodes(type) if type != 'title' else self.graph.nodes('paper')
            data = pd.DataFrame({"ent idx": virtual_num.numpy().tolist()})
            merge_data = pd.merge(data, self.data_dict[type], how='left').sort_values(by='ent idx', ascending=True)
            self.real_node[type] = merge_data.drop_duplicates().dropna()["ent idx"].values.tolist()
            self.raw_node[type] = len(merge_data)
            print(type, len(merge_data), len(self.real_node[type]))
            texts = []
            for x in tqdm(merge_data.values[:, -1]):
                if pd.isnull(x):
                    texts.append(torch.tensor([self.cls_id]))
                    continue
                x_ = str(x).replace(" ", "")[1:-1].split(",")
                texts.append(torch.tensor([int(i) for i in x_]))
            # texts = [[int(i) for i in x[2:-2].split(",")] for x in merge_data.values[:, -1]]
            self.virtual_dict[type] = pad_sequence(texts, batch_first=True, padding_value=0).long()

    def load_dict(self, text_path):
        data_dict = pd.read_csv(text_path).dropna().drop_duplicates()  # paper的id信息和paper的摘要信息 N*2
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

    def reconstructEmbed(self, new_feat: dict):
        original_graph = self.graph
        data_dict = {}
        for src_type, etype, dst_type in original_graph.canonical_etypes:
            src, dst = original_graph.edges(etype=etype)
            data_dict[(src_type, etype, dst_type)] = (src, dst)

        self.new_graph = dgl.heterograph(data_dict=data_dict,
                                         num_nodes_dict={ntype: original_graph.number_of_nodes(ntype) for ntype in
                                                         original_graph.ntypes})

        self.new_graph.nodes['paper'].data['year'] = original_graph.nodes['paper'].data['year']
        # concat the content and structural feature of paper
        for ntype, nfeat in new_feat.items():
            self.new_graph.nodes[ntype].data['feat'] = nfeat
    def getTextInGraph(self, type, ids):  # 取出对应的text信息,这里面text是不对齐的
        return self.virtual_dict[type][ids]

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


def concateData(nodes, datas: list, total_num, bertModel: transformers.BertTokenizer):
    res = []
    token_type = []
    exists = torch.zeros(total_num)
    select_id = torch.zeros(len(datas[0]))
    for i, (com_ids, text_ids) in enumerate(zip(datas[0], datas[1])):
        com_id_filter = com_ids[com_ids != bertModel.pad_token_id][1:-1]
        text_id_filter = text_ids[text_ids != bertModel.pad_token_id][1:-1]
        if len(text_id_filter) == 0 and len(com_id_filter) == 0:
            continue
        elif len(com_id_filter) == 0:
            dt_list = [torch.tensor([bertModel.cls_token_id]), text_id_filter, torch.tensor([bertModel.sep_token_id])]
            token_id = torch.tensor([0] * (len(text_id_filter) + 2))
        elif len(text_id_filter) == 0:
            dt_list = [torch.tensor([bertModel.cls_token_id]), com_id_filter, torch.tensor([bertModel.sep_token_id])]
            token_id = torch.tensor([0] * (len(com_id_filter) + 2))
        else:
            dt_list = [torch.tensor([bertModel.cls_token_id]), com_id_filter, torch.tensor([bertModel.sep_token_id]),
                       text_id_filter, torch.tensor([bertModel.sep_token_id])]
            first_token = torch.tensor([0] * (len(com_id_filter) + 2))
            second_token = torch.tensor([1] * (len(text_id_filter) + 1))
            token_id = torch.cat([first_token, second_token])
        exists[nodes[i]] = exists[nodes[i]] + 1

        select_id[i] = 1
        res.append(torch.cat(dt_list))
        token_type.append(token_id)
    if len(res) == 0:
        return torch.tensor([]), None, None, None
    text_ids = pad_sequence(res, batch_first=True, padding_value=0)[:, :511]
    token_ids = pad_sequence(token_type, batch_first=True, padding_value=1)[:, :511]
    return text_ids, token_ids, exists, select_id


def BertEmbedding(data_sets: MagDataset, text_feat: dict, encoder: transformers.BertModel, args: dict,
                  ntype, etype, pre_type, beh_type):

    edges = data_sets.graph.edges(etype=etype)  # (author, paper)
    print(pre_type, etype, beh_type)
    edges = torch.stack(edges, dim=1).numpy().tolist()
    # if ntype == 'field_of_study':
    #     edges = edges[edges[0] != 14055]
    dataloader = DataLoader(edges, batch_size=args["bs"], shuffle=False)
    empties = torch.zeros(data_sets.raw_node[ntype])
    new_Embed = text_feat[ntype].new_zeros(text_feat[ntype].shape)
    for nodes in tqdm(dataloader):  # 遍历 paper-author
        feat_author = data_sets.getTextInGraph(pre_type, nodes[0])[..., :30]
        feat_paper = data_sets.getTextInGraph(beh_type, nodes[1])[..., :300]
        embed_input, type_ids, exists, select_ids = concateData(nodes[0], [feat_author, feat_paper],
                                                                data_sets.raw_node[ntype],
                                                                data_sets.Bert_tokenizer)
        if len(embed_input) == 0:
            continue
        empties = empties + exists
        # print(select_ids.shape, embed_input.shape)
        # embed_input = embed_input[select_ids == 1, ...]
        attn_mask = embed_input.new_zeros(embed_input.shape, dtype=torch.long)
        attn_mask[embed_input != 0] = 1
        lengths = attn_mask.sum(dim=-1)
        outputs = encoder(input_ids=embed_input.to(device),
                          token_type_ids=type_ids.to(device),
                          attention_mask=attn_mask.to(device),
                          return_dict=True)
        word_emb = outputs.last_hidden_state.cpu()
        aver_feats = torch.sum(word_emb, dim=1) / lengths.unsqueeze(dim=1)
        new_Embed[nodes[0][select_ids == 1]] = new_Embed[nodes[0][select_ids == 1]] + aver_feats.cpu()
        # break
    print(f"{ntype}, {empties}")
    # out_degree = data_sets.graph.out_degrees(range(data_sets.raw_node[ntype]), etype=etype) - empties
    indexId = empties != 0
    text_feat[ntype][indexId] = new_Embed[indexId] / empties[indexId].unsqueeze(dim=1)


def sampleNei(graph: dgl.DGLHeteroGraph, ntype: dict, etypes: list, nei_num: int) -> dict:
    neigh_layer_1 = dgl.sampling.sample_neighbors(graph, ntype, nei_num, edge_dir='out')
    neigh_nodes = {dtype: [] for etype, dtype in etypes}
    for etype_tuple in etypes:
        etype, dsttype = etype_tuple
        neigh_nodes[dsttype].append(neigh_layer_1.edges(etype=etype)[1].unique())  # 一阶邻居：公司
    for ntype, nodes in neigh_nodes.items():
        neigh_nodes[ntype] = torch.cat(nodes).long()
    return neigh_nodes


def concateEntities(entities: list, data_sets: MagDataset):
    new_entities = [torch.tensor([data_sets.cls_id])]
    count = 0
    for i in range(len(entities)):
        entity = entities[i] if isinstance(entities[i], torch.Tensor) else torch.tensor(entities[i])
        real_entity = entity[entity != 0][1:-1].clone()
        if len(real_entity) == 0:
            continue
        elif count + len(real_entity) + 2 > 510:
            less = 509 - count
            new_entities.append(real_entity[:less])
            break
        new_entities.append(real_entity)
        count += len(real_entity)
    new_entities.append(torch.tensor([data_sets.sep_id]))
    return torch.cat(new_entities, dim=0)


def EmbedCenterNode(text_feat, ntype, data_sets: MagDataset, graph, args):
    dataloader = DataLoader(data_sets.real_node[ntype], batch_size=args["bs"], shuffle=False)
    etypes = {ntype: [] for ntype in graph.ntypes}
    for stype, etype, dtype in graph.canonical_etypes:
        etypes[stype].append((etype, dtype))
    for nodes in tqdm(dataloader):
        feat_inputs = []
        for node in nodes:
            ntype_input = [data_sets.getTextInGraph(ntype, node)[..., :450]]
            node_neighs = sampleNei(graph, {ntype: node}, etypes[ntype], nei_num=-1)
            for neigh_ntype, neigh_nodes in node_neighs.items():
                # print(neigh_ntype, neigh_nodes)
                if neigh_ntype == ntype or len(neigh_nodes) == 0:
                    continue
                neigh_input = data_sets.getTextInGraph(neigh_ntype, neigh_nodes)[..., :350].numpy().tolist()
                ntype_input.extend(neigh_input)
                # print(len(ntype_input))
            feat_input = concateEntities(ntype_input, data_sets)
            feat_inputs.append(feat_input)
        feat_inputs = pad_sequence(feat_inputs, batch_first=True).to(device)
        # feat_inputs = data_sets.getTextInGraph(ntype, nodes)[..., :350].to(device)
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
        # total_lens += sum(lengths)
        text_feat[ntype][nodes] = aver_feats.cpu()
        # break
    # print(total_lens/len(data_sets.real_node[ntype]))


def EmbedNode(text_feat, ntype, data_sets: MagDataset, args):
    dataloader = DataLoader(data_sets.real_node[ntype], batch_size=args["bs"], shuffle=False)
    for nodes in tqdm(dataloader):
        feat_inputs = data_sets.getTextInGraph(ntype, nodes)[..., :350].to(device)
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
        # total_lens += sum(lengths)
        text_feat[ntype][nodes] = aver_feats.cpu()
        # break
    # print(total_lens/len(data_sets.real_node[ntype]))


if __name__ == "__main__":
    save_path = '../../Data/OAG_Venue/'
    graph_output_path = '../../Data/OAG/model/bert_THLM.pkl'
    args = {
        'cuda': 1,
        'embedding_dim': 128,
        "bs": 128,
        'graph_path': "../../Data/OAG_Venue/Downstream/OAG_CS_Venue_Text.pk",
        "abstract_path": f"{save_path}/paper2token.csv",
        "title_path": f"{save_path}/title2token.csv",
        "field_path": f"{save_path}/field2token.csv",
        "author_path": f"{save_path}/author2token.csv",
        "affiliation_path": f"{save_path}/affiliation2token.csv",
        "bert_tokenizer": "../../Data/Bert-base-cased/",
        "pre_path": "../../save_models/THLM.pth"
    }

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'
    torch.cuda.set_device('cuda:{}'.format(args["cuda"]))
    torch.set_num_threads(1)

    encoder = BertModel.from_pretrained(args["bert_tokenizer"]).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()

    if not args["pre_path"].startswith("None"):
        reload_model(encoder, args['pre_path'])

    data_sets = MagDataset(graph_path=args['graph_path'], abstract_path=args['abstract_path'],
                           field_path=args['field_path'], author_path=args["author_path"],
                           title_path=args["title_path"],
                           affiliation_path=args["affiliation_path"],
                           bert_tokenizer=args['bert_tokenizer'])
    text_feat = {ntype: torch.zeros(data_sets.virtual_dict[ntype].shape[0], 768) for ntype in data_sets.types}
    # for ntype in data_sets.types:
    #     text_feat[ntype][..., :128] = data_sets.graph.nodes[ntype].data['meta_feat'][..., :128]

    EmbedNode(text_feat, "paper", data_sets, args)
    # EmbedCenterNode(text_feat, "paper", data_sets, data_sets.graph, args)
    print("====================save paper embedding====================")
    print(graph_output_path)
    print(args["pre_path"])

    data_sets.reconstructEmbed(text_feat)
    save_graphs(graph_output_path, data_sets.new_graph, data_sets.labels)


    BertEmbedding(data_sets, text_feat, encoder, args,
                  ntype="field_of_study", etype="rev_has_topic", pre_type="field_of_study", beh_type="paper")
    data_sets.reconstructEmbed(text_feat)
    save_graphs(graph_output_path, data_sets.new_graph, data_sets.labels)
    print("====================save field embedding====================")

    BertEmbedding(data_sets, text_feat, encoder, args,
                  ntype="institution", etype="rev_affiliated_with", pre_type="institution", beh_type="author")

    data_sets.reconstructEmbed(text_feat)
    save_graphs(graph_output_path, data_sets.new_graph, data_sets.labels)
    print("====================save institution embedding====================")

    BertEmbedding(data_sets, text_feat, encoder, args,
                  ntype="author", etype="writes", pre_type="author", beh_type="paper")
    data_sets.reconstructEmbed(text_feat)
    save_graphs(graph_output_path, data_sets.new_graph, data_sets.labels)
    print("====================save author embedding====================")


    sys.exit(0)
