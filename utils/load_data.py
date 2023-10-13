import random

import dgl
import numpy as np

import torch.nn as nn
import pandas as pd

import torch
from dgl import load_graphs

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorForLanguageModeling, RobertaTokenizer

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
             'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
             'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
             'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
             's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm',
             'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
             'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won',
             'wouldn', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '[pad]',
             '[sep]', '[cls]']


class FeatureDataSet(Dataset):
    def __init__(self, src_ids: list, idtypes: list):
        self.src = src_ids
        self.stype = idtypes

    def __getitem__(self, item):
        return self.src[item], self.stype[item]

    def __len__(self):
        return len(self.src)


class Data(object):
    def __init__(self, graph_path, abstract_path, title_path, company_path, applicant_path,
                 filter_patent, filter_company, filter_applicant, sample_title,
                 bert_tokenizer, batch_size, has_neighbor,
                 has_mask, sample_graph_num=5, pred_layer=2):
        self.batch_size = batch_size
        self.sample_title_num = sample_title
        self.types = ["patent", "title", "applicant", "company"]
        self.randrate, self.realmkrate, self.maskrate = 0.1, 0.8, 0.4
        self.pred_layer = pred_layer
        self.has_msk, self.has_neighbor = has_mask, has_neighbor
        self.data_dict, self.total_nodes = {}, {}
        self.data_dict["patent"] = self.load_dict(abstract_path)
        self.data_dict["title"] = self.load_dict(title_path)
        self.data_dict["applicant"] = self.load_dict(applicant_path)
        self.data_dict["company"] = self.load_dict(company_path)
        self.graph = self.load_graph(graph_path)
        self.load_bertTokenizer(bert_tokenizer)

        self.map_reIndex([filter_patent, filter_patent, filter_applicant, filter_company])
        self.load_dataset(batch_size, sample_graph_num)

    def load_dict(self, text_path):
        data_dict = pd.read_csv(text_path).dropna().drop_duplicates()
        return data_dict

    def map_reIndex(self, filter_list):
        self.node_num = {}
        self.idlist = {}
        self.virtual_dict = {}
        for i, type in enumerate(self.types):
            data = pd.read_csv(filter_list[i])
            merge_data = pd.merge(data, self.data_dict[type], how='left').sort_values(by=data.columns[-1],
                                                                                      ascending=True)
            if type != 'title':
                self.node_num[type] = len(data)
            self.total_nodes[type] = merge_data.dropna().values[:, 1]
            self.idlist[type] = np.array(range(len(data)))

            texts = []
            for x in tqdm(merge_data.values[:, -1]):
                if pd.isnull(x):
                    texts.append(torch.tensor([self.cls_id]))
                    continue
                x_ = str(x).replace(" ", "")[1:-1].split(",")
                texts.append(torch.cat([self.preFix[type], torch.tensor([int(i) for i in x_][1:-1])]))
            self.virtual_dict[type] = pad_sequence(texts, batch_first=True, padding_value=0).long()

    def load_dataset(self, batch_size, sample_graph_num):
        total_num, stype = [], []
        for ntype in self.etypes.keys():
            total_num.extend(self.total_nodes[ntype])
            stype.extend([ntype] * len(self.total_nodes[ntype]))
        self.total_num = len(total_num)
        self.dataloader = DataLoader(FeatureDataSet(src_ids=total_num, idtypes=stype), batch_size=batch_size,
                                     shuffle=True)
        print("training process, number of nodes:{}".format(self.total_num))

    def load_graph(self, graph_path):
        graph = load_graphs(graph_path)[0][0]
        self.etypes = {ntype: [] for ntype in graph.ntypes}
        self.embedding = dict()
        for ntype in graph.ntypes:
            if "feat" in graph.nodes[ntype].data:
                self.embedding[ntype] = graph.nodes[ntype].data["feat"]

        self.node_num = {}
        self.nodeEmb = nn.ModuleDict()
        for ntype in graph.ntypes:
            self.node_num[ntype] = len(graph.nodes(ntype))
            nor_feat = graph.nodes[ntype].data["feat"]

            self.nodeEmb[ntype] = nn.Embedding.from_pretrained(
                torch.cat([nor_feat, torch.zeros(1, nor_feat.shape[1])], dim=0), freeze=True)

        for stype, etype, dtype in graph.canonical_etypes:
            self.etypes[stype].append((etype, dtype))
        return graph

    def load_bertTokenizer(self, bert_tokenizer):
        self.Bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.maskcollator = DataCollatorForLanguageModeling(self.Bert_tokenizer, mlm=self.has_msk,
                                                            mlm_probability=0.4)
        self.preFix = dict()
        for ntype in self.types:
            ntype_Emb = \
                self.Bert_tokenizer("The " + ntype + " is:", padding=True, truncation=False, return_tensors="pt")[
                    'input_ids'].squeeze(dim=0)[:-1]
            self.preFix[ntype] = ntype_Emb
        self.total_vocab = self.Bert_tokenizer.vocab_size
        self.cls_id, self.mask_id = self.Bert_tokenizer.cls_token_id, self.Bert_tokenizer.mask_token_id
        self.pad_id, self.sep_id = self.Bert_tokenizer.pad_token_id, self.Bert_tokenizer.sep_token_id

    def sample_title(self, stype, sids, sample_num=5):
        res = []
        second_hops = self.sampleNei({stype: sids}, self.etypes[stype], sample_num)
        neighbors = second_hops["patent"].numpy().tolist()
        for sam_id in neighbors:
            if sam_id not in self.total_nodes["title"]:
                continue
            else:
                res.append(sam_id)
        return res

    def generate_PosNeg(self, dtype, pos_ids, neg_ids, neg_ratio=5):
        if len(pos_ids) != 0:
            pos_patents = pos_ids.unique().clone()
            flag = np.zeros(self.node_num[dtype], dtype=np.long)
            flag[neg_ids] = 1
            flag[pos_ids] = 2
            other_ids = self.idlist[dtype][flag == 0]
            neg_ids = torch.from_numpy(self.idlist[dtype][flag == 1])

            select_ids = np.unique(np.random.randint(low=0, high=len(other_ids),
                                                     size=min(len(pos_patents) * neg_ratio, len(other_ids)))).tolist()
            res = torch.tensor(other_ids[select_ids])
            neg_len = len(neg_ids) + len(res)
        else:
            pos_patents = torch.tensor([])
            res = torch.tensor([1])
            neg_ids = torch.tensor([])
            neg_len = len(res) + len(neg_ids)
        id_all = torch.cat([neg_ids, res, pos_patents]).long()
        truth_i = torch.tensor([0.0] * neg_len + [1.0] * len(pos_patents))
        mask_i = torch.zeros(len(truth_i)) if len(pos_ids) == 0 else torch.ones(len(truth_i))
        return id_all, truth_i, mask_i

    def sampleNei(self, ntype: dict, etypes: list, nei_num: int) -> dict:
        neigh_layer_1 = dgl.sampling.sample_neighbors(self.graph, ntype, nei_num, edge_dir='out')
        neigh_nodes = {dtype: [] for etype, dtype in etypes}
        for etype_tuple in etypes:
            etype, dsttype = etype_tuple
            neigh_nodes[dsttype].append(neigh_layer_1.edges(etype=etype)[1].unique())
        for ntype, nodes in neigh_nodes.items():
            neigh_nodes[ntype] = torch.cat(nodes).long()
        return neigh_nodes

    def return_neiLabels(self, input_ids: [torch.Tensor, str], nei_num=5, mask_ratio=3):
        layers = self.pred_layer
        return_dict = [{ntype: [] for ntype in self.etypes.keys()} for i in range(3)]
        for nid, ntype in zip(input_ids[0], input_ids[1]):
            preNeis = {ntype: torch.tensor([nid]).long()}
            high_Neis_dict = {ntype: [torch.tensor([nid])]}
            for i in range(0, layers):
                i_Nei = dict()
                for stype, sids in preNeis.items():
                    if len(sids) != 0:
                        second_hops = self.sampleNei({stype: sids}, self.etypes[stype], nei_num)
                        for dtype, dids in second_hops.items():
                            if len(dids) == 0:
                                continue
                            if dtype not in i_Nei:
                                i_Nei[dtype] = []
                            i_Nei[dtype].append(dids.unique())

                for stype, sids in i_Nei.items():
                    i_Nei[stype] = torch.cat(sids).unique()
                    if stype not in high_Neis_dict:
                        high_Neis_dict[stype] = []
                    high_Neis_dict[stype].append(i_Nei[stype])

                preNeis = i_Nei

            for stype in self.etypes.keys():
                if stype in high_Neis_dict:
                    pos_ids = torch.cat(high_Neis_dict[stype]).unique().long()
                else:
                    pos_ids = torch.tensor([])
                id_pairs, truth, masks = self.generate_PosNeg(dtype=stype,
                                                              pos_ids=pos_ids,
                                                              neg_ids=torch.tensor([]))

                return_dict[0][stype].append(id_pairs)
                return_dict[1][stype].append(truth)
                return_dict[2][stype].append(masks)

        new_dict = []
        for i, dict_lists in enumerate(return_dict):
            new_dict.append(dict())
            for ntype, nlist in dict_lists.items():
                if len(nlist) == 0:
                    continue
                new_dict[i][ntype] = pad_sequence(nlist, batch_first=True, padding_value=0)

        return new_dict

    def getTextInGraph(self, type, ids):
        return self.virtual_dict[type][ids]

    def maskSentences(self, texts):
        """
        :param texts: [paper,author,affiliation] / [paper]
        :return: raw_text: torch.size([k, F]), mask_text: torch.size([k, F]), len_i: [l_1,...l_k]
        """
        raw_texts, mask_texts, len_i = [], [], []

        for i in range(len(texts)):
            text_i = texts[i][texts[i] != self.pad_id][1:]
            assert len(text_i) > 0
            raw_texts.append(text_i.clone())
            len_i.append(len(text_i))
            mask_text = text_i
            mask_texts.append(mask_text)
            assert text_i.shape == mask_text.shape

        max_len = 509
        cut_num = len(texts) + 2
        if sum(len_i) > (max_len - cut_num):
            total_len = sum(len_i)
            more_len = (total_len - (max_len - cut_num))
            new_raw, new_mask, new_len = [], [], []
            if more_len == 0:
                more_len = 1
            for raw_text, mask_text, len_t in zip(raw_texts, mask_texts, len_i):
                end_len = int(len(raw_text) - more_len * (len(raw_text) / total_len))
                new_raw.append(raw_text[:end_len])
                new_mask.append(mask_text[:end_len])
                new_len.append(end_len)
        else:
            new_raw, new_mask, new_len = raw_texts, mask_texts, len_i
        assert sum(new_len) <= (max_len - cut_num)
        return new_raw, new_mask, new_len

    def random_word(self, tokens):
        sentence = self.Bert_tokenizer.convert_ids_to_tokens(tokens)
        for i, token in enumerate(tokens):
            if sentence[i].lower() in stopwords:
                # print("stopword " + sentence[i])
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < self.maskrate:
                prob /= self.maskrate

                # 80% randomly change token to mask token
                if prob < self.realmkrate:
                    tokens[i] = self.mask_id

                # 10% randomly change token to random token
                elif prob < self.realmkrate + self.randrate:
                    tokens[i] = random.choice(list(range(self.Bert_tokenizer.vocab_size)))

                # -> rest 10% randomly keep current token

        return tokens

    def generateSen_ids(self, len_id, keepSame=True):
        total_tokens = [torch.tensor([0])]
        pre_sen_token = torch.tensor([0] * len_id[0])
        total_tokens.append(pre_sen_token)
        total_tokens.append(torch.tensor([0]))
        for i in np.arange(1, len(len_id)):
            aut_sen_token = torch.tensor([0 if keepSame else 1] * len_id[i])
            total_tokens.append(aut_sen_token)
            total_tokens.append(torch.tensor([0 if keepSame else 1]))
        token_type_ids = torch.cat(total_tokens, dim=0)
        return token_type_ids

    def connectSentences(self, sentence_pairs: [torch.Tensor]):
        """
        :param sentence_pairs: torch.size([tensor,...,tensor]])
        :return: total_text: torch.size([N*(K+1)+1])
        """
        total_text = [torch.tensor([self.cls_id])]
        for i, sentence in enumerate(sentence_pairs):
            if i > 0:
                total_text.append(torch.tensor([self.sep_id]))
            total_text.append(sentence)
        total_text.append(torch.tensor([self.sep_id]))
        return torch.cat(total_text)

    def MaskToken(self, rawids: [torch.Tensor, str]):
        raw_texts, mask_texts, token_ids = [], [], []
        labels = None
        texts = []
        new_ids = [[], []]
        for sid, stype in zip(rawids[0], rawids[1]):
            if stype == 'patent':
                text_i = [self.getTextInGraph(type=stype, ids=sid)]
            else:
                text_i = []
            if stype != "patent":
                neiPats = self.sample_title(stype, sid, self.sample_title_num)
                for nei_pat in neiPats:
                    text_i.append(self.getTextInGraph(type="title", ids=nei_pat))  # torch.size([k, F])

            if len(text_i) == 0:
                continue
            new_ids[0].append(sid)
            new_ids[1].append(stype)
            texts.append(text_i)

        for pid in range(len(texts)):
            raw_text, mask_text, len_i = self.maskSentences(texts[pid])
            token_id = self.generateSen_ids(len_i)
            token_ids.append(token_id)
            connect_raw = self.connectSentences(raw_text)
            connect_mask = self.connectSentences(mask_text)
            assert len(connect_raw) == len(token_id)

            raw_texts.append(connect_raw)
            mask_texts.append(connect_mask)

        output_mask = self.maskcollator(raw_texts)
        mask_texts, label_texts = output_mask['input_ids'], output_mask['labels']
        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        #
        assert mask_texts.shape == token_ids.shape
        #
        flatten_mask = label_texts.flatten()
        total_len = torch.tensor(range(len(flatten_mask)))
        mask_ids = total_len[flatten_mask != -100]

        return label_texts, mask_texts, mask_ids, token_ids, labels, new_ids


class Data_OAG(object):
    def __init__(self, paper_path, author_path, affiliation_path, title_path, field_path, bert_tokenizer, batch_size,
                 has_mask, has_neighbor, sample_graph_num=5,
                 ):
        self.batch_size = batch_size
        self.has_msk, self.has_neighbor = has_mask, has_neighbor
        self.randrate, self.realmkrate, self.maskrate = 0.1, 0.8, 0.15
        self.data_dict, self.total_nodes = {}, {}
        self.types = ["paper", "author", "title", "field_of_study", "institution"]
        self.data_dict["paper"] = self.load_dict(paper_path)
        self.data_dict["title"] = self.load_dict(title_path)
        self.data_dict["author"] = self.load_dict(author_path)
        self.data_dict["field_of_study"] = self.load_dict(field_path)
        self.data_dict["institution"] = self.load_dict(affiliation_path)
        self.load_bertTokenizer(bert_tokenizer)
        self.graph = self.load_graph()
        self.map_reIndex()
        self.load_dataset(batch_size)

    def load_dict(self, text_path):
        data_dict = pd.read_csv(text_path).dropna().drop_duplicates()
        return data_dict

    def map_reIndex(self):
        self.idlist = {}
        self.virtual_dict, self.raw_vir_dict = {}, {}
        for type in self.types:
            if type == "title":
                virtual_num = self.graph.nodes("paper")
            else:
                virtual_num = self.graph.nodes(type)
            data = pd.DataFrame({"ent idx": virtual_num.numpy().tolist()})
            merge_data = pd.merge(data, self.data_dict[type], how='left').sort_values(by='ent idx', ascending=True)

            self.total_nodes[type] = torch.tensor(merge_data.dropna()['ent idx'].values.tolist())  # 查看情况
            self.idlist[type] = np.array(range(len(data)))
            print(type, f", columns are {merge_data.columns}, "
                        f"the number is:", len(self.total_nodes[type]))

            raw_texts, texts = [], []
            for x in tqdm(merge_data.values[:, -1]):
                if pd.isnull(x):
                    texts.append(torch.tensor([self.cls_id]))
                    raw_texts.append(torch.tensor([self.cls_id]))
                    continue
                x_ = str(x).replace(" ", "")[1:-1].split(",")
                x_ts = torch.tensor([int(i) for i in x_])
                texts.append(torch.cat([self.preFix[type][:-1], x_ts[1:]]))
                raw_texts.append(x_ts)
            self.virtual_dict[type] = pad_sequence(texts, batch_first=True, padding_value=0).long()
            self.raw_vir_dict[type] = pad_sequence(raw_texts, batch_first=True, padding_value=0).long()
            print(len(self.virtual_dict[type]), len(self.raw_vir_dict[type]))

    def load_dataset(self, batch_size):
        total_num, stype = [], []
        for ntype in ["paper"]:
            total_num.extend(self.total_nodes[ntype])
            stype.extend([ntype] * len(self.total_nodes[ntype]))
        self.total_num = len(total_num)
        self.dataloader = DataLoader(FeatureDataSet(src_ids=total_num, idtypes=stype), batch_size=batch_size,
                                     shuffle=True)
        print("training process, number of nodes:{}".format(self.total_num))

    def load_graph(self):
        self.dataset = dgl.load_graphs("../Data/OAG_Venue/model/bert_raw.pkl")
        graph = self.dataset[0][0]
        self.etypes = {ntype: [] for ntype in graph.ntypes}
        self.embedding = dict()
        for ntype in graph.ntypes:
            if "feat" in graph.nodes[ntype].data:
                self.embedding[ntype] = graph.nodes[ntype].data["feat"]

        self.node_num = {}
        self.nodeEmb = nn.ModuleDict()
        for ntype in graph.ntypes:
            self.node_num[ntype] = len(graph.nodes(ntype))
            nor_feat = graph.nodes[ntype].data["feat"]

            self.nodeEmb[ntype] = nn.Embedding.from_pretrained(
                torch.cat([nor_feat, torch.zeros(1, nor_feat.shape[1])], dim=0), freeze=True)

        for stype, etype, dtype in graph.canonical_etypes:
            self.etypes[stype].append((etype, dtype))
        return graph


    def load_bertTokenizer(self, bert_tokenizer):
        self.Bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.maskcollator = DataCollatorForLanguageModeling(self.Bert_tokenizer, mlm=self.has_msk,
                                                            mlm_probability=self.maskrate)
        self.preFix = dict()
        for ntype in self.types:
            ntype_Emb = \
                self.Bert_tokenizer("The " + ntype + " is:", padding=True, truncation=False, return_tensors="pt")[
                    'input_ids'].squeeze(dim=0)[:-1]
            self.preFix[ntype] = ntype_Emb
        self.total_vocab = self.Bert_tokenizer.vocab_size
        self.cls_id, self.mask_id = self.Bert_tokenizer.cls_token_id, self.Bert_tokenizer.mask_token_id
        self.pad_id, self.sep_id = self.Bert_tokenizer.pad_token_id, self.Bert_tokenizer.sep_token_id


    def sample_title(self, stype, sids, neigh_type, neigh_name, sample_num=10):
        res = []
        second_hops = self.sampleNei({stype: sids}, self.etypes[stype], sample_num)
        neighbors = second_hops[neigh_type].numpy().tolist()
        for sam_id in neighbors:
            if sam_id not in self.total_nodes[neigh_name]:
                continue
            else:
                res.append(sam_id)
        return res


    def generate_PosNeg(self, dtype, pos_ids, neg_ids, neg_ratio=5):
        if len(pos_ids) != 0:
            pos_patents = pos_ids.unique().clone()
            flag = np.zeros(self.node_num[dtype], dtype=np.long)
            flag[neg_ids] = 1
            flag[pos_ids] = 2
            other_ids = self.idlist[dtype][flag == 0]
            neg_ids = torch.from_numpy(self.idlist[dtype][flag == 1])

            select_ids = np.unique(np.random.randint(low=0, high=len(other_ids),
                                                     size=min(len(pos_patents) * neg_ratio, len(other_ids)))).tolist()
            res = torch.tensor(other_ids[select_ids])

            neg_len = len(neg_ids) + len(res)
        else:
            pos_patents = torch.tensor([])
            res = torch.tensor([1])
            neg_ids = torch.tensor([])
            neg_len = len(res) + len(neg_ids)
        id_all = torch.cat([neg_ids, res, pos_patents]).long()
        truth_i = torch.tensor([0.0] * neg_len + [1.0] * len(pos_patents))
        mask_i = torch.zeros(len(truth_i)) if len(pos_ids) == 0 else torch.ones(len(truth_i))
        return id_all, truth_i, mask_i


    def sampleNei(self, ntype: dict, etypes: list, nei_num: int) -> dict:
        neigh_layer_1 = dgl.sampling.sample_neighbors(self.graph, ntype, nei_num, edge_dir='out')
        neigh_nodes = {dtype: [] for etype, dtype in etypes}
        for etype_tuple in etypes:
            etype, dsttype = etype_tuple
            neigh_nodes[dsttype].append(neigh_layer_1.edges(etype=etype)[1].unique())
        for ntype, nodes in neigh_nodes.items():
            neigh_nodes[ntype] = torch.cat(nodes).long()
        return neigh_nodes


    def return_neiLabels(self, input_ids: [torch.Tensor, str], layers=2, nei_num=[4, 2], mask_ratio=2):
        return_dict = [{ntype: [] for ntype in self.etypes.keys()} for i in range(3)]
        for nid, ntype in zip(input_ids[0], input_ids[1]):
            preNeis = {ntype: torch.tensor([nid]).long()}
            high_Neis_dict = {ntype: [torch.tensor([nid])]}
            for i in range(0, layers):
                i_Nei = dict()
                for stype, sids in preNeis.items():
                    if len(sids) != 0:
                        second_hops = self.sampleNei({stype: sids}, self.etypes[stype], nei_num[i])
                        for dtype, dids in second_hops.items():
                            if len(dids) == 0:
                                continue
                            if dtype not in i_Nei:
                                i_Nei[dtype] = []
                            i_Nei[dtype].append(dids.unique())

                for stype, sids in i_Nei.items():
                    i_Nei[stype] = torch.cat(sids).unique()
                    if stype not in high_Neis_dict:
                        high_Neis_dict[stype] = []
                    high_Neis_dict[stype].append(i_Nei[stype])
                preNeis = i_Nei

            for stype in self.etypes.keys():
                if stype in high_Neis_dict:
                    pos_ids = torch.cat(high_Neis_dict[stype]).unique().long()
                else:
                    pos_ids = torch.tensor([])
                id_pairs, truth, masks = self.generate_PosNeg(dtype=stype,
                                                              pos_ids=pos_ids,
                                                              neg_ids=torch.tensor([]),
                                                              neg_ratio=mask_ratio)
                return_dict[0][stype].append(id_pairs)
                return_dict[1][stype].append(truth)
                return_dict[2][stype].append(masks)

        new_dict = []
        for i, dict_lists in enumerate(return_dict):
            new_dict.append(dict())
            for ntype, nlist in dict_lists.items():
                if len(nlist) == 0:
                    continue
                new_dict[i][ntype] = pad_sequence(nlist, batch_first=True, padding_value=0)

        return new_dict


    def getTextInGraph(self, type, ids, raw_text=False):
        if raw_text:
            return self.raw_vir_dict[type][ids]
        return self.virtual_dict[type][ids]


    def maskSentences(self, texts):
        """
        :param texts: [paper,author,affiliation] / [paper]
        :return: raw_text: torch.size([k, F]), mask_text: torch.size([k, F]), len_i: [l_1,...l_k]
        """
        raw_texts, mask_texts, len_i = [], [], []

        for i in range(len(texts)):
            text_i = texts[i][texts[i] != self.pad_id][1:]
            assert len(text_i) > 0
            raw_texts.append(text_i.clone())
            len_i.append(len(text_i))
            mask_text = self.random_word(text_i) if self.has_msk else text_i
            mask_texts.append(mask_text)
            assert text_i.shape == mask_text.shape

        max_len = 510
        cut_num = len(texts) + 2
        if sum(len_i) > (max_len - cut_num):
            total_len = sum(len_i)
            more_len = (total_len - (max_len - cut_num))
            new_raw, new_mask, new_len = [], [], []
            if more_len == 0:
                more_len = 1
            for raw_text, mask_text, len_t in zip(raw_texts, mask_texts, len_i):
                end_len = int(len(raw_text) - more_len * (len(raw_text) / total_len))
                new_raw.append(raw_text[:end_len])
                new_mask.append(mask_text[:end_len])
                new_len.append(end_len)
        else:
            new_raw, new_mask, new_len = raw_texts, mask_texts, len_i
        assert sum(new_len) <= (max_len - cut_num)
        return new_raw, new_mask, new_len


    def random_word(self, tokens):
        sentence = self.Bert_tokenizer.convert_ids_to_tokens(tokens)
        for i, token in enumerate(tokens):
            if sentence[i].lower() in stopwords:
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < self.maskrate:
                prob /= self.maskrate

                # 80% randomly change token to mask token
                if prob < self.realmkrate:
                    tokens[i] = self.mask_id

                # 10% randomly change token to random token
                elif prob < self.realmkrate + self.randrate:
                    tokens[i] = random.choice(list(range(self.Bert_tokenizer.vocab_size)))

                # -> rest 10% randomly keep current token

        return tokens


    def generateSen_ids(self, len_id, keepSame=True):
        total_tokens = [torch.tensor([0])]
        pre_sen_token = torch.tensor([0] * len_id[0])
        total_tokens.append(pre_sen_token)
        total_tokens.append(torch.tensor([0]))
        for i in np.arange(1, len(len_id)):
            aut_sen_token = torch.tensor([0 if keepSame else 1] * len_id[i])
            total_tokens.append(aut_sen_token)
            total_tokens.append(torch.tensor([0 if keepSame else 1]))
        token_type_ids = torch.cat(total_tokens, dim=0)
        return token_type_ids


    def connectSentences(self, sentence_pairs: [torch.Tensor]):
        """
        :param sentence_pairs: torch.size([tensor,...,tensor]])
        :return: total_text: torch.size([N*(K+1)+1])
        """
        total_text = [torch.tensor([self.cls_id])]
        for i, sentence in enumerate(sentence_pairs):
            if i > 0:
                total_text.append(torch.tensor([self.sep_id]))
            total_text.append(sentence)
        total_text.append(torch.tensor([self.sep_id]))
        return torch.cat(total_text)


    def MaskToken(self, rawids: [torch.Tensor, str]):
        raw_texts, token_ids, mask_texts = [], [], []
        labels = None
        texts = []
        new_ids = [[], []]
        for sid, stype in zip(rawids[0], rawids[1]):
            text_i = []
            if stype != "paper":
                if stype != 'institution':
                    neiPats = self.sample_title(stype, sid, neigh_type="paper", neigh_name="title")
                else:
                    neiPats = self.sample_title(stype, sid, neigh_type="author", neigh_name="author")
                if len(neiPats) < 2:
                    continue

                if stype == 'author':
                    preFix = self.Bert_tokenizer("The papers written by ", padding=True, truncation=False,
                                                 return_tensors="pt")[
                        'input_ids']
                    neigh_type = 'title'
                elif stype == 'field_of_study':
                    preFix = self.Bert_tokenizer("The papers which have topic ", padding=True, truncation=False,
                                                 return_tensors="pt")[
                        'input_ids']
                    neigh_type = 'title'
                elif stype == 'institution':
                    preFix = self.Bert_tokenizer("The authors who are affiliated with affiliation ", padding=True,
                                                 truncation=False, return_tensors="pt")[
                        'input_ids']
                    neigh_type = 'author'
                else:
                    continue
                text_i.append(preFix)
                text_i.append(self.getTextInGraph(type=stype, ids=sid, raw_text=True))
                for nei_pat in neiPats:
                    text_i.append(self.getTextInGraph(type=neigh_type, ids=nei_pat))  # torch.size([k, F])
            else:
                text_i = [self.getTextInGraph(type=stype, ids=sid)]
            new_ids[0].append(sid)
            new_ids[1].append(stype)
            texts.append(text_i)

        if len(texts) > 0:
            for pid in range(len(texts)):
                raw_text, mask_text, len_i = self.maskSentences(texts[pid])
                token_id = self.generateSen_ids(len_i)
                token_ids.append(token_id)
                connect_raw = self.connectSentences(raw_text)
                connect_mask = self.connectSentences(mask_text)
                assert len(connect_raw) == len(token_id)

                raw_texts.append(connect_raw)
                mask_texts.append(connect_mask)

            raw_texts = pad_sequence(raw_texts, batch_first=True, padding_value=0)
            mask_texts = pad_sequence(mask_texts, batch_first=True, padding_value=0)
            token_ids = pad_sequence(token_ids, batch_first=True, padding_value=1)

            assert raw_texts.shape == mask_texts.shape
            assert mask_texts.shape == token_ids.shape

            flatten_mask = mask_texts.flatten()
            total_len = torch.tensor(range(len(flatten_mask)))
            mask_ids = total_len[flatten_mask == self.mask_id]
        else:
            mask_ids = torch.tensor([])
        return raw_texts, mask_texts, mask_ids, token_ids, labels, new_ids


class Data_GoodReads(object):
    def __init__(self, graph_path, abstract_path, title_path, publisher_path, author_path,
                 filter_book, filter_publisher, filter_author, sample_title,
                 bert_tokenizer, batch_size, has_neighbor,
                 has_mask, sample_graph_num=5, pred_layer=2):
        self.batch_size = batch_size
        self.sample_title_num = sample_title
        self.types = ["patent", "title", "applicant", "company"]
        self.randrate, self.realmkrate, self.maskrate = 0.1, 0.8, 0.4
        self.pred_layer = pred_layer
        self.has_msk, self.has_neighbor = has_mask, has_neighbor
        self.data_dict, self.total_nodes = {}, {}
        self.data_dict["patent"] = self.load_dict(abstract_path)
        self.data_dict["title"] = self.load_dict(title_path)
        self.data_dict["applicant"] = self.load_dict(author_path)
        self.data_dict["publisher"] = self.load_dict(publisher_path)
        self.graph = self.load_graph(graph_path)
        self.load_bertTokenizer(bert_tokenizer)

        self.map_reIndex([filter_book, filter_book, filter_author, filter_publisher])
        self.load_dataset(batch_size, sample_graph_num)

    def load_dict(self, text_path):
        data_dict = pd.read_csv(text_path).dropna().drop_duplicates()
        return data_dict

    def map_reIndex(self, filter_list):
        self.node_num = {}
        self.idlist = {}
        self.virtual_dict = {}
        for i, type in enumerate(self.types):
            data = pd.read_csv(filter_list[i])
            merge_data = pd.merge(data, self.data_dict[type], how='left').sort_values(by=data.columns[-1],
                                                                                      ascending=True)
            if type != 'title':
                self.node_num[type] = len(data)
            self.total_nodes[type] = merge_data.dropna().values[:, 1]
            self.idlist[type] = np.array(range(len(data)))

            texts = []
            for x in tqdm(merge_data.values[:, -1]):
                if pd.isnull(x):
                    texts.append(torch.tensor([self.cls_id]))
                    continue
                x_ = str(x).replace(" ", "")[1:-1].split(",")
                texts.append(torch.cat([self.preFix[type], torch.tensor([int(i) for i in x_][1:-1])]))
            self.virtual_dict[type] = pad_sequence(texts, batch_first=True, padding_value=0).long()

    def load_dataset(self, batch_size, sample_graph_num):
        total_num, stype = [], []
        for ntype in self.etypes.keys():
            total_num.extend(self.total_nodes[ntype])
            stype.extend([ntype] * len(self.total_nodes[ntype]))
        self.total_num = len(total_num)
        self.dataloader = DataLoader(FeatureDataSet(src_ids=total_num, idtypes=stype), batch_size=batch_size,
                                     shuffle=True)
        print("training process, number of nodes:{}".format(self.total_num))

    def load_graph(self, graph_path):
        graph = load_graphs(graph_path)[0][0]
        self.etypes = {ntype: [] for ntype in graph.ntypes}
        self.embedding = dict()
        for ntype in graph.ntypes:
            if "feat" in graph.nodes[ntype].data:
                self.embedding[ntype] = graph.nodes[ntype].data["feat"]

        self.node_num = {}
        self.nodeEmb = nn.ModuleDict()
        for ntype in graph.ntypes:
            self.node_num[ntype] = len(graph.nodes(ntype))
            nor_feat = graph.nodes[ntype].data["feat"]

            self.nodeEmb[ntype] = nn.Embedding.from_pretrained(
                torch.cat([nor_feat, torch.zeros(1, nor_feat.shape[1])], dim=0), freeze=True)

        for stype, etype, dtype in graph.canonical_etypes:
            self.etypes[stype].append((etype, dtype))
        return graph

    def load_bertTokenizer(self, bert_tokenizer):
        self.Bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.maskcollator = DataCollatorForLanguageModeling(self.Bert_tokenizer, mlm=self.has_msk,
                                                            mlm_probability=0.4)
        self.preFix = dict()
        for ntype in self.types:
            ntype_Emb = \
                self.Bert_tokenizer("The " + ntype + " is:", padding=True, truncation=False, return_tensors="pt")[
                    'input_ids'].squeeze(dim=0)[:-1]
            self.preFix[ntype] = ntype_Emb
        self.total_vocab = self.Bert_tokenizer.vocab_size
        self.cls_id, self.mask_id = self.Bert_tokenizer.cls_token_id, self.Bert_tokenizer.mask_token_id
        self.pad_id, self.sep_id = self.Bert_tokenizer.pad_token_id, self.Bert_tokenizer.sep_token_id

    def sample_title(self, stype, sids, sample_num=5):
        res = []
        second_hops = self.sampleNei({stype: sids}, self.etypes[stype], sample_num)
        neighbors = second_hops["patent"].numpy().tolist()
        for sam_id in neighbors:
            if sam_id not in self.total_nodes["title"]:
                continue
            else:
                res.append(sam_id)
        return res

    def generate_PosNeg(self, dtype, pos_ids, neg_ids, neg_ratio=5):
        if len(pos_ids) != 0:
            pos_patents = pos_ids.unique().clone()
            flag = np.zeros(self.node_num[dtype], dtype=np.long)
            flag[neg_ids] = 1
            flag[pos_ids] = 2
            other_ids = self.idlist[dtype][flag == 0]
            neg_ids = torch.from_numpy(self.idlist[dtype][flag == 1])

            select_ids = np.unique(np.random.randint(low=0, high=len(other_ids),
                                                     size=min(len(pos_patents) * neg_ratio, len(other_ids)))).tolist()
            res = torch.tensor(other_ids[select_ids])
            neg_len = len(neg_ids) + len(res)
        else:
            pos_patents = torch.tensor([])
            res = torch.tensor([1])
            neg_ids = torch.tensor([])
            neg_len = len(res) + len(neg_ids)
        id_all = torch.cat([neg_ids, res, pos_patents]).long()
        truth_i = torch.tensor([0.0] * neg_len + [1.0] * len(pos_patents))
        mask_i = torch.zeros(len(truth_i)) if len(pos_ids) == 0 else torch.ones(len(truth_i))
        return id_all, truth_i, mask_i

    def sampleNei(self, ntype: dict, etypes: list, nei_num: int) -> dict:
        neigh_layer_1 = dgl.sampling.sample_neighbors(self.graph, ntype, nei_num, edge_dir='out')
        neigh_nodes = {dtype: [] for etype, dtype in etypes}
        for etype_tuple in etypes:
            etype, dsttype = etype_tuple
            neigh_nodes[dsttype].append(neigh_layer_1.edges(etype=etype)[1].unique())
        for ntype, nodes in neigh_nodes.items():
            neigh_nodes[ntype] = torch.cat(nodes).long()
        return neigh_nodes

    def return_neiLabels(self, input_ids: [torch.Tensor, str], nei_num=5, mask_ratio=3):
        layers = self.pred_layer
        return_dict = [{ntype: [] for ntype in self.etypes.keys()} for i in range(3)]
        for nid, ntype in zip(input_ids[0], input_ids[1]):
            preNeis = {ntype: torch.tensor([nid]).long()}
            high_Neis_dict = {ntype: [torch.tensor([nid])]}
            for i in range(0, layers):
                i_Nei = dict()
                for stype, sids in preNeis.items():
                    if len(sids) != 0:
                        second_hops = self.sampleNei({stype: sids}, self.etypes[stype], nei_num)
                        for dtype, dids in second_hops.items():
                            if len(dids) == 0:
                                continue
                            if dtype not in i_Nei:
                                i_Nei[dtype] = []
                            i_Nei[dtype].append(dids.unique())

                for stype, sids in i_Nei.items():
                    i_Nei[stype] = torch.cat(sids).unique()
                    if stype not in high_Neis_dict:
                        high_Neis_dict[stype] = []
                    high_Neis_dict[stype].append(i_Nei[stype])

                preNeis = i_Nei

            for stype in self.etypes.keys():
                if stype in high_Neis_dict:
                    pos_ids = torch.cat(high_Neis_dict[stype]).unique().long()
                else:
                    pos_ids = torch.tensor([])
                id_pairs, truth, masks = self.generate_PosNeg(dtype=stype,
                                                              pos_ids=pos_ids,
                                                              neg_ids=torch.tensor([]))

                return_dict[0][stype].append(id_pairs)
                return_dict[1][stype].append(truth)
                return_dict[2][stype].append(masks)

        new_dict = []
        for i, dict_lists in enumerate(return_dict):
            new_dict.append(dict())
            for ntype, nlist in dict_lists.items():
                if len(nlist) == 0:
                    continue
                new_dict[i][ntype] = pad_sequence(nlist, batch_first=True, padding_value=0)

        return new_dict

    def getTextInGraph(self, type, ids):
        return self.virtual_dict[type][ids]

    def maskSentences(self, texts):
        """
        :param texts: [paper,author,affiliation] / [paper]
        :return: raw_text: torch.size([k, F]), mask_text: torch.size([k, F]), len_i: [l_1,...l_k]
        """
        raw_texts, mask_texts, len_i = [], [], []

        for i in range(len(texts)):
            text_i = texts[i][texts[i] != self.pad_id][1:]
            assert len(text_i) > 0
            raw_texts.append(text_i.clone())
            len_i.append(len(text_i))
            mask_text = text_i
            mask_texts.append(mask_text)
            assert text_i.shape == mask_text.shape

        max_len = 509
        cut_num = len(texts) + 2
        if sum(len_i) > (max_len - cut_num):
            total_len = sum(len_i)
            more_len = (total_len - (max_len - cut_num))
            new_raw, new_mask, new_len = [], [], []
            if more_len == 0:
                more_len = 1
            for raw_text, mask_text, len_t in zip(raw_texts, mask_texts, len_i):
                end_len = int(len(raw_text) - more_len * (len(raw_text) / total_len))
                new_raw.append(raw_text[:end_len])
                new_mask.append(mask_text[:end_len])
                new_len.append(end_len)
        else:
            new_raw, new_mask, new_len = raw_texts, mask_texts, len_i
        assert sum(new_len) <= (max_len - cut_num)
        return new_raw, new_mask, new_len

    def random_word(self, tokens):
        sentence = self.Bert_tokenizer.convert_ids_to_tokens(tokens)
        for i, token in enumerate(tokens):
            if sentence[i].lower() in stopwords:
                # print("stopword " + sentence[i])
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < self.maskrate:
                prob /= self.maskrate

                # 80% randomly change token to mask token
                if prob < self.realmkrate:
                    tokens[i] = self.mask_id

                # 10% randomly change token to random token
                elif prob < self.realmkrate + self.randrate:
                    tokens[i] = random.choice(list(range(self.Bert_tokenizer.vocab_size)))

                # -> rest 10% randomly keep current token

        return tokens

    def generateSen_ids(self, len_id, keepSame=True):
        total_tokens = [torch.tensor([0])]
        pre_sen_token = torch.tensor([0] * len_id[0])
        total_tokens.append(pre_sen_token)
        total_tokens.append(torch.tensor([0]))
        for i in np.arange(1, len(len_id)):
            aut_sen_token = torch.tensor([0 if keepSame else 1] * len_id[i])
            total_tokens.append(aut_sen_token)
            total_tokens.append(torch.tensor([0 if keepSame else 1]))
        token_type_ids = torch.cat(total_tokens, dim=0)
        return token_type_ids

    def connectSentences(self, sentence_pairs: [torch.Tensor]):
        """
        :param sentence_pairs: torch.size([tensor,...,tensor]])
        :return: total_text: torch.size([N*(K+1)+1])
        """
        total_text = [torch.tensor([self.cls_id])]
        for i, sentence in enumerate(sentence_pairs):
            if i > 0:
                total_text.append(torch.tensor([self.sep_id]))
            total_text.append(sentence)
        total_text.append(torch.tensor([self.sep_id]))
        return torch.cat(total_text)

    def MaskToken(self, rawids: [torch.Tensor, str]):
        raw_texts, mask_texts, token_ids = [], [], []
        labels = None
        texts = []
        new_ids = [[], []]
        for sid, stype in zip(rawids[0], rawids[1]):
            if stype == "patent":
                text_i = [self.getTextInGraph(type=stype, ids=sid)]
            else:
                text_i = []
            if stype != "patent":
                neiPats = self.sample_title(stype, sid, self.sample_title_num)
                for nei_pat in neiPats:
                    text_i.append(self.getTextInGraph(type="title", ids=nei_pat))  # torch.size([k, F])

            if len(text_i) == 0:
                continue
            new_ids[0].append(sid)
            new_ids[1].append(stype)
            texts.append(text_i)

        for pid in range(len(texts)):
            raw_text, mask_text, len_i = self.maskSentences(texts[pid])
            token_id = self.generateSen_ids(len_i)
            token_ids.append(token_id)
            connect_raw = self.connectSentences(raw_text)
            connect_mask = self.connectSentences(mask_text)
            assert len(connect_raw) == len(token_id)

            raw_texts.append(connect_raw)
            mask_texts.append(connect_mask)

        output_mask = self.maskcollator(raw_texts)
        mask_texts, label_texts = output_mask['input_ids'], output_mask['labels']
        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=0).long()
        #
        assert mask_texts.shape == token_ids.shape
        #
        flatten_mask = label_texts.flatten()
        total_len = torch.tensor(range(len(flatten_mask)))
        mask_ids = total_len[flatten_mask != -100]

        return label_texts, mask_texts, mask_ids, token_ids, labels, new_ids