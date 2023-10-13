from transformers import BertModel
import torch
import torch.nn as nn
from Modules.MaskLM import BertForMaskedLM
from Modules.NeigborPrediction import get_neighborPrediction

class Bert(nn.Module):
    def __init__(self, bertName, dropout, device, vocab_num, pred_type,
                 has_mlm, has_neighbor, node_num, graph, mskrate=0.2, pretrainEmb=None):
        super(Bert, self).__init__()
        self.encoder = BertModel.from_pretrained(bertName)
        self.pred_type = pred_type
        self.vob_num = vocab_num
        self.has_mlm = has_mlm
        self.node_num = node_num
        self.has_neighbor = has_neighbor
        self.mskrate = mskrate
        if self.has_mlm:
            self.maskLM = BertForMaskedLM(768, False, 1e-12, self.vob_num)
        if self.has_neighbor:
            self.predNeig = get_neighborPrediction(graph, pred_type, node_num, 768, dropout, pretrainEmb, mskrate=mskrate)

        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, ids, graph, sentences, token_type_ids, id_pairs_dict, nodeEmbed):
        feat_dst = sentences
        attn_mask = feat_dst.new_zeros(feat_dst.shape, dtype=torch.long)
        attn_mask[feat_dst != 0] = 1
        lengths = attn_mask.sum(dim=-1) + 1e-6

        batch_sz = feat_dst.shape[0]

        assert feat_dst.shape == token_type_ids.shape
        word_embs = self.encoder(input_ids=feat_dst,
                     token_type_ids=token_type_ids,
                     attention_mask=attn_mask,
                     return_dict=True)
        word_emb = self.dropout(word_embs.last_hidden_state)
        aver_feats = torch.sum(word_emb, dim=1) / lengths.unsqueeze(dim=1)

        if self.has_mlm:
            prob_logit = self.maskLM(word_emb, self.encoder.embeddings.word_embeddings.weight).reshape(-1, self.vob_num)
            assert prob_logit.shape == torch.Size([batch_sz*word_emb.shape[1], self.vob_num])
        else:
            prob_logit = None

        if self.has_neighbor:
            if self.pred_type == "matrixEmb":
                neigh_pred_list = self.predNeig(id_pairs_dict, nodeEmbed, word_emb, aver_feats, attn_mask)
            elif self.pred_type in ["gcnEmb"]:
                neigh_pred_list = self.predNeig(graph, id_pairs_dict, nodeEmbed, word_emb, aver_feats, attn_mask)
            else:
                neigh_pred_list = None
        else:
            neigh_pred_list = None

        return prob_logit, neigh_pred_list