import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
# from utils.util import convert_all_data_to_gpu
import datetime


def recall_score_(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=top_k)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    # print(f'recall {top_k} predict_values: {value[8:11]}')
    # print(f'recall {top_k} predict_indices: {predict_indices[8:11]}')
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = (torch.logical_and(torch.logical_and(predict, truth), truth == 1)).sum(-1), truth.sum(-1)
    # print(f'recall {top_k}, tp: {tp[8:11]}, t: {t[8:11]}, ')
    # end_time = datetime.datetime.now()
    # print("recall_score cost %d seconds" % (end_time - start_time).seconds)
    return (tp.float() / t.float()).sum().item()


def precision(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=top_k)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = (torch.logical_and(torch.logical_and(predict, truth), truth == 1)).sum(-1), predict.sum(-1)
    # print(f'recall {top_k}, tp: {tp[8:11]}, t: {t[8:11]}, ')
    # end_time = datetime.datetime.now()
    # print("recall_score cost %d seconds" % (end_time - start_time).seconds)
    return (tp.float() / t.float()).sum().item()


def F1_scr(y_true, y_pred, top_k=8, method='macro'):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=top_k)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    # print(predict, truth)
    tp, p, t = (torch.logical_and(torch.logical_and(predict, truth), truth == 1)).sum(-1), predict.sum(-1), truth.sum(
        -1)
    recall = tp.float() / t.float()
    prec = tp.float() / p.float()
    # print(recall)
    # print(prec)
    eps = 1e-6
    f1 = 2 * recall * prec / (recall + prec + eps)
    if method == 'macro':
        p, r = 0, 0
        recall_, precision_ = 0, 0
    else:
        precision_ = precision_score(truth, predict, average='macro').item()
        recall_ = recall_score(truth, predict, average='macro').item()

        p = precision_score(truth, predict, average='micro').item()
        r = recall_score(truth, predict, average='micro').item()
    # return recall.mean().item(), prec.mean().item(), f1.mean().item(), p, r
    return recall_, precision_, f1.mean().item(), p, r


def dcg(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):

    Returns:

    """
    value, predict_indices = y_pred.topk(k=top_k)
    predict_indices = predict_indices[:, :top_k]
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg_score(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):
    Returns:

    """
    # start_time = datetime.datetime.now()
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    # end_time = datetime.datetime.now()
    # print("ndcg cost %d seconds" % (end_time - start_time).seconds)
    return (dcg_score / idcg_score).mean().item()


def PHR(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=top_k)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    hit_num = torch.logical_and(predict, truth).sum(dim=1).nonzero(as_tuple=False).shape[0]
    # end_time = datetime.datetime.now()
    # print("PHR cost %d seconds" % (end_time - start_time).seconds)
    return hit_num / truth.shape[0]
    # return hit_num


def get_metric(y_true, y_pred, idx=1, method='macro', stage='train'):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
        Returns:
            scores: dict
    """
    # idx_list = [[1],[1,8,5],[1,8,5],[1,8,5]]
    # idx_list = [[1], [1, 8, 5], [1, 8, 5]]
    idx_list = [[1, 3, 5], [1, 3, 5]]
    result = {}
    for top_k in idx_list[idx]:
        recall, prec, F1, P, R = F1_scr(y_true, y_pred, top_k=top_k, method=method)
        result.update({
            f'Mar_P_{top_k}': prec,
            f'Mar_R_{top_k}': recall,
            f'F1_{top_k}': F1,
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'Mic_P_{top_k}': P,
            f'Mic_R_{top_k}': R,
        })
        if stage != 'train':
            result.update({
                f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k),
            })
    return result


if __name__ == '__main__':
    a = torch.randn([4, 3])
    b = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
    print(get_metric(b, a))
