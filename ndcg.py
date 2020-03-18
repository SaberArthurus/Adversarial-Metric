# see https://arxiv.org/pdf/2003.07573.pdf
import math
import torch
import torch.nn.functional as F

def find_index(scores, thr):
    start_point = 1
    pre_index = None
    while True:
        value, index  = torch.topk(scores, start_point)
        if torch.sum(value) <= thr: 
            start_point += 1
            continue
        break
    return index.cpu().numpy().tolist()
        


def NDCG(adv_logits, clean_logits, thr):
    assert adv_logits.shape[0] == 1 and clean_logits.shape[0] == 1
    adv_scores = F.softmax(adv_logits)
    clean_scores = F.softmax(clean_logits)
    index_adv = find_index(adv_scores, thr)
    index_clean = find_index(clean_scores, thr)
    adcg = 0
    idcg = 0
    for idx, tensor_index in enumerate(index_clean):
        conf = clean_scores[tensor_index].item()
        idcg += (2**conf - 1) / math.log2(1+idx)
    
    for idx, tensor_index in enumerate(index_adv):
        if tensor_index in index_clean:
            clean_idx = index_clean.index(tensor_index)
            conf = clean_scores[clean_idx].item()
            adcg += (2 ** conf - 1) / math.log2(1 + idx)
    return adcg / (idcg + 1e-7)

