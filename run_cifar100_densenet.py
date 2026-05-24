import os
import time
from util.args_loader import get_args
from util import metrics
import torch
# import faiss
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from numpy.linalg import norm
import torch

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


prepos_feat = lambda x: np.ascontiguousarray(x[:, range(282, 624)]) # Last Layer only


t1 = 0.08
ord_def = 2.5

q = 1/(1-1/ord_def)
normalizer = lambda x: x / (np.linalg.norm(x, ord=ord_def, axis=-1, keepdims=True) + 1e-10)

def dist(a, b):
    distance = - 1/2 * (np.linalg.norm(a, ord=q, axis=-1, keepdims=True) + 1e-10) ** 2 + np.dot(a, (b * np.abs(b) ** (q - 2) / (np.linalg.norm(b, ord=q, axis=-1, keepdims=True) + 1e-10) ** (q - 2)))
    return distance

cache_name_feat = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers_feat.npy"
cache_name_label = f"cache/{args.in_dataset}_train_{args.name}_in_alllayers_label.npy"
feat_log = np.load(cache_name_feat, allow_pickle=True)
label_log = np.load(cache_name_label, allow_pickle=True)
feat_log = feat_log.T.astype(np.float32)
class_num = 100

cache_name_feat = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers_feat.npy"
cache_name_label = f"cache/{args.in_dataset}_val_{args.name}_in_alllayers_label.npy"
feat_log_val = np.load(cache_name_feat, allow_pickle=True)
label_log_val = np.load(cache_name_label, allow_pickle=True)
feat_log_val = feat_log_val.T.astype(np.float32)

ood_feat_log_all = {}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers_feat.npy"
    ood_feat_log = np.load(cache_name, allow_pickle=True)
    ood_feat_log = ood_feat_log.T
    ood_feat_log_all[ood_dataset] = ood_feat_log

ftrain = normalizer(prepos_feat(feat_log))
ftest = normalizer(prepos_feat(feat_log_val))

food_all = {}
for ood_dataset in args.out_datasets:
    food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

class_protytypes = np.zeros((class_num, ftrain.shape[1]))

score_log_val_exp = np.zeros((ftest.shape[0], class_num))

for k in range(class_num):
    mask_k = (label_log == k)
    mask_k_reverse = 1 - mask_k
    k_protype = normalizer(np.mean(feat_log[:, range(282, 624)]*mask_k[:, np.newaxis], axis=0))
    k_protype = np.asarray(k_protype)
    class_protytypes[k, :] = k_protype

alpha = 1
num = np.shape(ftrain)[0]
ftrain = ftrain[np.random.choice(num, int(num*alpha), replace=True)]
score_log_exp = np.exp(dist(ftrain, class_protytypes.T)/t1)
score_log_val_exp = np.exp(dist(ftest, class_protytypes.T)/t1)
score_log_val_exp = score_log_val_exp/np.sum(score_log_exp, axis=0)

scores_in = np.log(np.sum(score_log_val_exp, axis=1) + 1e-15)

all_results = []
for ood_dataset, food in food_all.items():
    food = normalizer(food)
    ood_score_log_exp = np.exp(dist(food, class_protytypes.T) / t1)
    ood_score_log_exp = ood_score_log_exp / np.sum(score_log_exp, axis=0)
    scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1) + 1e-15)
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'ours, t1={t1}, ord_def={ord_def}')

