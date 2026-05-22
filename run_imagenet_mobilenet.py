import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import numpy as np
from scipy import stats
from scipy.special import softmax
from models.mobilenetv2 import mobilenet_v2
from models.ash import *

args = get_args()

seed = args.seed
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class_num = 1000
prepos_feat = lambda x: np.ascontiguousarray(x).astype(np.float32)
t1 = 0.01
ord_def = 1.8
q = 1/(1-1/ord_def)

normalizer = lambda x: x / (np.linalg.norm(x, ord=ord_def, axis=-1, keepdims=True) + 1e-10)

def dist(a, b):
    distance = - 1/2 * (np.linalg.norm(a, ord=q, axis=-1, keepdims=True) + 1e-10) ** 2 + np.dot(a, (b * np.abs(b) ** (q - 2) / (np.linalg.norm(b, ord=q, axis=-1, keepdims=True) + 1e-10) ** (q - 2)))
    return distance

feat_log_val = np.load(f"F:/OneDrive/imagenet_feature_mobilenet_v2_sorted/val.npy")# you should load from your own address
ftest = normalizer(prepos_feat(feat_log_val))

score_log_exp = np.zeros(class_num)
# class_protytypes = np.zeros((class_num, 1280))

class_protytypes = np.load(f"cache/class_protytypes_mobilev2.npy")# you should load from your own address
class_protytypes = normalizer(class_protytypes)

for k in range(class_num):
    zero = np.zeros(class_num)
    zero[k] = 1
    k_ftrain = np.load(f"F:/OneDrive/imagenet_feature_mobilenet_v2_sorted/class_{k}.npy".format(k=k)) # you should load from your own address

    # k_protype = prepos_feat(np.mean(k_ftrain, axis=0))
    # class_protytypes[k, :] = k_protype

    alpha = 1
    num = np.shape(k_ftrain)[0]
    k_ftrain = k_ftrain[np.random.choice(num, int(num * alpha), replace=True)]

    score_log_exp = score_log_exp + np.sum(np.exp(dist(normalizer(k_ftrain), class_protytypes.T) / t1), axis=0)*zero[np.newaxis,:]
    # score_log_exp = score_log_exp + np.sum(np.exp(dist(normalizer(k_ftrain), class_protytypes.T) / t1), axis=0)


# np.save(f"cache/class_protytypes_mobilev2.npy", class_protytypes) # you should save into your own address

score_log_val_exp = np.exp(dist(ftest, class_protytypes.T)/t1)
score_log_val_exp = score_log_val_exp/score_log_exp
scores_in = np.log(np.sum(score_log_val_exp, axis=1))

food_all = {}
ood_dataset_size = {
    'inat':10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640
}

for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 1280))
    food_all[ood_dataset] = normalizer(prepos_feat(ood_feat_log))

all_results = []
for ood_dataset, food in food_all.items():
    ood_score_log_exp = np.exp(dist(food, class_protytypes.T)/t1)
    ood_score_log_exp = ood_score_log_exp / score_log_exp
    scores_ood_test = np.log(np.sum(ood_score_log_exp, axis=1))
    results = metrics.cal_metric(scores_in, scores_ood_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, f'ours, t1={t1}, ord_def={ord_def}')


