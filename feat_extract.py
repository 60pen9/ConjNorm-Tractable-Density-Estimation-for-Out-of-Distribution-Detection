import torch

import os
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
from models.preprocessing import *

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.M == 1:
    config_type = "eval"
else:
    config_type = "default"
loader_in_dict = get_loader_in(args, config_type=config_type, split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
print(num_classes)

if args.in_dataset == 'waterbirds':
    model = get_model(args, num_classes, load_ckpt=False)
else:
    model = get_model(args, num_classes, load_ckpt=True)
model.cuda()
model.eval()

batch_size = args.batch_size

FORCE_RUN = True

dummy_input = torch.zeros((1, 3, 32, 32)).cuda()

if args.score == "maha":
    out_features = model.intermediate_forward(dummy_input)
    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    out_features = torch.mean(out_features, 2)
    featdims = [out_features.shape[1]]
else:
    score, feature_list = model.feature_list(dummy_input)
    featdims = [feat.shape[1] for feat in feature_list]


begin = time.time()

if args.score == "maha":
    sample_mean, precision = sample_estimator(model, num_classes, trainloaderIn)

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn),]:
    if args.imb_factor>0:
        cache_name = f"./cache/{args.in_dataset}_{split}_{args.name}_{args.imb_factor}_in_alllayers.npy"
    else:
        cache_name = f"./cache/{args.in_dataset}_{split}_{args.name}_in_alllayers.npy"
    if FORCE_RUN or not os.path.exists(cache_name):

        feat_log = np.zeros((len(in_loader.dataset)*args.M, sum(featdims)))

        score_log = np.zeros((len(in_loader.dataset)*args.M, num_classes))
        label_log = np.zeros(len(in_loader.dataset)*args.M)

        model.eval()
        for m in range(0, args.M):
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx * batch_size + m*len(in_loader.dataset)
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset)) + m*len(in_loader.dataset)

                if args.score == "odin" and split == "val":
                    inputs = input_preprocessing_odin(args, inputs, model)
                elif args.score == "maha" and split == "val":
                    inputs = input_preprocessing_maha(args, inputs, model, num_classes, sample_mean, precision)
                else:
                    inputs = inputs

                if args.score == "maha":
                    out_features = model.intermediate_forward(inputs)
                    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                    outs = torch.mean(out_features, 2)
                    feat_log[start_ind:end_ind, :] = outs.data.cpu().numpy()
                    score = None
                else:
                    score, feature_list = model.feature_list(inputs)
                    # for layer_feat in feature_list:
                    #     print(F.adaptive_avg_pool2d(layer_feat, 1).squeeze().size())
                    out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list],
                                    dim=1)
                    feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()

                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                if score is not None:
                    score_log[start_ind:end_ind] = score.data.cpu().numpy()

                if batch_idx % 100 == 0:
                    print(f"{m}/{args.M}, {batch_idx}/{len(in_loader)}")

        np.save(cache_name, (feat_log.T, score_log.T, label_log))
    else:
        feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log, score_log = feat_log.T, score_log.T


for ood_dataset in args.out_datasets:
    loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
    out_loader = loader_test_dict.val_ood_loader
    if args.imb_factor>0:
        cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.imb_factor}_{args.name}_out_alllayers.npy"
    else:
        cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npy"
    if FORCE_RUN or not os.path.exists(cache_name):
        ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
        ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

        model.eval()
        for batch_idx, (inputs, _) in enumerate(out_loader):
            inputs = inputs.to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

            if args.score == "odin":
                inputs = input_preprocessing_odin(args, inputs, model)
            elif args.score == "maha":
                inputs = input_preprocessing_maha(args, inputs, model, num_classes, sample_mean, precision)
            else:
                inputs = inputs

            if args.score == "maha":
                out_features = model.intermediate_forward(inputs)
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                outs = torch.mean(out_features, 2)
                ood_feat_log[start_ind:end_ind, :] = outs.data.cpu().numpy()
                score = None
            else:
                score, feature_list = model.feature_list(inputs)
                out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)
                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()

            if score is not None:
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()

            if batch_idx % 100 == 0:
                print(f"{batch_idx}/{len(out_loader)}")
        np.save(cache_name, (ood_feat_log.T, ood_score_log.T))
    else:
        ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
        ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T


print(time.time() - begin)

