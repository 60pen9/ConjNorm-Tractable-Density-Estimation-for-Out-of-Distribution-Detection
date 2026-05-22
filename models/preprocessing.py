from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


def input_preprocessing_odin(args, inputs, model, temp=1000, Magnitude=0.002):
    criterion = nn.CrossEntropyLoss()

    inputs = Variable(inputs, requires_grad=True)
    outputs, _ = model.feature_list(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    outputs = outputs / temp

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # if args.model_arch.startswith("densenet"):
    #     gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
    #     gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
    #     gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)
    # elif args.model_arch.startswith("resnet"):
    #     gradient[:, 0] = (gradient[:, 0]) / (0.2023)
    #     gradient[:, 1] = (gradient[:, 1]) / (0.1994)
    #     gradient[:, 2] = (gradient[:, 2]) / (0.2010)

    tempInputs = torch.add(inputs.data,  -Magnitude, gradient)

    return tempInputs

def input_preprocessing_maha(args, inputs, model, num_classes, sample_mean, precision, layer_index=3, Magnitude=0.002):

    inputs = Variable(inputs, requires_grad=True)

    # out_features = model.intermediate_forward(inputs, layer_index=layer_index)
    out_features = model.intermediate_forward(inputs)

    out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
    out_features = torch.mean(out_features, 2)

    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1, 1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)
    pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
    loss = torch.mean(-pure_gau)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # if args.model_arch.startswith("densenet"):
    #     gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
    #                          gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
    #     gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
    #                          gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
    #     gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
    #                          gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
    # elif args.model_arch.startswith("resnet"):
    #     gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
    #                          gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
    #     gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
    #                          gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
    #     gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
    #                          gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))


    tempInputs = torch.add(inputs.data,  -Magnitude, gradient)

    return tempInputs


def sample_estimator(model, num_classes, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    # correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        # total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        # pred = output.data.max(1)[1]
        # equal_flag = pred.eq(target.cuda()).cpu()
        # correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision