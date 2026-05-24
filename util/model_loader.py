import os
import models.densenet as dn
# import models.wideresnet as wn

import numpy as np
import torch

def get_model(args, num_classes, load_ckpt=True, load_epoch=None):
    # info = np.load(f"./{args.in_dataset}_{args.model_arch}_feat_stat.npy")
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == "mobilenetv2":
            from models.mobilenetv2 import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50-supcon':
            from models.resnet_supcon import SupConResNet
            model = SupConResNet(num_classes=num_classes)
            if load_ckpt:
                checkpoint = torch.load("./checkpoints/{in_dataset}/pytorch_{model_arch}_imagenet/supcon.pth".format(
                    in_dataset=args.in_dataset, model_arch=args.model_arch))
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
                model.load_state_dict(state_dict, strict=False)
    else:
        # create model
        if args.model_arch == 'densenet':
            model = dn.DenseNet3(args, args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=True,
                                 dropRate=args.droprate, normalizer=None, method=args.method, p=args.p)
        elif args.model_arch == 'densenet-supcon':
            from models.densenet import DenseNet3
            model = DenseNet3(args, args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=True,
                                     dropRate=args.droprate, normalizer=None, method=args.method, p=args.p)
        elif args.model_arch == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet32':
            from models.resnet_cifar_lt import resnet32
            model = resnet32(num_classes=num_classes)
        elif args.model_arch == 'resnet18-supcon':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet18-supce':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet34':
            from models.resnet import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet18-cider':
            from models.resnet_cider import resnet18
            model = resnet18()
        elif args.model_arch == 'resnet34-cider':
            from models.resnet_cider import resnet34
            model = resnet34()
        elif args.model_arch == 'resnet34-supcon':
            from models.resnet_cider import resnet34
            model = resnet34()
        elif args.model_arch == 'resnet34-supce':
            from models.resnet_ss import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'wrn':
            from models.wrn import WideResNet
            model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=args.droprate)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)


        if load_ckpt:
            epoch = args.epochs
            if load_epoch is not None:
                epoch = load_epoch
            if args.model_arch != 'wrn':
                # checkpoint = torch.load("./supcon-cifar100-1w7xyhfc-ep=999.ckpt", map_location='cpu')
                if args.model_arch == "resnet34-supcon":
                    checkpoint = torch.load(
                        "./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth".format(
                            in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch), map_location='cpu')
                    # for param_tensor in model.state_dict():
                    #     print(param_tensor)
                    # print(1)
                    # for key, value in checkpoint["model"].items():
                    #     print(key)
                elif args.model_arch == "resnet32":
                    checkpoint = torch.load(
                        "./checkpoints/{in_dataset}/{model_arch}/checkpoint_200_{imb_factor}.pth.tar".format(
                            in_dataset=args.in_dataset, model_arch="resnet32", imb_factor=int(args.imb_factor)),
                        map_location='cpu')
                else:
                    checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch), map_location='cpu')

                if args.model_arch not in ['resnet18-cider', 'resnet34-cider', 'resnet34-supcon', "resnet32"]:
                    checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict']. items()}}

                elif args.model_arch in ['resnet18-cider', 'resnet34-cider']:
                    checkpoint2 = {'state_dict': {}}
                    for key, value in checkpoint['state_dict']. items():
                        if not (key.startswith("fc") or key.startswith("head")):
                            checkpoint2['state_dict'][key.replace("encoder.", "")] = value
                    checkpoint = checkpoint2

                elif args.model_arch == 'resnet34-supcon':
                    checkpoint2 = {'state_dict': {}}
                    for key, value in checkpoint['model']. items():
                        if not (key.startswith("fc") or key.startswith("head")):
                            checkpoint2['state_dict'][key.replace("encoder.", "")] = value
                    checkpoint = checkpoint2

                else:
                    checkpoint2 = {'state_dict': {}}
                    for key, value in checkpoint['state_dict_model'].items():
                        checkpoint2["state_dict"][key.replace("module.", "")] = value
                    for key, value in checkpoint['state_dict_classifier'].items():
                        checkpoint2["state_dict"][key.replace("module.", "")] = value
                    checkpoint = checkpoint2


                # for param_tensor in model.state_dict():
                #     print(param_tensor)
                # print(args.model_arch)
                # for key, value in checkpoint2['state_dict'].items():
                #     print(key)

                model.load_state_dict(checkpoint['state_dict'])


    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
