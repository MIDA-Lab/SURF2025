# train.py

import os
import random
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    vgg16, VGG16_Weights
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd

# Attacks
from general_torch_model import GeneralTorchModel
from RayS_Single          import RayS
from OPT_attack_lf        import OPT_attack_lf
from Sign_OPT_lf          import OPT_attack_sign_SGD_lf

# Constants & Configs
QUERY_BUDGET = 10000
EPSILONS = {
    "MNIST":   0.3,
    "CIFAR10": 0.031,
    "ImageNet":0.05
}
NORM_STATS = {
    "MNIST":   ([0.1307]*3, [0.3081]*3),
    "CIFAR10": ([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010]),
    "ImageNet":([0.485,0.456,0.406], [0.229,0.224,0.225]),
}
ATTACK_CONFIGS = {
    'OPT':      {'alpha':0.2, 'beta':0.001, 'iterations':1500},
    'Sign-OPT': {'alpha':0.2, 'beta':0.001, 'iterations':1500, 'k':200},
    'RayS':     {}    # only needs ε
}

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Utilities
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calc_perturbation(x, adv):
    if adv is None: 
        return float('inf'), float('inf')
    d = (adv - x).view(-1)
    return float(d.norm(2).item()), float(d.abs().max().item())

# DataLoaders
def get_mnist_loaders(batch_size):
    mean, std = NORM_STATS['MNIST']
    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=tf)
    test  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )

def get_cifar10_loaders(batch_size):
    mean, std = NORM_STATS['CIFAR10']
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    testset  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )

def get_imagenet_loaders(batch_size, root, model_name):
    name = model_name.lower()
    if name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2
    elif name == 'densenet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1
    elif name == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unsupported ImageNet model: {model_name}")

    train_tf = weights.transforms()
    val_tf   = weights.transforms(kind=weights.transforms.KIND_VAL)

    train_ds = torchvision.datasets.ImageFolder(os.path.join(root,'train'), transform=train_tf)
    val_ds   = torchvision.datasets.ImageFolder(os.path.join(root,'val'),   transform=val_tf)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    )

def get_dataloaders(dataset, batch_size, imagenet_root, model_name):
    ds = dataset.lower()
    if ds == 'mnist':
        return get_mnist_loaders(batch_size)
    if ds == 'cifar10':
        return get_cifar10_loaders(batch_size)
    if ds == 'imagenet':
        return get_imagenet_loaders(batch_size, imagenet_root, model_name)
    raise ValueError(f"Unsupported dataset: {dataset}")

# Model Factory
def get_model(num_classes, dataset, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds, name = dataset.lower(), model_name.lower()
    use_pre = (ds == 'imagenet')

    if name == 'resnet50':
        w = ResNet50_Weights.IMAGENET1K_V2 if use_pre else None
        m = resnet50(weights=w)
        if ds == 'mnist':
            m.conv1   = nn.Conv2d(3,64,3,1,1,bias=False)
            m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == 'densenet121':
        w = DenseNet121_Weights.IMAGENET1K_V1 if use_pre else None
        m = densenet121(weights=w)
        # if ds == 'mnist':
        if ds in ['mnist','cifar10']:
            # m.features.conv0 = nn.Conv2d(3,64,3,1,1,bias=False)
            m.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            m.features.pool0 = nn.Identity()
        m.classifier = nn.Linear(1024, num_classes)

    elif name == 'vgg16':
        w = VGG16_Weights.IMAGENET1K_V1 if use_pre else None
        m = vgg16(weights=w)
        if ds in ['mnist','cifar10']:
            m.avgpool    = nn.AdaptiveAvgPool2d((1,1))
            m.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return m.to(device)

# Training Loop
def train_one_epoch(model, loader, optimizer, criterion, args, atk_obj=None):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    batch_count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if args.defense.lower() == 'at' and atk_obj:
            # Adversarial Training
            if isinstance(atk_obj, RayS):
                out = atk_obj(x, y, query_limit=QUERY_BUDGET)
                adv = out[0] if out else None
            else:
                out = atk_obj.attack_untargeted(
                    x, y,
                    alpha=ATTACK_CONFIGS[args.train_attack]['alpha'],
                    beta=ATTACK_CONFIGS[args.train_attack]['beta'],
                    iterations=ATTACK_CONFIGS[args.train_attack]['iterations'],
                    query_limit=QUERY_BUDGET
                )
                adv = out[0] if out and not isinstance(out, str) else None

            if adv is not None:
                loss = 0.5 * (criterion(model(x), y) + criterion(model(adv), y))
            else:
                loss = criterion(model(x), y)

        elif args.defense.lower() == 'rst' and atk_obj:
            # Robust Self-Training
            with torch.no_grad():
                pseudo = model(x).argmax(1)
            if isinstance(atk_obj, RayS):
                out = atk_obj(x, pseudo, query_limit=QUERY_BUDGET)
                adv = out[0] if out else None
            else:
                out = atk_obj.attack_untargeted(
                    x, pseudo,
                    alpha=ATTACK_CONFIGS[args.train_attack]['alpha'],
                    beta=ATTACK_CONFIGS[args.train_attack]['beta'],
                    iterations=ATTACK_CONFIGS[args.train_attack]['iterations'],
                    query_limit=QUERY_BUDGET
                )
                adv = out[0] if out and not isinstance(out, str) else None

            loss = criterion(model(x), y)
            if adv is not None:
                loss += 0.5 * criterion(model(adv), pseudo)

        else:
            # Standard Training
            loss = criterion(model(x), y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    # return the average training loss for this epoch
    return total_loss / batch_count

def train_model(args):
    set_seed(args.seed)

    # pass model_name so ImageNet transforms match
    train_loader, val_loader = get_dataloaders(
        args.dataset, args.batch_size, args.imagenet_root, args.model
    )

    num_cls = 10 if args.dataset.lower() != 'imagenet' else 1000
    model   = get_model(num_cls, args.dataset, args.model)

    # instantiate attacker for defense
    general = GeneralTorchModel(model, num_cls, *NORM_STATS[args.dataset])
    if args.train_attack == 'RayS':
        atk_obj = RayS(
            general,
            ds_mean=NORM_STATS[args.dataset][0],
            ds_std =NORM_STATS[args.dataset][1],
            order=np.inf,
            epsilon=EPSILONS[args.dataset],
            early_stopping=True
        )
    elif args.train_attack == 'OPT':
        atk_obj = OPT_attack_lf(
            general,
            ds_mean=NORM_STATS[args.dataset][0],
            ds_std =NORM_STATS[args.dataset][1]
        )
    else:
        atk_obj = OPT_attack_sign_SGD_lf(
            general,
            ds_mean=NORM_STATS[args.dataset][0],
            ds_std =NORM_STATS[args.dataset][1],
            k=ATTACK_CONFIGS['Sign-OPT']['k']
        )

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    stopper   = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    best_acc  = 0.0

    for epoch in range(1, args.epochs+1):
        # train_one_epoch(model, train_loader, optimizer, criterion, args, atk_obj)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args, atk_obj)
        # scheduler.step()

        # validation
        device = next(model.parameters()).device
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += preds.eq(y).sum().item()
                total   += y.size(0)
        acc = correct / total
        # print(f"Epoch {epoch} | Val Acc: {acc*100:.2f}%")
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            ckpt = f"models/{args.dataset.lower()}_{args.model.lower()}_{args.defense.lower()}.pth"
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best model to {ckpt}")

        stopper(1-acc)
        scheduler.step()
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

# Evaluation Loop
def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    num_cls = 10 if args.dataset.lower() != 'imagenet' else 1000

    # iterate each model in your list
    for model_name, defense, train_atk in zip(args.model_list, args.defense_list, args.train_attack_list):
        # reload loaders with correct model_name (for ImageNet transforms)
        _, val_loader = get_dataloaders(
            args.dataset, args.batch_size, args.imagenet_root, model_name
        )

        model = get_model(num_cls, args.dataset, model_name)
        ckpt  = f"models/{args.dataset.lower()}_{model_name.lower()}_{defense.lower()}.pth"
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        general = GeneralTorchModel(model, num_cls, *NORM_STATS[args.dataset])
        attackers = [
            ('RayS',     RayS(general, ds_mean=NORM_STATS[args.dataset][0], ds_std=NORM_STATS[args.dataset][1], order=np.inf, epsilon=EPSILONS[args.dataset], early_stopping=True)),
            ('OPT',      OPT_attack_lf(general, ds_mean=NORM_STATS[args.dataset][0], ds_std=NORM_STATS[args.dataset][1])),
            ('Sign-OPT', OPT_attack_sign_SGD_lf(general, ds_mean=NORM_STATS[args.dataset][0], ds_std=NORM_STATS[args.dataset][1], k=ATTACK_CONFIGS['Sign-OPT']['k']))
        ]

        for atk_name, atk_obj in attackers:
            total = valid = success = 0
            queries_sum = 0

            for x, y in tqdm(val_loader, desc=f"{model_name}-{atk_name}"):
                x, y = x.to(device), y.to(device)
                for i in range(x.size(0)):
                    xi, yi = x[i:i+1], y[i:i+1]
                    if model(xi).argmax(1).item() != yi.item():
                        continue
                    total += 1

                    try:
                        if atk_name == 'RayS':
                            out = atk_obj(xi, yi, query_limit=QUERY_BUDGET)
                            if not out: 
                                continue
                            adv, qc, _, raw_succ = out
                        else:
                            out = atk_obj.attack_untargeted(
                                xi, yi,
                                alpha=ATTACK_CONFIGS[atk_name]['alpha'],
                                beta=ATTACK_CONFIGS[atk_name]['beta'],
                                iterations=ATTACK_CONFIGS[atk_name]['iterations'],
                                query_limit=QUERY_BUDGET
                            )
                            if not out or isinstance(out, str) or out[0] is None:
                                continue
                            adv = out[0]
                            qc  = out[2] if atk_name=='OPT' else out[3]
                            raw_succ = (model(adv).argmax(1).item() != yi.item())

                        valid += 1
                        l2, linf = calc_perturbation(
                            (xi * NORM_STATS[args.dataset][1]) + NORM_STATS[args.dataset][0],
                            adv
                        )
                        if raw_succ and linf <= EPSILONS[args.dataset] and qc <= QUERY_BUDGET:
                            success += 1
                        queries_sum += qc

                    except Exception as e:
                        print(f"Error on sample {total}: {e}")

            if total > 0:
                results.append([
                    model_name, 
                    defense, 
                    train_atk, 
                    atk_name, 
                    success/total, 
                    queries_sum / valid if valid else 0
                ])

    df = pd.DataFrame(
        results,
        columns=[ 'Model','Defense','TrainAtk','Attack','SuccessRate','AvgQueries' ]
    )
    os.makedirs('results', exist_ok=True)
    path = f"results/{args.dataset.lower()}_hardlabel_eval.csv"
    df.to_csv(path, index=False)
    print(f"Saved evaluation to {path}")

# Main entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate robust models")
    parser.add_argument('--mode', choices=['train','eval'], required=True)
    parser.add_argument('--dataset',     choices=['MNIST','CIFAR10','ImageNet'], required=True)
    parser.add_argument('--model',       choices=['resnet50','densenet121','vgg16'], required=True)
    parser.add_argument('--model_list',  nargs='+', default=['resnet50'])
    parser.add_argument('--defense',     choices=['standard','at','rst'], default='standard')
    parser.add_argument('--defense_list', nargs='+', default=['standard'])
    parser.add_argument('--train_attack',     choices=['OPT','Sign-OPT','RayS'], default='OPT')
    parser.add_argument('--train_attack_list', nargs='+', default=['OPT'])
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--lr',         type=float, default=0.1)
    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--patience',   type=int,   default=50)
    parser.add_argument('--min_delta',  type=float, default=1e-4)
    parser.add_argument('--imagenet_root', type=str, default='./data/imagenet')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    else:
        evaluate_model(args)