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
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from general_torch_model import GeneralTorchModel
from RayS_Single import RayS
from OPT_attack import OPTAttack
from Sign_OPT import SignOPT

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
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

# Data loaders
def get_cifar10_loaders(batch_size=256):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader, tuple(mean), tuple(std)

def get_mnist_loaders(batch_size=128):
    mean = [0.1307, 0.1307, 0.1307]
    std  = [0.3081, 0.3081, 0.3081]
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader, tuple(mean), tuple(std)

def get_imagenet_loaders(batch_size=128, imagenet_root='./data/imagenet'):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dir = os.path.join(imagenet_root, 'ILSVRC2012_img_train')
    val_dir   = os.path.join(imagenet_root, 'ILSVRC2012_img_val')
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    valset   = torchvision.datasets.ImageFolder(val_dir,   transform=val_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, valloader, tuple(mean), tuple(std)

def get_dataloaders(dataset, batch_size=128, imagenet_root='./data/imagenet'):
    ds = dataset.lower()
    if ds == 'cifar10':
        return get_cifar10_loaders(batch_size)
    if ds == 'mnist':
        return get_mnist_loaders(batch_size)
    if ds == 'imagenet':
        return get_imagenet_loaders(batch_size, imagenet_root)
    raise ValueError(f"Dataset {dataset} not supported.")

# Model builder
def get_model(num_classes, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet50_Weights.ImageNet1K_V2 if dataset.lower()=='imagenet' else None
    model = resnet50(weights=weights)
    if dataset.lower()=='mnist':
        model.conv1   = nn.Conv2d(3,64,3,1,1,bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# Attack evaluation constants
QUERY_BUDGET = 10000
EPSILONS     = {"MNIST":0.3, "CIFAR10":0.031, "ImageNet":0.05}
MAX_LINF     = 0.3
ATTACK_CONFIGS = {
    'OPT':         {'alpha':0.2,'beta':0.001,'iterations':2200},
    'Sign-OPT':    {'alpha':0.2,'beta':0.001,'iterations':2200,'k':200},
    'RayS-Single': {}
}

def calc_perturbation(x, adv):
    """Return (L2, Linf)."""
    if adv is None:
        return float('inf'), float('inf')
    d = (adv - x).view(-1)
    return float(d.norm(2).item()), float(d.abs().max().item())

# Training loop with AT and RST

def train_one_epoch(model, loader, optimizer, criterion, args, atk_obj=None):
    model.train()
    device = next(model.parameters()).device
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if args.defense.lower() == 'at' and atk_obj is not None:
            # Adversarial Training
            if isinstance(atk_obj, RayS):
                adv, _, _, _ = atk_obj(x, y, query_limit=QUERY_BUDGET)
            else:
                res = atk_obj.attack(
                    x, y,
                    alpha=ATTACK_CONFIGS[args.train_attack]['alpha'],
                    beta=ATTACK_CONFIGS[args.train_attack]['beta'],
                    iterations=ATTACK_CONFIGS[args.train_attack]['iterations'],
                    query_limit=QUERY_BUDGET
                )
                adv = res[0] if res is not None else None

            if adv is not None:
                out_c = model(x)
                out_a = model(adv)
                loss = 0.5 * (criterion(out_c, y) + criterion(out_a, y))
            else:
                loss = criterion(model(x), y)

        elif args.defense.lower() == 'rst' and atk_obj is not None:
            # Robust Self-Training
            with torch.no_grad():
                pseudo = model(x).argmax(1)
            if isinstance(atk_obj, RayS):
                adv, _, _, _ = atk_obj(x, pseudo, query_limit=QUERY_BUDGET)
            else:
                res = atk_obj.attack(
                    x, pseudo,
                    alpha=ATTACK_CONFIGS[args.train_attack]['alpha'],
                    beta=ATTACK_CONFIGS[args.train_attack]['beta'],
                    iterations=ATTACK_CONFIGS[args.train_attack]['iterations'],
                    query_limit=QUERY_BUDGET
                )
                adv = res[0] if res is not None else None

            loss = criterion(model(x), y)
            if adv is not None:
                out_a = model(adv)
                loss += 0.5 * criterion(out_a, pseudo)

        else:
            # Standard Training
            loss = criterion(model(x), y)

        loss.backward()
        optimizer.step()

# Full training

def train_model(args):
    set_seed(args.seed)
    train_loader, val_loader, mean, std = get_dataloaders(args.dataset, args.batch_size, args.imagenet_root)
    num_cls = 10 if args.dataset.lower()!='imagenet' else 1000
    model   = get_model(num_cls, args.dataset)

    # choose hard-label attacker for defense
    general = GeneralTorchModel(model, num_cls, mean, std)
    if args.train_attack == 'RayS-Single':
        atk_obj = RayS(general, order=np.inf, epsilon=EPSILONS[args.dataset], early_stopping=True)
    elif args.train_attack == 'OPT':
        atk_obj = OPTAttack(general)
    elif args.train_attack == 'Sign-OPT':
        atk_obj = SignOPT(general)
    else:
        atk_obj = None

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    stopper   = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    best_acc  = 0.0

    for epoch in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, optimizer, criterion, args, atk_obj)
        scheduler.step()
        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                preds = model(x).argmax(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch} | Val Acc: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            ckpt = f"models/{args.dataset.lower()}_{args.defense.lower()}_{args.train_attack.lower()}.pth"
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best model to {ckpt}")
        stopper(1 - acc)
        if stopper.early_stop:
            print("Early stopping.")
            break

# Hard-label evaluation

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, mean, std = get_dataloaders(args.dataset, args.batch_size, args.imagenet_root)
    results = []
    for defense, train_atk in zip(args.defense_list, args.train_attack_list):
        num_cls = 10 if args.dataset.lower()!='imagenet' else 1000
        model   = get_model(num_cls, args.dataset)
        ckpt    = f"models/{args.dataset.lower()}_{defense.lower()}_{train_atk.lower()}.pth"
        if not os.path.exists(ckpt): print(f"Not found {ckpt}"); continue
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
        general = GeneralTorchModel(model, num_cls, mean, std)
        attackers = [
            ('RayS-Single', RayS(general, order=np.inf, epsilon=EPSILONS[args.dataset], early_stopping=True)),
            ('OPT', OPTAttack(general)),
            ('Sign-OPT', SignOPT(general)),
        ]
        for atk_name, atk_obj in attackers:
            total= len(val_loader.dataset); succ=0; qsum=0
            for x, y in tqdm(val_loader, desc=f"Eval {defense}-{atk_name}"):
                x, y = x.to(device), y.to(device)
                if atk_name == 'RayS-Single':
                    adv, qc, _, raw_succ = atk_obj(x, y, query_limit=QUERY_BUDGET)
                else:
                    res = atk_obj.attack(x, y,
                                         alpha=ATTACK_CONFIGS[atk_name]['alpha'],
                                         beta=ATTACK_CONFIGS[atk_name]['beta'],
                                         iterations=ATTACK_CONFIGS[atk_name]['iterations'],
                                         k=ATTACK_CONFIGS[atk_name].get('k'))
                    if res is None: continue
                    adv, qc = res; raw_succ = (model(adv).argmax(1).item()!=y.item())
                if adv is None: continue
                _, linf = calc_perturbation(x, adv)
                ok = bool(raw_succ) and (linf<=MAX_LINF) and (qc<=QUERY_BUDGET)
                succ += int(ok); qsum += qc
            results.append([defense, train_atk, atk_name, succ/total, qsum/total])
            print(f"{defense}-{train_atk}|{atk_name}|Succ:{succ/total:.3f}|AvgQ:{qsum/total:.1f}")
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results, columns=['Defense','TrainAtk','Attack','SuccRate','AvgQ'])
    out = f"results/{args.dataset.lower()}_hardlabel_eval.csv"
    df.to_csv(out, index=False)
    print(f"Saved eval → {out}")

# Main
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    t = sub.add_parser('train')
    t.add_argument('--dataset', choices=['CIFAR10','MNIST','ImageNet'], default='CIFAR10')
    t.add_argument('--defense',  choices=['None','AT','RST'], default='None')
    t.add_argument('--train_attack', choices=['RayS-Single','OPT','Sign-OPT'], default='OPT')
    t.add_argument('--epochs', type=int, default=300)
    t.add_argument('--batch_size', type=int, default=256)
    t.add_argument('--lr', type=float, default=0.1)
    t.add_argument('--patience', type=int, default=20)
    t.add_argument('--min_delta', type=float, default=1e-4)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--imagenet_root', default='./data/imagenet')
    e = sub.add_parser('eval')
    e.add_argument('--dataset', choices=['CIFAR10','MNIST','ImageNet'], default='CIFAR10')
    e.add_argument('--batch_size', type=int, default=256)
    e.add_argument('--imagenet_root', default='./data/imagenet')
    e.add_argument('--defense_list', nargs='+', default=['None','AT','RST'])
    e.add_argument('--train_attack_list', nargs='+', default=['OPT','OPT','OPT'])
    args = parser.parse_args()
    if args.mode == 'train': train_model(args)
    else: evaluate_model(args)