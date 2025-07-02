import os
import random
import argparse
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescentPyTorch, CarliniL2Method
from art.estimators.classification import PyTorchClassifier

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

# Drop-Max Layer for MIRST-DM
class DropMaxLayer(nn.Module):
    def __init__(self):
        super(DropMaxLayer, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_reshaped = x.view(batch_size, channels, -1)
        max_values, max_indices = torch.max(x_reshaped, dim=2, keepdim=True)
        mask = torch.ones_like(x_reshaped).scatter_(2, max_indices, 0)
        x_dropped = x_reshaped * mask
        x_dropped = x_dropped.view(batch_size, channels, height, width)
        return x_dropped

# MIRST-DM Model
class MIRSTDMModel(nn.Module):
    def __init__(self, num_classes=10, dataset='cifar10'):
        super(MIRSTDMModel, self).__init__()
        weights = ResNet50_Weights.ImageNet1K_V2 if dataset.lower() == 'imagenet' else None
        self.resnet = resnet50(weights=weights)
        if dataset.lower() == 'mnist':
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet.maxpool = nn.Identity()
        self.drop_max = DropMaxLayer()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.drop_max(x)  # Apply Drop-Max after layer3
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

# Set random seed
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Data Loaders
def get_cifar10_loaders(batch_size=256):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    mean, std = tuple(mean), tuple(std)
    return trainloader, testloader, mean, std

def get_mnist_loaders(batch_size=128):
    mean = [0.1307, 0.1307, 0.1307]
    std = [0.3081, 0.3081, 0.3081]
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    mean, std = tuple(mean), tuple(std)
    return trainloader, testloader, mean, std

def get_imagenet_loaders(batch_size=128, imagenet_root='./data/imagenet'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
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
    val_dir = os.path.join(imagenet_root, 'ILSVRC2012_img_val')
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    valset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    mean, std = tuple(mean), tuple(std)
    return trainloader, valloader, mean, std

def get_dataloaders(dataset, batch_size=128, imagenet_root='./data/imagenet'):
    ds = dataset.lower()
    if ds == 'cifar10':
        return get_cifar10_loaders(batch_size)
    elif ds == 'mnist':
        return get_mnist_loaders(batch_size)
    elif ds == 'imagenet':
        return get_imagenet_loaders(batch_size, imagenet_root)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

# Model Builder
def get_model(model_name='ResNet50', num_classes=10, dataset='CIFAR10', defense='None'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = dataset.lower()
    if model_name != 'ResNet50':
        raise NotImplementedError(f"Model {model_name} not supported.")
    if defense.lower() == 'mirst-dm':
        return MIRSTDMModel(num_classes=num_classes, dataset=dataset).to(device)
    weights = ResNet50_Weights.ImageNet1K_V2 if ds == 'imagenet' else None
    model = resnet50(weights=weights)
    if ds == 'mnist':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# Attack and Utility Functions
def mixup(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def psnr(a, b, mean_t, std_t):
    a = a * std_t + mean_t
    b = b * std_t + mean_t
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 20 * math.log10(1 / math.sqrt(mse.item()))

def get_attack(name: str, clf, eps=8/255, iters=7):
    name = name.lower()
    if name == 'fgsm':
        return FastGradientMethod(estimator=clf, eps=eps)
    if name == 'pgd':
        return ProjectedGradientDescentPyTorch(estimator=clf, eps=eps, eps_step=eps/4, max_iter=iters)
    if name in ('cw', 'c&w'):
        return CarliniL2Method(estimator=clf, max_iter=iters*10, learning_rate=0.01)
    raise ValueError(f"Unknown attack {name}")

# Training and Evaluation
def train_one_epoch(model, loader, optimizer, criterion, clf, args, base_attack):
    model.train()
    device = next(model.parameters()).device
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if args.dataset.lower() in ('cifar10', 'mnist'):
            x, y_a, y_b, lam = mixup(x, y)
        else:
            y_a, y_b, lam = y, y, 1.0
        defense = args.defense.lower()
        if defense == 'none':
            x_in, targets = x, (y_a, y_b, lam)
        elif defense == 'at':
            atk = base_attack or get_attack(args.train_attack, clf, iters=args.pgd_steps)
            x_adv = atk.generate(x=x.cpu().numpy())
            x_in = torch.from_numpy(x_adv).to(device)
            targets = (y_a, y_b, lam)
        elif defense == 'rst':
            with torch.no_grad():
                pseudo = model(x).argmax(dim=1)
            atk = base_attack or get_attack(args.train_attack, clf, iters=args.pgd_steps)
            x_adv = atk.generate(x=x.cpu().numpy(), y=pseudo.cpu().numpy())
            x_in, targets = torch.from_numpy(x_adv).to(device), pseudo
        elif defense == 'mirst-dm':
            x_in, targets = x, y  # Rely on Drop-Max layer, not the adversarial perturbation
        else:
            raise ValueError(f"Unknown defense {args.defense}")
        outputs = model(x_in)
        if defense in ('none', 'at'):
            loss = criterion(outputs, targets[0]) * targets[2] + criterion(outputs, targets[1]) * (1 - targets[2])
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def train_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, mean, std = get_dataloaders(args.dataset, args.batch_size, args.imagenet_root)
    model = get_model('ResNet50', num_classes=10 if args.dataset.lower() != 'imagenet' else 1000, dataset=args.dataset, defense=args.defense)
    clf = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=args.lr),
        input_shape=next(iter(train_loader))[0].shape[1:],
        nb_classes=10 if args.dataset.lower() != 'imagenet' else 1000,
        preprocessing=(tuple(mean), tuple(std))
    )
    base_attack = None if args.train_attack.lower() == 'mix' else get_attack(args.train_attack, clf, iters=args.pgd_steps)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, clf, args, base_attack)
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch:03d} | Val Acc: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            save_f = f"models/{args.dataset.lower()}_{args.defense.lower()}_{args.train_attack.lower()}.pth"
            os.makedirs(os.path.dirname(save_f), exist_ok=True)
            torch.save(model.state_dict(), save_f)
            print(f"Saved best model to {save_f}")
        stopper(1 - acc)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

def plot_results(dataset, results):
    defenses = sorted(set(r[0] for r in results))
    attacks = sorted(set(r[2] for r in results))
    clean_accs = {d: 0 for d in defenses}
    robust_accs = {d: {a: 0 for a in attacks} for d in defenses}
    for r in results:
        defense, _, atk, clean_acc, robust_acc, _ = r
        clean_accs[defense] = clean_acc
        robust_accs[defense][atk] = robust_acc
    plt.figure(figsize=(10, 5))
    for defense in defenses:
        plt.plot(attacks, [robust_accs[defense][a] for a in attacks], marker='o', label=f"{defense} Robust Acc")
        plt.axhline(clean_accs[defense], linestyle='--', label=f"{defense} Clean Acc")
    plt.xlabel('Attack')
    plt.ylabel('Accuracy')
    plt.title(f'Performance Comparison on {dataset}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{dataset.lower()}_comparison.png')
    plt.close()

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, mean, std = get_dataloaders(args.dataset, args.batch_size, args.imagenet_root)
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    results = []
    for defense, train_atk in zip(args.defense_list, args.train_attack_list):
        model = get_model('ResNet50', num_classes=10 if args.dataset.lower() != 'imagenet' else 1000, dataset=args.dataset, defense=defense)
        path = f"models/{args.dataset.lower()}_{defense.lower()}_{train_atk.lower()}.pth"
        if not os.path.exists(path):
            print(f"Model not found: {path}")
            continue
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        clean_acc = correct / total
        clf = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=next(iter(val_loader))[0].shape[1:],
            nb_classes=10 if args.dataset.lower() != 'imagenet' else 1000,
            preprocessing=(tuple(mean), tuple(std))
        )
        eps_fgsm = 16/255 if args.dataset.lower() == 'imagenet' else 8/255
        eval_attacks = {
            'FGSM': FastGradientMethod(estimator=clf, eps=eps_fgsm),
            'PGD40': ProjectedGradientDescentPyTorch(estimator=clf, eps=8/255, eps_step=2/255, max_iter=40),
            'CW100': CarliniL2Method(estimator=clf, max_iter=100, learning_rate=0.01)
        }
        for atk_name, atk in eval_attacks.items():
            succ, psnr_sum = 0, 0.0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                adv = atk.generate(x=x.cpu().numpy())
                adv_t = torch.from_numpy(adv).to(device)
                preds = model(adv_t).argmax(dim=1).cpu().numpy()
                succ += (preds != y.cpu().numpy()).sum()
                for i in range(x.size(0)):
                    psnr_sum += psnr(x[i].cpu(), adv_t[i].cpu(), mean_t, std_t)
            robust_acc = 1 - succ / total
            avg_psnr = psnr_sum / total
            results.append([defense, train_atk, atk_name, clean_acc, robust_acc, avg_psnr])
            print(f"{defense}-{train_atk} | {atk_name} | Clean: {clean_acc:.3f} | Robust: {robust_acc:.3f} | PSNR: {avg_psnr:.2f}")
    os.makedirs('results', exist_ok=True)
    out_f = f"results/{args.dataset.lower()}_comparison.csv"
    with open(out_f, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Defense', 'TrainAtk', 'EvalAtk', 'CleanAcc', 'RobustAcc', 'PSNR'])
        writer.writerows(results)
    print(f"Saved comparison table to {out_f}")
    plot_results(args.dataset, results)
    print(f"Saved comparison plot to results/{args.dataset.lower()}_comparison.png")

# Main runner
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/Evaluate ResNet50 defenses")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    train_p = subparsers.add_parser('train', help='Train model')
    train_p.add_argument('--dataset', choices=['CIFAR10', 'MNIST', 'ImageNet'], default='CIFAR10')
    train_p.add_argument('--defense', choices=['None', 'AT', 'RST', 'MIRST-DM'], default='None')
    train_p.add_argument('--train_attack', choices=['FGSM', 'PGD', 'CW', 'MIX'], default='PGD')
    train_p.add_argument('--epochs', type=int, default=300)
    train_p.add_argument('--batch_size', type=int, default=256)
    train_p.add_argument('--lr', type=float, default=0.1)
    train_p.add_argument('--pgd_steps', type=int, default=7)
    train_p.add_argument('--patience', type=int, default=20)
    train_p.add_argument('--min_delta', type=float, default=1e-4)
    train_p.add_argument('--seed', type=int, default=0)
    train_p.add_argument('--imagenet_root', default='./data/imagenet')
    eval_p = subparsers.add_parser('eval', help='Evaluate defenses')
    eval_p.add_argument('--dataset', choices=['CIFAR10', 'MNIST', 'ImageNet'], default='CIFAR10')
    eval_p.add_argument('--batch_size', type=int, default=256)
    eval_p.add_argument('--imagenet_root', default='./data/imagenet')
    eval_p.add_argument('--defense_list', nargs='+', default=['None', 'AT', 'RST', 'MIRST-DM'])
    eval_p.add_argument('--train_attack_list', nargs='+', default=['PGD', 'PGD', 'PGD', 'PGD'])
    args = parser.parse_args()
    if args.mode == 'train':
        train_model(args)
    else:
        evaluate_model(args)
