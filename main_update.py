# main.py
# Runs hard‐label black‐box attacks (OPT, Sign‐OPT, RayS‐Single) on a fixed set of correctly classified test images, using a single 10 000‐query budget, logging per-image distortion & queries, 
# clamping L∞ ≤ 0.3 for success, and saving all adversarial examples.

# main.py
import os
import math
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# official RayS wrapper + attacker
from general_torch_model import GeneralTorchModel
from RayS_Single import RayS

# Use the _lf variants for OPT and Sign-OPT
from OPT_attack_lf import OPT_attack_lf
from Sign_OPT_lf import OPT_attack_sign_SGD_lf

QUERY_BUDGET = 10000

EPSILONS = {
    "MNIST":   0.3,
    "CIFAR10": 0.031,
    "ImageNet":0.05
}

MAX_LINF = 0.3  # L∞ ≤ 0.3

NORM_STATS = {
    "MNIST":   ([0.1307]*3, [0.3081]*3),
    "CIFAR10": ([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010]),
    "ImageNet":([0.485,0.456,0.406], [0.229,0.224,0.225]),
}

ATTACK_CONFIGS = {
    'OPT':      {'alpha':0.2, 'beta':0.001, 'iterations':1500},
    'Sign-OPT': {'alpha':0.2, 'beta':0.001, 'iterations':1500, 'k':200},
    'RayS': {}    # RayS-Single only needs epsilon from EPSILONS
}

# ResNet model doesn't have a predict_label method, we need to wrap the model with a wrapper class
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict_label(self, x):
        with torch.no_grad():
            output = self.model(x)
            return output.argmax(1).item()

def get_transform(ds):
    mean, std = NORM_STATS[ds]
    if ds=="MNIST":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if ds=="CIFAR10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if ds=="ImageNet":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    raise ValueError(ds)

def get_dataset(ds):
    tf = get_transform(ds)
    if ds=="MNIST":
        return datasets.MNIST('./data', train=False, download=True, transform=tf)
    if ds=="CIFAR10":
        return datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    if ds=="ImageNet":
        return datasets.ImageFolder('./data/imagenet/val', transform=tf)
    raise ValueError(ds)

def get_raw_model(ds, device):
    if ds=="ImageNet":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        m = models.resnet50(weights=None, num_classes=10)
        m.conv1   = torch.nn.Conv2d(3,64,3,1,1,bias=False)
        m.maxpool = torch.nn.Identity()
        w = f"./weights/{ds.lower()}_resnet50_standard.pth"
        m.load_state_dict(torch.load(w, map_location=device))
    return m.eval().to(device)

def calc_perturbation(x, adv):
    if adv is None:
        return float('inf'), float('inf')
    d = (adv - x).view(-1)
    return float(d.norm(2).item()), float(d.abs().max().item())

def calc_psnr(x, adv):
    mse = torch.mean((x-adv)**2).item()
    return float('inf') if mse==0 else 20*math.log10(1.0/math.sqrt(mse))

def save_adv_example(x, adv, ds, atk, idx, outdir):
    os.makedirs(outdir, exist_ok=True)
    img = torch.cat([x.squeeze().cpu(), adv.squeeze().cpu()], dim=2)
    img = (img - img.min())/(img.max()-img.min())
    plt.imsave(os.path.join(outdir, f"{ds}_{atk}_ex_{idx}.png"),
               img.permute(1,2,0))

def main():
    p = argparse.ArgumentParser(description="Hard-label attacks @10k queries")
    p.add_argument('--attack',   choices=['OPT','Sign-OPT','RayS','ALL'], default='ALL')
    p.add_argument('--dataset',  choices=['MNIST','CIFAR10','ImageNet','ALL'], default='ALL')
    p.add_argument('--num_samples', type=int, default=100,
                   help="number of correctly classified test images")
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--output_dir',type=str, default='./results_10k')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_atks = ['OPT','Sign-OPT','RayS']
    attacks  = all_atks if args.attack=='ALL' else [args.attack]
    datasets = ['MNIST','CIFAR10','ImageNet'] if args.dataset=='ALL' else [args.dataset]

    summary = []
    for ds in datasets:
        full_ds   = get_dataset(ds)
        raw_model = get_raw_model(ds, DEVICE)

        mean, std = NORM_STATS[ds]
        n_cls = 10 if ds!='ImageNet' else 1000
        model = GeneralTorchModel(raw_model, n_class=n_cls, im_mean=None, im_std=None)

        idx_file = os.path.join(args.output_dir, f"{ds}_fixed_idx.npy")
        if os.path.exists(idx_file):
            sel = np.load(idx_file).tolist()
            print(f"Loaded fixed indices for {ds} ({len(sel)})")
        else:
            inds = list(range(len(full_ds)))
            random.Random(args.seed).shuffle(inds)
            sel = []
            for i in inds:
                if len(sel)>=args.num_samples: break
                x,y = full_ds[i][0].unsqueeze(0).to(DEVICE), full_ds[i][1]
                with torch.no_grad():
                    if raw_model(x).argmax(1).item()==y:
                        sel.append(i)
            if len(sel)<args.num_samples:
                raise RuntimeError(f"Only {len(sel)} correct, needed {args.num_samples}")
            np.save(idx_file, np.array(sel, dtype=np.int64))
            print(f"Saved fixed indices for {ds} → {idx_file}")

        subset = Subset(full_ds, sel)
        loader = DataLoader(subset, batch_size=1, shuffle=True, num_workers=2)
        print(f"\n=== {ds}: attacking {len(sel)} samples ===")

        for atk in attacks:
            cfg = ATTACK_CONFIGS[atk]
            eps = EPSILONS[ds]

            # --------- Attack instantiation ----------
            if atk=='OPT':
                wrapped_model = ModelWrapper(raw_model)  # Add wrapper for OPT,  wrapped_model = ModelWrapper(raw_model)
                atk_obj = OPT_attack_lf(wrapped_model)
            elif atk=='Sign-OPT':
                wrapped_model = ModelWrapper(raw_model) # Add wrapper for Sign-OPT, we need use wrapper, wrapped_model = ModelWrapper(raw_model)
                atk_obj = OPT_attack_sign_SGD_lf(wrapped_model, k=cfg['k'])
            else:  # RayS
                mean, std = NORM_STATS[ds]
                atk_obj = RayS(
                    model,
                    ds_mean=mean,
                    ds_std=std,
                    order=np.inf,
                    epsilon=eps,
                    early_stopping=True
                )

            print(f"--- {atk} on {ds} @ {QUERY_BUDGET} queries, ε={eps} ---")
            records = []

            for idx,(x,y) in enumerate(tqdm(loader, desc=f"{atk}@{ds}")):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                if atk == 'OPT':
                    res = atk_obj.attack_untargeted(
                        x, y,
                        alpha=cfg['alpha'],
                        beta=cfg['beta'],
                        iterations=cfg['iterations']
                    )
                    if isinstance(res, str) or res[0] is None:  # Check if res is "NA" string
                        continue
                    adv, g_theta, qc = res
                    if isinstance(adv, str):  # Additional check
                        continue
                    raw_success = (raw_model(adv).argmax(1).item() != y.item())

                elif atk == 'Sign-OPT':
                    res = atk_obj.attack_untargeted(
                        x, y,
                        alpha=cfg['alpha'],
                        beta=cfg['beta'],
                        iterations=cfg['iterations'],
                        query_limit=QUERY_BUDGET
                    )
                    if res[0] is None or not res[2]:
                        continue
                    adv, g_theta, success, qc, _ = res
                    raw_success = (raw_model(adv).argmax(1).item() != y.item())

                else:  # RayS‐Single
                    ret = atk_obj(x, y, query_limit=QUERY_BUDGET)
                    if ret is None:
                        continue
                    adv, qc, _, raw_success = ret
                    if adv is None:
                        continue

                raw_l2, raw_linf = calc_perturbation(x, adv)
                psnr = calc_psnr(x, adv)

                succ_flag = (
                    bool(raw_success)
                    and (raw_linf <= MAX_LINF)
                    and (qc <= QUERY_BUDGET)
                )

                records.append({
                    'idx':         idx,
                    'success':     int(succ_flag),
                    'queries':     qc,
                    'perturb_l2':  raw_l2,
                    'perturb_linf':raw_linf,
                    'psnr':        psnr
                })

                if succ_flag:
                    save_adv_example(
                        x, adv, ds, atk, idx,
                        os.path.join(args.output_dir, f"{ds}_{atk}")
                    )

            # per-image CSV
            df = pd.DataFrame(records)
            out_csv = os.path.join(args.output_dir, f"{ds}_{atk}_10k.csv")
            df.to_csv(out_csv, index=False)
            print(f"  → Saved {out_csv}")

            summary.append({
                'Dataset':     ds,
                'Attack':      atk,
                'Success Rate':df.success.mean(),
                'Avg Queries': df.queries.mean(),
                'Avg L2':      df.perturb_l2.mean(),
                'Avg Linf':    df.perturb_linf.mean(),
                'Avg PSNR':    df.psnr.mean()
            })

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(args.output_dir, "attack_summary_10k_table.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary → {summary_csv}")

    plt.figure(figsize=(8,5))
    for atk in attacks:
        sub = summary_df[summary_df.Attack == atk]
        plt.bar(sub.Dataset, sub['Success Rate'], label=atk)
    plt.ylabel("Success Rate")
    plt.title("Success @10k queries, L∞≤0.3")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "attack_success_10k_barplot.png"))
    print("Saved barplot.")

if __name__=="__main__":
    main()
