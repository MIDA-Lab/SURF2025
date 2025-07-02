# make_summary.py — write a separate summary CSV for every per-image file
import argparse, glob, os, pandas as pd

def collect_metrics(df):
    succ = df.success == 1
    return {
        "Success Rate" : succ.mean(),
        "Avg Queries"  : df.loc[succ, "queries"].mean(),
        "Med Queries"  : df.loc[succ, "queries"].median(),
        "Avg L2"       : df.loc[succ, "perturb_l2"].mean(),
        "Avg Linf"     : df.loc[succ, "perturb_linf"].mean(),
        "Avg PSNR"     : df.loc[succ, "psnr"].mean(),
        "N Samples"    : len(df)
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="results_10k")
    args = p.parse_args()

    for csv in glob.glob(os.path.join(args.input_dir, "*_10k.csv")):
        base       = os.path.basename(csv).replace("_10k.csv", "")
        ds, atk    = base.split("_", 1)
        df         = pd.read_csv(csv)
        metrics    = collect_metrics(df)
        metrics["Dataset"] = ds
        metrics["Attack"]  = atk

        out = os.path.join(args.input_dir, f"{base}_summary_10k.csv")
        pd.DataFrame([metrics]).to_csv(out, index=False)
        print("→ saved", out)

if __name__ == "__main__":
    main()