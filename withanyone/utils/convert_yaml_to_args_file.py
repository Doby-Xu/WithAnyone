

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--yaml", type=str, required=True)
parser.add_argument("--arg", type=str, required=True)
args = parser.parse_args()


with open(args.yaml, "r") as f:
    data = yaml.safe_load(f)

with open(args.arg, "w") as f:
    for k, v in data.items():
        if isinstance(v, list):
            v = list(map(str, v))
            v = " ".join(v)
        if v is None:
            continue
        print(f"--{k} {v}", end=" ", file=f)
