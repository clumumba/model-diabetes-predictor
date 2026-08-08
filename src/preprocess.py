import pandas as pd
import yaml, os
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

while not (_root / "params.yml").exists() and _root != _root.parent:
    _root = _root.parent

with open(_root / "params.yml", 'r')as f:
    params = yaml.safe_load(f)["preprocess"]

def preprocess_data(data_path):
    data_file = _root / data_path
    df = pd.read_csv(data_file)
    df.info()

    os.makedirs(os.path.dirname(params["output"]), exist_ok=True)
    df.to_csv(params["output"], index=False, header=False)
    print(f"Preprocessed data saved to {params['output']}")

    return df

if __name__ == "__main__":
    preprocess_data(params['input'])
    