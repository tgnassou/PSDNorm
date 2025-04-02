from wilds import get_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="camelyon17")
args = parser.parse_args()

dataset = get_dataset(dataset=args.dataset, download=True)
