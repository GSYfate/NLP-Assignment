import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ReverseDataset
from datasets import load_dataset
from tqdm import tqdm

from utils import validate, load_model_and_tokenizer

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main(args):
    # Initialize
    model, tokenizer, config, device = load_model_and_tokenizer(args.model_name)

    # Load checkpoint
    model = load_checkpoint(args.checkpoint_path, model).to(device)

    val_data = load_dataset(args.dataset_path, split='train[80%:]')
    val_dataset = ReverseDataset(tokenizer, val_data, args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)


    predictions, actuals = validate(tokenizer, model, device, val_loader, args.max_length)
    
    sources = [text[::-1] for text in actuals]
    df = pd.DataFrame({'English Text': sources, 'Generated Text': predictions, 'Actual Reversed Text': actuals})
    df.to_csv(os.path.join(args.output_dir, "predictions.csv"))
    print('Output Files generated for review')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
