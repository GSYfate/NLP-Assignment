import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import cuda
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration

from dataset import ReverseDataset
from tqdm import tqdm

from utils import train, validate, load_model_and_tokenizer


def main(args):
    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    print("Loading Model:", args.model_name)
    model, tokenizer, config, device = load_model_and_tokenizer(args.model_name)

    print("Loading Dataset:", args.dataset_path)
    train_data= load_dataset(args.dataset_path, split='train[:80%]')
    val_data = load_dataset(args.dataset_path, split='train[80%:]')
    test_data = load_dataset(args.dataset_path, split='train[:1%]')
    train_len, val_len = len(train_data), len(val_data)
    print(f"Train: {train_len}, Val: {val_len}")
   
    train_dataset = ReverseDataset(tokenizer,train_data, args.max_length)
    # print(train_dataset[0])
    val_dataset = ReverseDataset(tokenizer,val_data, args.max_length)
    test_dataset = ReverseDataset(tokenizer,test_data, args.max_length)
    train_params = {
        'batch_size': args.train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    val_params = {
        'batch_size': args.val_batch_size,
        'shuffle': False,
        'num_workers': 0
    }
    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)
    test_loader = DataLoader(test_dataset, **val_params)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    print(f"Initiating Fine-Tuning for {args.model_name} the mo on {args.dataset_path} dataset")

    # Training
    for epoch in range(args.epoch_nums):
        train(epoch, tokenizer, model, device, train_loader, optimizer, args.output_dir)
        predictions_rev, actuals_rev = validate(tokenizer, model, device, test_loader, args.max_length)
        print('Generating Text')
        predictions = [text[::-1] for text in predictions_rev]
        actuals = [text[::-1] for text in actuals_rev]
        final_df = pd.DataFrame({'Actual Text': actuals, 'Generated Text (Reversed)': predictions,
                                 'Actual Text (Reversed)': actuals_rev, 'Generated Text': predictions_rev})
        final_df.to_csv('test_' + str(epoch) + '.csv', index=False)
        

    # Eval
    predictions, actuals = validate(tokenizer, model, device, val_loader, args.max_length)
    

    # Save Model and predictions
    sources = [text[::-1] for text in actuals]
    df = pd.DataFrame({'English Text':sources, 'Generated Text':predictions, 'Actual Reversed Text':actuals})
    df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    print('Output Files generated for review')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)

    # Dataset params
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--max_length", type=int, default=128)

    # Training params
    parser.add_argument("--epoch_nums", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    
    args = parser.parse_args()

    main(args)