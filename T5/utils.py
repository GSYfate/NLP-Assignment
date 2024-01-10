import torch
import os
from torch import cuda
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration

def load_model_and_tokenizer(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()
    print("device: ", device, "  device count: ",device_count)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    model = model.to(device)
    return model, tokenizer, config, device

def train(epoch, tokenizer, model, device, loader, optimizer, output_dir):
    model.train()
    total_loss = 0
    num_batches = 0 
    progress_bar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
    for i, data in enumerate(progress_bar):
        y = data['labels'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        total_loss += loss.item() 
        num_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  

    avg_loss = total_loss / num_batches
    print(f'Epoch {epoch}, Average Loss: {avg_loss}')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'average_loss': avg_loss,
    }
    torch.save(checkpoint, os.path.join(output_dir, f'model_checkpoint_epoch_{epoch}.pt'))

def validate(tokenizer, model, device, loader, max_length):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Validation", leave=False)
        for i, data in enumerate(progress_bar):
            y = data['labels'].to(device, dtype = torch.long)
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids, attention_mask = mask, max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )

            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            targets = [tokenizer.decode(t, skip_special_tokens=True) for t in y]
            if i%100==0:
                print(f'Completed {i}')

            predictions.extend(preds)
            actuals.extend(targets)
    return predictions, actuals
