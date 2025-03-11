'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model, add_peft_configuration
from peft import TaskType, PeftConfig, PeftModel
from types import SimpleNamespace
from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args, lora_config=None):
    super().__init__()
    
    if lora_config:
      # Use LoRA for fine-tuning
      self.gpt = AutoModelForCausalLM.from_pretrained(args.model_size)
      self.gpt = add_peft_configuration(self.gpt, lora_config)
      self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
      self.tokenizer.pad_token = self.tokenizer.eos_token
      

    else:
      self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
      self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      self.tokenizer.pad_token = self.tokenizer.eos_token

      # By default, fine-tune the full model. TODO: this is maybe not idea.
      for param in self.gpt.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    '''
    gpt_output = self.gpt(input_ids, attention_mask)
    sequence_output = gpt_output['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(sequence_output)
    '''
    gpt_output = self.gpt(input_ids, attention_mask)
    logits = gpt_output.logits
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def save_lora_model(model, optimizer, args, filepath):
  peft_model_id = f"{args.epochs}-{args.lr}-LoRA_rank{args.lora_rank}_alpha{args.lora_alpha}"
  model.gpt.save_pretrained(peft_model_id)
  model.gpt.config.save_pretrained(peft_model_id)
  save_model(model, optimizer, args, filepath)
  print(f"save the model to {filepath}")
 
def load_model(args, device, lora_config=None):
  saved = torch.load(args.filepath, map_location=torch.device(device))
  model = SonnetGPT(saved['args'], lora_config=lora_config)
  model.load_state_dict(saved['model'])
  return model

def load_lora_model(args, device):
  # peft_model_id = f"{args.epochs}-{args.lr}-LoRA_rank{args.lora_rank}_alpha{args.lora_alpha}"
  # config = PeftConfig.from_pretrained(peft_model_id)
  # base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
  # model = PeftModel.from_pretrained(base_model, peft_model_id)
  # return model
  peft_model_id = f"{args.epochs}-{args.lr}-LoRA_rank{args.lora_rank}_alpha{args.lora_alpha}"
  config = PeftConfig.from_pretrained(peft_model_id)
  base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
  peft_model = PeftModel.from_pretrained(base_model, peft_model_id)

  # Now wrap that PeftModel inside your custom class
  # which expects to store it in `self.gpt`.
  custom_model = SonnetGPT(args, lora_config=None)  
  custom_model.gpt = peft_model  # Overwrite the .gpt with the LoRA model

  # Also fix tokenizer & pad token
  custom_model.tokenizer = AutoTokenizer.from_pretrained(args.model_size)
  custom_model.tokenizer.pad_token_id = custom_model.tokenizer.eos_token_id
  custom_model.gpt.config.pad_token_id = custom_model.tokenizer.eos_token_id

  return custom_model

#Added function to validate the model
@torch.no_grad()
def validate(model, dataloader, device):
  model.eval()
  val_loss = 0.0
  num_batches = 0

  for batch in tqdm(dataloader, desc = 'val', disable = TQDM_DISABLE):
    # Get the input and move it to the gpu.
    b_ids, b_mask = batch['token_ids'], batch['attention_mask']
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
    labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
    loss = F.cross_entropy(logits, labels, reduction='mean')

    val_loss += loss.item()
    num_batches += 1
  return val_loss / num_batches if num_batches > 0 else float('inf')

def train(args, lora_config):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  # Added a split for validation date set
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  collate_fn = sonnet_dataset.collate_fn
  train_size = int(args.train_size * len(sonnet_dataset))
  val_size = len(sonnet_dataset) - train_size
  sonnet_dataset, val_dataset = torch.utils.data.random_split(sonnet_dataset, [train_size, val_size])
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=collate_fn)
  val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args, lora_config=lora_config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Added argument for early stopping
  best_val_loss = float("inf")
  patience = args.patience
  no_improvement_count = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    # Added validation phase
    val_loss = validate(model, val_dataloader, device)
    if val_loss == float('inf'):
      print("No data for validation")
    
    # Early stopping
    if val_loss < best_val_loss:
      print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}!")
      best_val_loss = val_loss
      no_improvement_count = 0
      if lora_config:
        save_lora_model(model, optimizer, args, f'best_{args.filepath}')
      else:
        save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
      no_improvement_count += 1
      print(f"No improvement in val loss for {no_improvement_count} epoch(s).")
      if no_improvement_count >= patience:
          print("Early stopping triggered!")
          break
    print(f"Epoch {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
    
    
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    # save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # TODO: changed to early stop file
  # saved = torch.load(f'best_{args.filepath}', weights_only=False)
  # saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
  if args.use_lora:
    model = load_lora_model(args, device)
    #tokenizer = AutoTokenizer.from_pretrained(args.model_size)
    tokenizer = model.tokenizer
    
  else:
    model = load_model(args, device)
    tokenizer = model.tokenizer

  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  #Added argument for early stopping
  parser.add_argument("--train_size", type=int, default=0.9)
  parser.add_argument("--patience", type=int, default=2)

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  # LoRA Config parameters
  parser.add_argument("--use_lora", action='store_true', help="Use LoRA for fine-tuning")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=16)
  parser.add_argument("--lora_dropout", type=float, default=0.1)

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  
  # TODO: add LoRA config
  lora_config = None
  if args.use_lora:
    lora_config = SimpleNamespace(
      lora_rank = args.lora_rank,
      lora_alpha = args.lora_alpha, 
      lora_dropout = args.lora_dropout,
      lora_task_type = TaskType.CAUSAL_LM
    )
  train(args, lora_config)
  generate_submission_sonnets(args)