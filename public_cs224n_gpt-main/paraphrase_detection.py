'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch
from types import SimpleNamespace
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model, add_peft_configuration
from peft import TaskType, PeftModel, PeftConfig
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


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args, lora_config=None):
    super().__init__()
    # TODO: added lora config
    if lora_config:
      self.gpt = AutoModelForCausalLM.from_pretrained(args.model_size)
      self.gpt = add_peft_configuration(self.gpt, lora_config)
    else:
      self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    gpt_output = self.gpt.transformer(input_ids=input_ids, attention_mask=attention_mask)
    # TODO: added to change last token manually
    sequence_output = gpt_output['last_hidden_state']
    last_non_pad_idx = attention_mask.sum(dim=1) - 1
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]
    # gpt_output =  self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    # last_token = gpt_output['last_token'] # 8 x 768
    logits = self.paraphrase_detection_head(last_token) # 8 x 2
    return logits


def save_lora_model(model, optimizer, args, filepath):
  peft_model_id = f"{args.epochs}-{args.lr}-LoRA_rank{args.lora_rank}_alpha{args.lora_alpha}"
  model.gpt.save_pretrained(peft_model_id)
  save_model(model, optimizer, args, filepath)
  print(f"save the model to {filepath}")

def load_lora_model(args):
  # old_load = load_model(args, device, lora_config)
  # model_params = cur_model.state_dict()
  # lora_params = [x for x in model_params.keys() if 'lora' in x]
  peft_model_id = f"{args.epochs}-{args.lr}-LoRA_rank{args.lora_rank}_alpha{args.lora_alpha}"
  config = PeftConfig.from_pretrained(peft_model_id)
  base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
  model = PeftModel.from_pretrained(base_model, peft_model_id)
  return model

def load_model(args, device, lora_config):
  saved = torch.load(args.filepath, map_location=torch.device(device))
  model = ParaphraseGPT(saved['args'], lora_config)
  model.load_state_dict(saved['model'])
  return model

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


def train(args, lora_config):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args, lora_config)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask) # shape (batch_size, 2)
      preds = torch.argmax(logits, dim=1) # shape (batch_size)
      
      # ADDED - Map 'No'-3919 to 0 and 'Yes'-8505 to 1
      labels = torch.where(labels == 3919, torch.tensor(0), torch.tensor(1)).to(device=device)

      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      if lora_config:
        save_lora_model(model, optimizer, args, args.filepath)
      else:
        save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


@torch.no_grad()
def test(args, lora_config):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  #TODO: changed this to fit loading lora
  if args.use_lora:
    model = load_lora_model(args)
  else:
    model = load_model(args, device, lora_config)
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  label_map = {1: 8505, 0: 3919}
  # ADDED - Map 0 to 'No'-3919 and 1 to 'Yes'-8505
  dev_para_y_pred = [label_map[x] for x in dev_para_y_pred]
  test_para_y_pred = [label_map[x] for x in test_para_y_pred]

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  # Parameters for LoRA config
  parser.add_argument("--use_lora", action='store_true')
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
  lora_params = f"_lora_rank{args.lora_rank}_alpha{args.lora_alpha}"
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase{lora_params if args.use_lora else ""}.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.

  lora_config = None
  if args.use_lora:
    lora_config = SimpleNamespace(
      lora_rank = args.lora_rank,
      lora_alpha = args.lora_alpha,
      lora_dropout = args.lora_dropout,
      lora_task_type = TaskType.QUESTION_ANS
  )
  train(args, lora_config)
  test(args, lora_config)
