import os       # os.path.exists

 

"""

The most atomic way to train and run inference for a GPT in pure, dependency-free Python.

Modified version: Attention pattern changed from KV cache to basic matrix multiplication.

This implements the full attention matrix view as described in the transformer paper.

 

@karpathy

Modified for matrix-based attention computation

PyTorch version: Value scalar autograd replaced with torch tensors.

The only change is replacing the Value class (scalar-level graph) with

torch.nn.Parameter tensors. Each op (matmul, softmax, layernorm) is now a

single node in PyTorch's autograd graph regardless of tensor size, so

backward() is O(ops) instead of O(individual scalars).

"""

 

import time

import math

import random

import json

import csv

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset

from tqdm import tqdm

from contextlib import nullcontext

random.seed(42) # Let there be order among chaos

torch.manual_seed(42)

 

# ── Device / distributed setup ────────────────────────────────────────────

_USE_DDP = "LOCAL_RANK" in os.environ and torch.cuda.is_available()

if _USE_DDP:

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    world_size = dist.get_world_size()

    rank       = dist.get_rank()

    device     = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)

    is_main    = (rank == 0)

else:

    world_size = 1

    rank       = 0

    is_main    = True

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 

# Per-rank seed so each GPU samples different training windows

random.seed(42 + rank)

# Set up autocast if 'gpu'

device_type = "cuda" if torch.cuda.is_available() else "cpu" 
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type = device_type, dtype = torch.float16)

# Load ONLY the train split — the model must generalise to unseen pairs via

# digit-level representations, not memorisation.

dataset_train = load_dataset("akash-deep321/Scratchpadded-product25", split="train")

if is_main:

    print(f"train docs: {len(dataset_train)}")

# Set Loading and Dumping files for param logging during training

LOAD_PATH = "/kaggle/input/models/akashdeep321/jason-25digit-scratch-logger/other/default/1/logger.pt"
SAVE_PATH = "logger.pt"
SAVE_STEPS = 300_000

# ── Digit-level tokeniser ────────────────────────────────────────────────────

#Vocbulary conversion to token ids

#changes made here
vocab_string = """Input:
1325679048*Targe<sch>[,] di.A=kBC+END/"""

CtoT = {}

for i in range(len(vocab_string)):
    CtoT[vocab_string[i]] = i 

print(CtoT)

BOS = len(vocab_string)

EOS = len(vocab_string) + 1

vocab_size = len(vocab_string) + 2

if is_main:

    print(f"vocab size: {vocab_size}  (digit-level)")


# Initialize the parameters, to store the knowledge of the model

n_layer = 4

n_embd = 256    # wider: must learn full 25-digit multiplication table

block_size = 10503 # number of tokens in 1 scratchpad

n_head = 4

head_dim = n_embd // n_head



# ── Model definition (nn.Module so DDP can sync gradients) ───────────────

def layernorm(x):

    # x: (T, n_embd) — normalises each row independently

    mean = x.mean(dim=-1, keepdim=True)
    x_centered = x - mean

    var = (x_centered * x_centered).mean(dim=-1, keepdim=True)

    return x_centered * (var + 1e-5).rsqrt()

 

class ArithGPT(nn.Module):

    """Lightweight GPT for arithmetic, wrapped as nn.Module for DDP."""

    def __init__(self):

        super().__init__()

        std = 0.08

        self.wte     = nn.Parameter(torch.randn(vocab_size, n_embd) * std)

        self.wpe     = nn.Parameter(torch.randn(block_size, n_embd) * std)

        self.lm_head = nn.Parameter(torch.randn(vocab_size, n_embd) * std)

        for i in range(n_layer):

            setattr(self, f'layer{i}_attn_wq', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_attn_wk', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_attn_wv', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_attn_wo', nn.Parameter(torch.randn(n_embd, n_embd) * std))

            setattr(self, f'layer{i}_mlp_fc1', nn.Parameter(torch.randn(4 * n_embd, n_embd) * std))

            setattr(self, f'layer{i}_mlp_fc2', nn.Parameter(torch.randn(n_embd, 4 * n_embd) * std))

 

    def forward(self, token_batch):

        """token_batch: (B, T) LongTensor"""

        dev = self.wte.device

        B, T = token_batch.shape

        pos  = torch.arange(T, device=dev)               # (T,)

 

        X = self.wte[token_batch] + self.wpe[pos]        # (B, T, n_embd)

        X = layernorm(X)

 

        causal_mask = torch.triu(

            torch.full((T, T), float('-inf'), device=dev), diagonal=1

        )  # (T, T) — broadcast over batch and heads

 

        for li in range(n_layer):

            wq  = getattr(self, f'layer{li}_attn_wq')

            wk  = getattr(self, f'layer{li}_attn_wk')

            wv  = getattr(self, f'layer{li}_attn_wv')

            wo  = getattr(self, f'layer{li}_attn_wo')

            fc1 = getattr(self, f'layer{li}_mlp_fc1')

            fc2 = getattr(self, f'layer{li}_mlp_fc2')

 

            # 1) Multi-head Attention

            X_residual = X

            X = layernorm(X)

            Q = X @ wq.T  # (B, T, n_embd)

            K = X @ wk.T

            V = X @ wv.T

 

            # (B, n_head, T, head_dim)

            Q = Q.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

            K = K.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

            V = V.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

 

            X_attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
            X_attn = X_attn.permute(0, 2, 1, 3).contiguous().view(B, T, n_embd)

            X = X_attn @ wo.T

            X = X + X_residual

 

            # 2) MLP

            X_residual = X

            X = layernorm(X)

            X = X @ fc1.T

            X = F.gelu(X, approximate='tanh')

            X = X @ fc2.T

            X = X + X_residual

 

        return X @ self.lm_head.T  # (B, T, vocab_size)

 

# Instantiate model, move to device, wrap with DDP for multi-GPU training

model = ArithGPT().to(device)

if _USE_DDP:

    model = DDP(model, device_ids=[local_rank])

elif not _USE_DDP and torch.cuda.device_count() > 1:
    
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0") 
    
if is_main:

    print(f"num params: {sum(p.numel() for p in model.parameters())}")

    print(f"device: {device}  |  world_size: {world_size}")

print(type(model), getattr(model, 'device_ids', 'N/A')) 

#inittialize GradScaler for autocasting to float16

scaler = torch.amp.GradScaler()

# AdamW with strong weight decay — this is the key trigger for grokking

# (generalisation to unseen pairs).  Without weight decay transformers

# memorise the training set but never generalise.

learning_rate = 3e-4

weight_decay  = 2**(-10)   # strong L2 as in Power et al. 2022 (Grokking paper)

optimizer = torch.optim.AdamW(

    model.parameters(), lr=learning_rate,

    betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay

)

 

batch_size   = 16

num_steps    = 3

warmup_steps = 5_000

 

Loss = 0

LOG_EVERY = 1

loss_history = []       # (step, loss)

batch_loss_history = [] # (batch, avg_loss)

def get_raw_model():
    return model.module if (_USE_DDP or isinstance(model, torch.nn.DataParallel)) else model

# define the logging function to log mid training

def save_checkpoint(step, loss_history, batch_loss_history, accuracy_history):
    raw = get_raw_model()
    torch.save({
        'step'               : step,
        'model_state'        : raw.state_dict(),
        'optimizer_state'    : optimizer.state_dict(),
        'scaler_state'       : scaler.state_dict(),
        'loss_history'       : loss_history,
        'batch_loss_history' : batch_loss_history,
        'accuracy_history'   : accuracy_history,
    }, SAVE_PATH)
    if is_main:
        tqdm.write(f"[ckpt] saved at step {step} → {SAVE_PATH}")

# Loading function to resume training

def load_checkpoint():
    if not os.path.exists(LOAD_PATH):
        if is_main:
            print("No checkpoint found — starting from scratch.")
        return 0, [], [], []
 
    ckpt = torch.load(LOAD_PATH, map_location=device)
    get_raw_model().load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scaler.load_state_dict(ckpt['scaler_state'])
 
    start_step = ckpt['step']
    if is_main:
        print(f"Resumed from step {start_step}.")
    
    #Returns (start_step, loss_history, batch_loss_history, accuracy_history).
    return (
        start_step,
        ckpt.get('loss_history', []),
        ckpt.get('batch_loss_history', []),
        ckpt.get('accuracy_history', []),
    )

# Pre-parse training data

def to_digits(s):

    x = [CtoT[c] for c in s]

    tokens  = [BOS] + x

    targets = x + [EOS]

    return tokens, targets

parsed_dataset = []
test_dataset = []

for doc in dataset_train:

    s = doc["Input"] + '\n' + doc["Label"].strip()
    parsed_dataset.append(s)
 
random.shuffle(parsed_dataset)

if is_main:

    print(f"Training pairs: {len(parsed_dataset)}")

start_step, loss_history, batch_loss_history, accuracy_history = load_checkpoint()

total_start = time.perf_counter()

 

for step in tqdm(range(start_step, min(num_steps, start_step + SAVE_STEPS)), disable=not is_main):

    # Sample a batch

    batch   = random.choices(parsed_dataset, k=batch_size)

    tok_np  = [to_digits(s)[0] for s in batch]

    tgt_np  = [to_digits(s)[1] for s in batch]

    token_batch  = torch.tensor(tok_np, dtype=torch.long, device=device)   # (B, 10503)

    target_batch = torch.tensor(tgt_np, dtype=torch.long, device=device)   # (B, 10503)

    #running forward pass in the autocast
    with ctx:

        logits = model(token_batch)  # (B, 10503, vocab_size)

        # Full-sequence loss — every position contributes gradient to the embeddings.

        loss = F.cross_entropy(

            logits.reshape(-1, vocab_size),

            target_batch.reshape(-1)

        )
        
    optimizer.zero_grad()
    
    #Scaling gradients for backward pass along with gradient clipping
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

    # Linear warmup then constant LR

    lr_t = learning_rate * min(1.0, (step + 1) / warmup_steps)

    for pg in optimizer.param_groups:

        pg['lr'] = lr_t

    scaler.step(optimizer)
    scaler.update()

 

    # All-reduce loss across GPUs so rank-0 logs the true average

    if _USE_DDP:

        loss_reduced = loss.detach().clone()

        dist.all_reduce(loss_reduced, op=dist.ReduceOp.AVG)

        step_loss = loss_reduced.item()

    else:

        step_loss = loss.item()

 

    if is_main:

        loss_history.append((step, step_loss))

        Loss += step_loss

        if (step + 1) % LOG_EVERY == 0:

            avg = Loss / LOG_EVERY

            batch_loss_history.append(((step + 1) // LOG_EVERY, avg))

            tqdm.write(f"Step {step+1:6d}/{num_steps} | loss {avg:.4f} | lr {lr_t:.2e}")

            Loss = 0

        elif step == num_steps - 1 and Loss > 0:

            remainder = (step + 1) % LOG_EVERY or LOG_EVERY

            avg = Loss / remainder

            batch_loss_history.append(((step + 1) // LOG_EVERY + 1, avg))

            tqdm.write(f"Step {step+1:6d}/{num_steps} | loss {avg:.4f} | lr {lr_t:.2e}")

            Loss = 0

 

elapsed = time.perf_counter() - total_start

if is_main:

    print(f"\nTotal training time : {elapsed:.1f}s")

    print(f"Per step            : {elapsed / num_steps * 1000:.1f}ms")

    save_checkpoint(min(num_steps, start_step + SAVE_STEPS), loss_history, batch_loss_history, accuracy_history)

    # Unwrap DDP to access raw parameters for saving

    raw_model = model.module if (_USE_DDP or isinstance(model, torch.nn.DataParallel)) else model

    state_dict_data = {}

    for name, param in raw_model.named_parameters():

        # Map attribute names back to original dot-separated JSON key format

        # e.g. layer0_attn_wq → layer0.attn_wq

        if name.startswith('layer'):

            idx = name.index('_')

            key = name[:idx] + '.' + name[idx+1:]

        else:

            key = name

        state_dict_data[key] = param.detach().cpu().tolist()

    with open("Matrix-microGPT-25digit-torch.json","w") as f:

        json.dump(state_dict_data, f)

 

    # Save loss history to CSV

    with open("train_loss.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["step", "loss"])

        writer.writerows(loss_history)

    print("Loss history saved to train_loss.csv")

 

    with open("train_loss_batch.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["batch", "avg_loss"])

        writer.writerows(batch_loss_history)

    print("Batch loss history saved to train_loss_batch.csv")

 

    # Auto-plot loss curve

    import subprocess, sys

    subprocess.run([sys.executable, "plot_loss.py"], check=False)

    #just some code to make the plots if using kaggle (subprocesses wont work here)
    
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend, works on Kaggle with no display
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # read from CSV so it works even if loss_history is partially in memory
    df_loss  = pd.read_csv("train_loss_batch.csv")

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # loss plot
    ax1.plot(df_loss["batch"], np.log(df_loss["avg_loss"]), color='steelblue', linewidth=1)
    ax1.set_xlabel("Step batch")
    ax1.set_ylabel("Avg Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Plot saved successfully")

if _USE_DDP:

    dist.destroy_process_group()

#——————————————————————————————————————————————————————————————————————#
