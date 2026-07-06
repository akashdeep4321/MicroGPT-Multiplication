import os       # os.path.exists
import pandas as pd
 

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

dataset_test = load_dataset(
    "parquet",
    data_files={"test": "https://huggingface.co/datasets/akash-deep321/Jason-10Digit-Scratchpadded50M/resolve/main/data/test-00000.parquet"},
    split="test",
)

if is_main:

    print(f"train docs: {len(dataset_test)}")

#Model Path

MODEL_PATH = "/kaggle/input/models/akashdeep321/10digits-jason-1m/other/default/1/Matrix-microGPT-25digit-torch.json"

# ── Digit-level tokeniser ────────────────────────────────────────────────────

#Vocbulary conversion to token ids

#changes made here
vocab_string = """Input:
1325679048*Targe<sch>[,] di.A=kBC+END/"""

CtoT = {}

for i in range(len(vocab_string)):
    CtoT[vocab_string[i]] = i 

TtoC = {i: c for c, i in CtoT.items()}
print(CtoT)

BOS = len(vocab_string)

EOS = len(vocab_string) + 1

vocab_size = len(vocab_string) + 2

if is_main:

    print(f"vocab size: {vocab_size}  (digit-level)")


# Initialize the parameters, to store the knowledge of the model

n_layer = 4

n_embd = 256    # wider: must learn full 2-digit multiplication table

block_size = 1998 # number of tokens in 1 scratchpad

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

            X = F.gelu(X, approximate = 'tanh')

            X = X @ fc2.T

            X = X + X_residual

 

        return X @ self.lm_head.T  # (B, T, vocab_size)

 

# Loading model weights

assert os.path.exists(MODEL_PATH), f"JSON not found at {MODEL_PATH}"
 
print(f"Loading weights from {MODEL_PATH} ...")

with open(MODEL_PATH, "r") as f:
    state_dict_data = json.load(f)

state_dict = {}
for key, val in state_dict_data.items():
    if key.startswith('layer'):
        # "layer0.attn_wq" → "layer0_attn_wq"
        attr_name = key.replace('.', '_', 1)
    else:
        attr_name = key
    state_dict[attr_name] = torch.tensor(val)
 
model = ArithGPT().to(device)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded. num params: {sum(p.numel() for p in model.parameters()):,}")

if _USE_DDP:

    model = DDP(model, device_ids=[local_rank])

elif not _USE_DDP and torch.cuda.device_count() > 1:
    
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0") 
    
if is_main:

    print(f"device: {device}  |  world_size: {world_size}")

print(type(model), getattr(model, 'device_ids', 'N/A')) 

def get_raw_model():
    return model.module if (_USE_DDP or isinstance(model, torch.nn.DataParallel)) else model

# Pre-parse training data

def to_digits(s):

    x = [CtoT[c] for c in s]

    tokens  = [BOS] + x

    targets = x + [EOS]

    return tokens, targets

parsed_dataset = []

for doc in dataset_test:

    parsed_dataset.append((doc["Input"],doc["Label"].strip().split('\n')[-1].strip()))

if is_main:

    print(f"Test pairs: {len(parsed_dataset)}")

accuracy = {"Step" : [], "Accuracy" : []}
cnt = 0

total_start = time.perf_counter()

num_steps = 1_000
batch = 2

for step in tqdm(range(0, num_steps, batch), disable=not is_main):

    # Sample a batch

    sample   = parsed_dataset[step:step+batch]
    tok_np  = [to_digits(s[0])[0] for s in sample]

    tgt_np  = [to_digits(s[1])[0][1:] for s in sample]

    token_batch  = torch.tensor(tok_np, dtype=torch.long, device=device)   # (B, 58)
    #running forward pass in the autocast
    with ctx:
        pred_id = [BOS]*batch
        while pred_id[0] != EOS:
            
            logits = model(token_batch)  # (B, 10503, vocab_size)
            pred_id = logits[:, -1, :].argmax(dim=-1)
            token_batch = torch.cat([token_batch, pred_id.unsqueeze(1)], dim=1)
            
    preds = token_batch[:,-40:-1].tolist()
    for b in range(batch):
        cnt += int(preds[b] == tgt_np[b])
    print("Correct -", cnt)
    for b in range(batch):
        print("Pred - ", preds[b]); 
        print("Truth - ", tgt_np[b])
    
    accuracy["Step"].append(step)
    accuracy["Accuracy"].append(cnt/(step+2))

elapsed = time.perf_counter() - total_start

if is_main:

    print(f"\nTotal training time : {elapsed:.1f}s")

    print(f"Per step            : {elapsed / num_steps * 1000:.1f}ms")

    df = pd.DataFrame(accuracy)
    df.to_csv("accuracies.csv")
    
    #just some code to make the plots if using kaggle (subprocesses wont work here)
    
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend, works on Kaggle with no display
    import matplotlib.pyplot as plt
    
    # read from CSV so it works even if loss_history is partially in memory
    df_acc  = pd.read_csv("accuracies.csv")

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # loss plot
    ax1.plot(df_acc["Step"], df_acc["Accuracy"], color='steelblue', linewidth=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracies")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Plot saved successfully")
    print("Final Accuracy -", accuracy["Accuracy"][-1])
if _USE_DDP:

    dist.destroy_process_group()

#——————————————————————————————————————————————————————————————————————#
