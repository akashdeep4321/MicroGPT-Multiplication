import os       # os.path.exists
from contextlib import nullcontext

ENABLE_TRACKING = os.environ.get("ENABLE_TRACKING", "0") == "1"
TRACKING_ENABLED = False

class NullMLflow:
    @staticmethod
    def start_run():
        return nullcontext()

    @staticmethod
    def log_param(*args, **kwargs):
        return None

    @staticmethod
    def log_metric(*args, **kwargs):
        return None

    @staticmethod
    def log_artifact(*args, **kwargs):
        return None

try:
    if ENABLE_TRACKING:
        import dagshub
        import mlflow

        dagshub.init(repo_owner='cs1251117', repo_name='Matrix-microgpt', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/cs1251117/Matrix-microgpt.mlflow")
        mlflow.set_experiment("2-digit-product-Matrix-simple")
        TRACKING_ENABLED = True
    else:
        mlflow = NullMLflow()
except ImportError:
    mlflow = NullMLflow()
    print("Tracking disabled because dagshub/mlflow is not installed.")


"""

The most atomic way to train and run inference for a GPT in pure, dependency-free Python.

Modified version: Attention pattern changed from KV cache to basic matrix multiplication.

This implements the full attention matrix view as described in the transformer paper.

 

@karpathy

Modified for matrix-based attention computation

PyTorch version: Value scalar autograd replaced with torch tensors.

The only change is replacing the Value class (scalar-level graph) with

torch.nn.Parameter tensors. Each op (matmul, softmax, rmsnorm) is now a

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


import math     # math.log, math.exp

from datasets import load_dataset

from tqdm import tqdm

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

 

# Load ONLY the train split — the model must generalise to unseen pairs via

# digit-level representations, not memorisation.

dataset_train = load_dataset("akash-deep321/Product-Arithmetic", split="train")

if is_main:

    print(f"train docs: {len(dataset_train)}")

 

# ── Digit-level tokeniser ────────────────────────────────────────────────────

# Digits 0-9 map to themselves (token IDs 0-9).

# This gives the model numeric structure: similar digits → similar embeddings

# after training.  The model can then generalise to unseen (left, right) pairs.

STAR      = 10   # '*'

EQ        = 11   # '='

BOS       = 12

EOS       = 13

vocab_size = 14

# Sequence for 23*47=1081:

#   tokens : [BOS, 2, 3, STAR, 4, 7, EQ, 1, 0, 8, 1]   (11 tokens, positions 0-10)

#   targets: [2, 3, STAR, 4, 7, EQ, 1, 0, 8, 1, EOS]   (shift-by-1)

# Loss is computed only on the 4 result digits (target positions 6-9).

if is_main:

    print(f"vocab size: {vocab_size}  (digit-level)")

 

# Initialize the parameters, to store the knowledge of the model

n_layer = 4

n_embd = 256    # wider: must learn full 2-digit multiplication table

block_size = 11 # [BOS l1 l2 STAR r1 r2 EQ res1 res2 res3 res4] = 11 tokens

n_head = 4

head_dim = n_embd // n_head

 

# ── Model definition (nn.Module so DDP can sync gradients) ───────────────

import sys
sys.setrecursionlimit(50000)

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
dataset = load_dataset("akash-deep321/Product-Arithmetic", split = "train")
print(f"num docs: {len(dataset)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
NUM_TOKENS = 10000
STAR = NUM_TOKENS
EQUALS = STAR + 1
MASK = EQUALS + 1
BOS = MASK + 1
vocab_size = BOS + 1
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
input_size = 20
block_size = input_size * 6  # [BOS, x, *, y, =, z] repeated for a 20-example window
answer_reveal_prob = 0.5
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):

    # x: (T, n_embd) — normalises each row independently

    ms = (x * x).mean(dim=-1, keepdim=True)

    return x * (ms + 1e-5).rsqrt()

 

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

        X = rmsnorm(X)

 

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

            X = rmsnorm(X)

            Q = X @ wq.T  # (B, T, n_embd)

            K = X @ wk.T

            V = X @ wv.T

 

            # (B, n_head, T, head_dim)

            Q = Q.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

            K = K.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

            V = V.view(B, T, n_head, head_dim).permute(0, 2, 1, 3)

 

            attn_logits  = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)  # (B, n_head, T, T)

            attn_weights = F.softmax(attn_logits + causal_mask, dim=-1)

            X_attn = (attn_weights @ V).permute(0, 2, 1, 3).contiguous().view(B, T, n_embd)

            X = X_attn @ wo.T

            X = X + X_residual

 

            # 2) MLP

            X_residual = X

            X = rmsnorm(X)

            X = X @ fc1.T

            X = F.relu(X)

            X = X @ fc2.T

            X = X + X_residual

 

        return X @ self.lm_head.T  # (B, T, vocab_size)

 

# Instantiate model, move to device, wrap with DDP for multi-GPU training

model = ArithGPT().to(device)

if _USE_DDP:

    model = DDP(model, device_ids=[local_rank])

if is_main:

    print(f"num params: {sum(p.numel() for p in model.parameters())}")

    print(f"device: {device}  |  world_size: {world_size}")

 

# AdamW with strong weight decay — this is the key trigger for grokking

# (generalisation to unseen pairs).  Without weight decay transformers

# memorise the training set but never generalise.

learning_rate = 3e-4

weight_decay  = 1.0   # strong L2 as in Power et al. 2022 (Grokking paper)

optimizer = torch.optim.AdamW(

    model.parameters(), lr=learning_rate,

    betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay

)

 

batch_size   = 128

num_steps    = 1_000_000  

warmup_steps = 5_000

 

Loss = 0


loss_history = []       # (step, loss)

batch_loss_history = [] # (batch, avg_loss)

 

# Pre-parse training data

def to_digits(left, right, result):

    """Encode one multiplication example as digit tokens.

    tokens  = [BOS, l1, l2, STAR, r1, r2, EQ, d1, d2, d3, d4]  length 11

    targets = [l1,  l2, STAR, r1, r2, EQ, d1, d2, d3, d4, EOS] length 11

    Result is zero-padded to 4 digits (max 99*99=9801).

    """

    l1, l2 = left  // 10, left  % 10

    r1, r2 = right // 10, right % 10

    d1 = result // 1000

    d2 = (result // 100) % 10

    d3 = (result // 10)  % 10

    d4 =  result         % 10

    tokens  = [BOS, l1, l2, STAR, r1, r2, EQ, d1, d2, d3, d4]

    targets = [     l1, l2, STAR, r1, r2, EQ, d1, d2, d3, d4, EOS]

    return tokens, targets

 

parsed_dataset = []

for doc in dataset_train:

    l, r = doc["Expression"].split("*")

    parsed_dataset.append((int(l), int(r), int(doc["Result"])))

 

# Commutativity augmentation: a×b = b×a — doubles effective training data for free

parsed_dataset += [(r, l, res) for l, r, res in parsed_dataset if l != r]

random.shuffle(parsed_dataset)

if is_main:

    print(f"Training pairs (with commutativity): {len(parsed_dataset)}")

 

total_start = time.perf_counter()

 

for step in tqdm(range(num_steps), disable=not is_main):

    # Sample a batch

    batch   = random.choices(parsed_dataset, k=batch_size)

    tok_np  = [to_digits(l, r, res)[0] for l, r, res in batch]

    tgt_np  = [to_digits(l, r, res)[1] for l, r, res in batch]

    token_batch  = torch.tensor(tok_np, dtype=torch.long, device=device)   # (B, 11)

    target_batch = torch.tensor(tgt_np, dtype=torch.long, device=device)   # (B, 11)

 

    logits = model(token_batch)  # (B, 11, vocab_size)

 

    # Full-sequence loss — every position contributes gradient to the embeddings.

    # Predicting operand digits and structure tokens (STAR, EQ, BOS) is cheap

    # but forces the embedding layer to encode digit identity precisely, which

    # is what allows the model to generalise to unseen (left, right) pairs.

    loss = F.cross_entropy(

        logits.reshape(-1, vocab_size),

        target_batch.reshape(-1)

    )

 

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

 

    # Linear warmup then constant LR

    lr_t = learning_rate * min(1.0, (step + 1) / warmup_steps)

    for pg in optimizer.param_groups:

        pg['lr'] = lr_t

    optimizer.step()

 

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

 

        LOG_EVERY = 500

        if (step + 1) % LOG_EVERY == 0:

            avg = Loss / LOG_EVERY

            batch_loss_history.append(((step + 1) // LOG_EVERY, avg))

            tqdm.write(f"Step {step+1:6d}/{num_steps} | loss {avg:.4f} | lr {lr_t:.2e}")


def encode_example(doc, reveal_answer=True):
    left, right = doc["Expression"].split("*")
    result = doc["Result"]
    left = int(left)
    right = int(right)
    result = int(result)
    visible_result = result if reveal_answer else MASK
    input_chunk = [left, STAR, right, EQUALS, visible_result, BOS]
    target_chunk = [left, STAR, right, EQUALS, result, BOS]
    return input_chunk, target_chunk

def sample_reveal_flags(window_size):
    reveal_flags = [random.random() < answer_reveal_prob for _ in range(window_size)]
    if window_size > 1:
        if all(reveal_flags):
            reveal_flags[random.randrange(window_size)] = False
        elif not any(reveal_flags):
            reveal_flags[random.randrange(window_size)] = True
    return reveal_flags

def encode_window(data):
    reveal_flags = sample_reveal_flags(len(data))
    visible_sequence = [BOS]
    target_sequence = [BOS]
    for doc, reveal_answer in zip(data, reveal_flags):
        input_chunk, target_chunk = encode_example(doc, reveal_answer=reveal_answer)
        visible_sequence.extend(input_chunk)
        target_sequence.extend(target_chunk)
    return visible_sequence[:-1], target_sequence[1:], reveal_flags

with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("block_size", block_size)
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("answer_reveal_prob", answer_reveal_prob)
    
    for step in tqdm(range(num_steps)):
        # Randomly take a 20-example window and train autoregressively over it.
        start = random.randint(0, len(dataset) - input_size)
        tokens, target_ids, reveal_flags = encode_window([dataset[j] for j in range(start, start + input_size)])
        if step == 0:
            hidden_answers = sum(1 for flag in reveal_flags if not flag)
            print(
                "tokens", len(tokens), tokens,
                "\ntargets", len(target_ids), target_ids,
                "\nrevealed", sum(reveal_flags),
                "hidden", hidden_answers,
            )
        n = len(tokens)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        logits = gpt(tokens, list(range(n)))
        losses = []
        for pos_id in range(n):
            probs = softmax(logits[pos_id])
            target_id = target_ids[pos_id]
            losses.append(-probs[target_id].log())
        loss = (1 / n) * sum(losses)
        
        # Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward()
    
        # Adam optimizer update: update the model parameters based on the corresponding gradients
        lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0
        Loss += loss.data

        mlflow.log_metric("lr", lr_t, step=step)
        mlflow.log_metric("loss", loss.data, step=step)
        
        if (step+1)%32 == 0:
            tqdm.write(f"Batch {((step+1)//32):4d} / {((num_steps//32)+1):4d} | loss {(Loss/32):.4f}", end='\r')

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

 

    # Unwrap DDP to access raw parameters for saving

    raw_model = model.module if _USE_DDP else model

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

    with open("Matrix-microGPT-2digit-torch.json","w") as f:

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

 

if _USE_DDP:

    dist.destroy_process_group()

#——————————————————————————————————————————————————————————————————————#
