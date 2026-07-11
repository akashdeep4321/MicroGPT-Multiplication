import random
import string
import math
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import os

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

import re


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BUCKET_LENGTH = 1_000_000_000
DIGITS      = 10
SHARD_SIZE  = 500_000_000
BATCH_SIZE = 100_000
NUM_WORKERS = cpu_count()

BIG_LIMIT = 8_000_000_000
LOW_LIMIT = 3_000_000_000

BUCKETS = ["round-x-any","big-x-any", "small-x-any", "odd-x-odd", "even-x-even", "odd-x-even", "any-x-any"]

TOKEN_ID = 'Enter Token ID'
REPO_ID = 'akash-deep321/Jason-10Digit-Test7B'
TMP_DIR = './tmp/hf_shards'

# ──────────────────────────────────────────────
# Config for Padder
# ──────────────────────────────────────────────
"""
Fixed widths (if 25-digit inputs):
  x, y list  : 25 digits
  A list      : 26 digits   (9 × 25 nines = 26-digit number)
  k           : 25 digits   (10^24 at most)
  B list      : 50 digits   (A_max × k_max)
  C, C_prev   : 50 digits   (product of two 25-digit numbers)
  answer      : 50 digits
"""

W_X = DIGITS
W_A = DIGITS + 1
W_K = DIGITS
W_B = 2 * DIGITS
W_C = 2 * DIGITS
W_ANS = 2 * DIGITS

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def list_to_string(a):
    return str(a).replace(' ', '')

def truncate_to_n_digit(x, n=4):
    return math.floor(x * (10 ** n)) / (10 ** n)


def get_algo_reasoning_str(args):
    """
    Unpacks (x, y, result, operator) tuple.
    Identical logic to original — just wrapped for multiprocessing.
    """
    x, y, result, operator = args
    x, y = str(x), str(y)

    len_x, len_y   = len(x), len(y)
    list_x, list_y = [int(d) for d in x], [int(d) for d in y]

    output_str  = f'Input:\n{x}{operator}{y}\n'
    output_str += f'Target:\n'
    output_str += f'<scratch>\n'
    output_str += f'{list_to_string(list_x)} has {len_x} digits.\n'
    output_str += f'{list_to_string(list_y)} has {len_y} digits.\n'

    # ── multiplication scratchpad ──
    if operator == '*':
        C = 0
        orig_list_y = [int(d) for d in y]   # keep a fresh copy
        for i in range(len_y):
            b     = orig_list_y[-(i + 1)]
            A     = b * int(x)
            B     = A * (10 ** i)
            C_prev = C
            C    += B
            A_digits = [int(d) for d in str(A)]
            B_digits = [int(d) for d in str(B)]
            suffix   = '' if i == len_y - 1 else '\n'
            output_str += (
                f'{list_to_string(list_x)} * {b} , '
                f'A={list_to_string(A_digits)} , '
                f'k={10**i} , '
                f'B={list_to_string(B_digits)} , '
                f'C={C_prev}+{B}={C}'
                f'{suffix}'
            )

        output_str += ' , END\n</scratch>\n'
        output_str += ' '.join(str(C))
        return output_str + '\n'

    raise ValueError(f'Operator {operator} not supported in this version.')

def generate_pairs(n: int, digits: int, bucket: str, seed: int) -> list:
    """
    Generating pairs of products
    """
    
    rng = np.random.default_rng(seed=seed)

    # Pre-generate in large batches to amortise Python loop overhead

    low = 10 ** (digits-1)
    high = 10 ** digits
    
    if bucket == "round-x-any":
        lowa = 10 ** (digits-2)
        higha = (10 ** (digits-1))

        for _ in range(n):
            x = int(rng.integers(lowa, higha)) * 10
            y = int(rng.integers(low, high))
            z = int(rng.integers(0, 2))
            if z:
                yield str(x), str(y), str(x*y), '*'
            else:
                yield str(y), str(x), str(x*y), '*'

    elif bucket == "big-x-any":
        lowa = BIG_LIMIT
        higha = high

        for _ in range(n):
            x = int(rng.integers(lowa, higha))
            y = int(rng.integers(low, high))
            z = int(rng.integers(0, 2))
            if z:
                yield str(x), str(y), str(x*y), '*'
            else:
                yield str(y), str(x), str(x*y), '*'

    elif bucket == "small-x-any":
        lowa = low
        higha = LOW_LIMIT

        for _ in range(n):
            x = int(rng.integers(lowa, higha))
            y = int(rng.integers(low, high))
            z = int(rng.integers(0, 2))
            if z:
                yield str(x), str(y), str(x*y), '*'
            else:
                yield str(y), str(x), str(x*y), '*' 

    elif bucket == "odd-x-odd":
        lowa = low // 2
        higha = high // 2

        for _ in range(n):
            x = (int(rng.integers(lowa, higha)) * 2) + 1
            y = (int(rng.integers(lowa, higha)) * 2) + 1
            yield str(y), str(x), str(x*y), '*' 

    elif bucket == "even-x-even":
        lowa = low // 2
        higha = high // 2

        for _ in range(n):
            x = (int(rng.integers(lowa, higha)) * 2) 
            y = (int(rng.integers(lowa, higha)) * 2) 
            yield str(y), str(x), str(x*y), '*'

    elif bucket == "odd-x-even":
        lowa = low // 2
        higha = high // 2

        for _ in range(n):
            x = (int(rng.integers(lowa, higha)) * 2) + 1
            y = (int(rng.integers(lowa, higha)) * 2) 
            z = int(rng.integers(0, 2)) 
            if z:
                yield str(x), str(y), str(x*y), '*'
            else:
                yield str(y), str(x), str(x*y), '*' 

    else:
        for _ in range(n):
            x = int(rng.integers(low, high))
            y = int(rng.integers(low, high))
            yield str(x), str(y), str(x*y), '*'

    
    
    
    

# ──────────────────────────────────────────────
# Helpers for the padder
# ──────────────────────────────────────────────
def _pad_num(s: str, width: int) -> str:
    """Left-pad a digit string with zeros to `width`."""
    return s.zfill(width)
 
 
def _pad_list(digits: str, width: int) -> str:
    """'123' -> '[0,0,1,2,3]' padded to `width` digits."""
    return '[' + ','.join(digits.zfill(width)) + ']'
 
 
def _digits(list_repr: str) -> str:
    """'[1,2,3]' -> '123'"""
    return list_repr.strip()[1:-1].replace(',', '')

# ──────────────────────────────────────────────
# Padder
# ──────────────────────────────────────────────
def pad_sample(raw: str) -> str:
    """
    Takes the full raw scratchpad string and returns it with every
    number zero-padded to its maximum possible width.
    """
    result = []
 
    for line in raw.split('\n'):
 
        # "[1,3,...] has 25 digits."
        m = re.match(r'^(\[[\d,]+\]) has \d+ digits\.$', line)
        if m:
            result.append(_pad_list(_digits(m.group(1)), W_X) + f' has {W_X} digits.')
            continue
 
        # "[x] * b , A=[a] , k=K , B=[B] , C=Cprev+Bnum=Cnew( , END)"
        m = re.match(
            r'^(\[[\d,]+\]) \* (\d+) , '
            r'A=(\[[\d,]+\]) , '
            r'k=(\d+) , '
            r'B=(\[[\d,]+\]) , '
            r'C=(\d+)\+(\d+)=(\d+)'
            r'(.*)?$',
            line
        )
        if m:
            result.append(
                f'{_pad_list(_digits(m.group(1)), W_X)} * {m.group(2)} , '
                f'A={_pad_list(_digits(m.group(3)), W_A)} , '
                f'k={_pad_num(m.group(4), W_K)} , '
                f'B={_pad_list(_digits(m.group(5)), W_B)} , '
                f'C={_pad_num(m.group(6), W_C)}+{_pad_num(m.group(7), W_B)}={_pad_num(m.group(8), W_C)}'
                f'{m.group(9) or ""}'
            )
            continue
 
        # answer line: space-separated digits e.g. "9 4 7 7 0 ..."
        stripped = line.strip()
        if (' ' in stripped and stripped.replace(' ', '').isdigit()) or len(stripped) == 1:
            result.append(' '.join(_pad_num(stripped.replace(' ', ''), W_ANS)))
            continue
 
        result.append(line)
 
    return '\n'.join(result)
# ──────────────────────────────────────────────
# Parse reasoning string → (Input, Scratchpad, Answer)
# ──────────────────────────────────────────────
def parse_reasoning(s: str):
    lines  = s.strip().split('\n')
    inp    = lines[1]
    answer = lines[-1]
    # everything between "Target:" line (index 2) and last line
    scratch = '\n'.join(lines[3: len(lines) - 1])
    return inp, scratch, answer

def generator(n, bucket,seed):
    
    print(f'[1/5] Generating {n} train pairs ...')
    pairs = generate_pairs(n, DIGITS, bucket, seed)

    print(f'[3/5] Building scratchpads ({NUM_WORKERS} workers) ...')
    
    with Pool(NUM_WORKERS) as pool:
        reasoning = pool.imap(get_algo_reasoning_str, pairs, chunksize=500)

        for r in reasoning:
            row = parse_reasoning(r)
            inp, scratch, ans = row

            s = "Input:\n" + inp.strip() + "\nTarget:\n" + scratch.strip() + "\n" + ans.strip() + "\n"
            x = (pad_sample(s).split('\n'))
            a,b = x[1].split('*')
            if len(a) == 1:
                a = '0' + a
            if len(b) == 1:
                b = '0' + b
            x[1] = a + '*' + b
            
            query = '\n'.join([x[0],x[1]])
            label = '\n'.join([x[i] for i in range(2,len(x))])

            yield {'Input': query, 'Label': label}

def writer(rows, shard_id, split, api, bucket):
    os.makedirs(TMP_DIR, exist_ok=True)
    local_path = os.path.join(TMP_DIR, f"{split}-{bucket}-{shard_id:05d}.parquet")

    write = None
    buffer = {"Input" : [], "Label" : []}
    rows_done = 0

    def flush():
        nonlocal write, buffer, rows_done
        if not buffer["Input"]:
            return
        table = pa.table(buffer)
        if write is None:
            write = pq.ParquetWriter(local_path, table.schema)
        write.write_table(table)
        rows_done += len(buffer["Input"])
        buffer = {"Input" : [], "Label" : []}

    for row in rows:
        buffer["Input"].append(row["Input"])
        buffer["Label"].append(row["Label"])

        if len(buffer["Input"]) >= BATCH_SIZE:
            flush()
    flush()

    if write is not None:
        write.close()

    print(f"Uploading Shard {shard_id} to HF. Rows written - {rows_done}")

    api.upload_file(
        path_or_fileobj = local_path,
        path_in_repo = f"data/{split}-{bucket}-{shard_id:05d}.parquet",
        repo_id = REPO_ID,
        repo_type = "dataset",
        token = TOKEN_ID
    )
    os.remove(local_path)
    print(f"Done uploading Shard {shard_id}")
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':

        api = HfApi()

        n_train_shards = BUCKET_LENGTH // SHARD_SIZE
        if BUCKET_LENGTH % SHARD_SIZE != 0:
            n_train_shards+=1

        print("Streaming shards to HF")
        b=0
        for bucket in BUCKETS:
            for i in range(n_train_shards):
                shard_gen = generator(SHARD_SIZE, bucket, seed = 1337+b)
                writer(shard_gen, i, "test", api, bucket)
                b+=1
    
        print("Done uploading files to Hugging Face")
