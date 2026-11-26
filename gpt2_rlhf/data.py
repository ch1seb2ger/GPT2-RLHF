import json
import random
from glob import glob
from itertools import islice

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

OPENAI_PAD_TOKEN_ID = 50259


def load_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def preference_batch(labels, indices, num_labels, pad_id, device):
    rows = [labels[i] for i in indices]
    query = torch.tensor([row["query"] for row in rows], dtype=torch.long, device=device)
    responses = [
        torch.tensor([row[f"sample{i}"] for row in rows], dtype=torch.long, device=device)
        for i in range(num_labels)
    ]
    best = torch.tensor([row["best"] for row in rows], dtype=torch.long, device=device)
    for tokens in [query, *responses]:
        tokens.masked_fill_(tokens == OPENAI_PAD_TOKEN_ID, pad_id)
    return query, responses, best


def parquet_texts(pattern, column):
    paths = sorted(glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    for path in paths:
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=1024, columns=[column]):
                yield from batch.column(0).to_pylist()


def books_generator():
    yield from parquet_texts("datasets/books/plain_text/train/*.parquet", "text")


def shuffled(texts, rng, buffer_size=10_000):
    buffer = list(islice(texts, buffer_size))
    for text in texts:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = text
    rng.shuffle(buffer)
    yield from buffer


class QueryTokens(IterableDataset):
    def __init__(self, tokenizer, query_length, seed):
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        pad_id = self.tokenizer.pad_token_id
        while True:
            for text in shuffled(books_generator(), rng):
                ids = self.tokenizer.encode(text, add_special_tokens=False)[:self.query_length]
                ids += [pad_id] * (self.query_length - len(ids))
                yield torch.tensor(ids, dtype=torch.long)


def query_loader(tokenizer, query_length, batch_size, seed):
    return DataLoader(QueryTokens(tokenizer, query_length, seed), batch_size=batch_size)
