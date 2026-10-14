
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import torch
import json
from typing import List, Tuple


def load_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


# TODO
class ExtractionDataset():

    def __init__(self, tokenizer, split="train") -> None:

        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.data = load_json(f"data/extraction/{split}.json")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass


# TODO
class HistoryDataset():

    def __init__(self, tokenizer, split="train") -> None:

        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.data = load_json(f"data/history/{split}.json")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass


# TODO
class PolicyDataset():

    def __init__(self, tokenizer, split="train") -> None:

        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.data = load_json(f"data/policy/{split}.json")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass


# TODO
class ResponseDataset():

    def __init__(self, tokenizer, split="train") -> None:

        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.data = load_json(f"data/response/{split}.json")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass
