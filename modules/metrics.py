from typing import List
from pathlib import Path
from tqdm import trange
import numpy as np
import torch
import os
import gc

from utils.common_utils import truncate_encoded


# =========================
# device utils
# =========================

def get_default_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def empty_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu":
        torch.npu.empty_cache()


# =========================
# helper
# =========================

def calculate_cum_size(arrays):
    cum_size = []
    start_index = 0
    for array in arrays:
        end_index = start_index + len(array)
        cum_size.append([start_index, end_index])
        start_index = end_index
    return cum_size


# =========================
# abstract scorer
# =========================

class AbstractScorer(object):
    def score(self, references: List[str], hypothesizes: List[str]):
        raise NotImplementedError

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        raise NotImplementedError


# =========================
# BLEURT scorer
# =========================

class BleurtScorer(AbstractScorer):
    def __init__(self, ckpt_path: Path, batch_size: int = 32, device=None):
        from bleurt_pytorch import (
            BleurtForSequenceClassification,
            BleurtTokenizer,
        )

        self.device = device or get_default_device()
        self.batch_size = batch_size

        self.bleurt_scorer = BleurtForSequenceClassification.from_pretrained(
            ckpt_path, trust_remote_code=True
        ).to(self.device)

        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True
        )

        self.bleurt_scorer.eval()

    def load_device(self):
        self.bleurt_scorer.to(self.device)

    def offload_device(self):
        self.bleurt_scorer.to("cpu")
        empty_cache(self.device)

    def _score(self, references, hypothesizes, verbose=False):
        scores = []

        for i in trange(
            0,
            len(references),
            self.batch_size,
            desc="BLEURT scoring",
            disable=not verbose,
        ):
            inputs = self.bleurt_tokenizer(
                references[i : i + self.batch_size],
                hypothesizes[i : i + self.batch_size],
                return_tensors="pt",
                padding="longest",
            ).to(self.device)

            trunc_inputs = truncate_encoded(inputs)

            with torch.no_grad():
                cur_scores = (
                    self.bleurt_scorer(**trunc_inputs)
                    .logits.flatten()
                    .tolist()
                )

            scores.extend(cur_scores)

        return scores

    def score(self, references: List[str], hypothesizes: List[str], keepdims=False):
        scores = self._score(references, hypothesizes)
        scores = np.array(scores)
        return scores if keepdims else scores.mean()

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        cum_size = calculate_cum_size(references)

        cum_references = [r for refs in references for r in refs]
        cum_hypothesizes = [h for hypos in hypothesizes for h in hypos]

        scores = self._score(cum_references, cum_hypothesizes)

        avg_scores = []
        for start, end in cum_size:
            avg_scores.append(np.array(scores[start:end]).mean())

        return avg_scores


# =========================
# COMET scorer
# =========================

class CometScorer(AbstractScorer):
    def __init__(self, ckpt_path: Path, batch_size: int = 128, device=None):
        import comet

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = device or get_default_device()
        self.batch_size = batch_size

        self.comet_scorer = comet.load_from_checkpoint(
            ckpt_path, reload_hparams=True
        )

        self.comet_scorer.to(self.device)
        self.comet_scorer.eval()

    def load_device(self):
        self.comet_scorer.to(self.device)

    def offload_device(self):
        self.comet_scorer.to("cpu")
        empty_cache(self.device)

    def _score(self, references, hypothesizes):
        data = [
            {"src": ref, "mt": hypo}
            for ref, hypo in zip(references, hypothesizes)
        ]

        with torch.no_grad():
            comet_output = self.comet_scorer.predict(
                data,
                batch_size=self.batch_size,
                progress_bar=False,
            )

        return comet_output.scores

    def score(self, references: List[str], hypothesizes: List[str], keepdims=False):
        scores = np.array(self._score(references, hypothesizes))
        return scores if keepdims else scores.mean()

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        cum_size = calculate_cum_size(references)

        cum_references = [r for refs in references for r in refs]
        cum_hypothesizes = [h for hypos in hypothesizes for h in hypos]

        scores = self._score(cum_references, cum_hypothesizes)

        avg_scores = []
        for start, end in cum_size:
            avg_scores.append(np.array(scores[start:end]).mean())

        return avg_scores
