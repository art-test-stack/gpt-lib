if not __name__ == "__main__":
    raise ImportError("This script is intended to be run as a standalone program.")

from gpt_lib.tokenizer.bpe import bpe, bpe_fast
from gpt_lib.tokenizer.config import TrainingTokenizerConfig
import argparse, pickle
from pathlib import Path
from gpt_lib.tokenizer.corpus import Corpus
from typing import Union, Generator
import time

parser = argparse.ArgumentParser(description="Train BPE tokenizer on a given corpus or evaluate BPE tokenizer against baselines.")
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train or evaluate on (default: 10B)')
parser.add_argument("--chars_per_doc", type=int, default=10_000, help="Maximum number of characters per document to use from the corpus for training or evaluation (default: 10,000).")
parser.add_argument("--vocab_size", type=int, default=32_000, help="Vocabulary size for BPE tokenizer (default: 30,000).")
parser.add_argument("--name", type=str, default="ic1_tokenizer", help="Name of the tokenizer (default: 'ic1_tokenizer').")
parser.add_argument("--corpus_path", type=str, default="./.data/corpus.txt", help="Path to the corpus file (default: './.data/corpus.txt').")
parser.add_argument("--write_corpus", action="store_true", help="Flag to indicate training mode.")
parser.add_argument("--fast", action="store_true", help="Flag to indicate using fast implementation.")
# parser.add_argument("--nb_special_tokens", type=int, default=16, help="Number of special tokens to reserve in the tokenizer vocabulary (default: 16).")
args = parser.parse_args()

config = TrainingTokenizerConfig(
    max_chars=args.max_chars,
    chars_per_doc=args.chars_per_doc,
    vocab_size=args.vocab_size,
    name=args.name,
)

corpus_path = args.corpus_path
if not corpus_path.endswith(".txt"):
    corpus_path = f"{corpus_path}.txt"
corpus_path = Path(corpus_path)


if args.write_corpus:
    corpus = Corpus.write_from_sources(
        name=args.name,
        sources=None,
        chars_per_doc=args.chars_per_doc,
        max_chars=args.max_chars,
    )
    corpus_path = corpus.out_path
else:
    corpus = Corpus.from_name(args.name)
    corpus_path = corpus.out_path

if args.fast:
    _bpe = bpe_fast
else:
    _bpe = bpe

def corpus_iter(path: Union[str, Path] = corpus_path) -> Generator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line.strip()

print(f"Loading corpus from {corpus_path}")
t0 = time.time()
merges, vocab = _bpe(corpus_iter, corpus_path, config)
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")

if not config.tokenizer_dir.exists():
    config.tokenizer_dir.mkdir(parents=True, exist_ok=True)
with open(config.tokenizer_dir / "merges.pkl", "wb") as mf:
    pickle.dump(merges, mf)

with open(config.tokenizer_dir / "vocab.pkl", "wb") as vf:
    pickle.dump(vocab, vf)

print("Trained BPE tokenizer on provided corpus.")
print("Merges", merges)
print("Vocabulary", vocab)
print("Vocab size", len(vocab))
