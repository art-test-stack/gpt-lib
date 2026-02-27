from pathlib import Path
import random, pickle
from gpt_lib.utils.default import RANDOM_SEED, CACHE_DIR
from gpt_lib.data.loader import load_datasets
from gpt_lib.data.normalizers import clean_codeparrot_example
from typing import Union, Dict, Callable, Optional

from tqdm import tqdm

class TokenizerCorpus:
    def __init__(
            self, 
            total_chars: int, 
            total_docs: int,
            corpus_dir: Union[str, Path],
            random_seed: int = RANDOM_SEED,
            sources: Optional[dict] = None,
        ):
        if isinstance(corpus_dir, str):
            corpus_dir = Path(corpus_dir)
        if not corpus_dir.suffix == ".txt":
            corpus_dir = corpus_dir.with_suffix(".txt")
        self.corpus_dir = corpus_dir
        self.random_seed = random_seed
        self.total_chars = total_chars
        self.total_docs = total_docs
        self.sources = sources 
    
    @property
    def meta_path(self):
        return self.corpus_dir.with_suffix(".meta.pkl")
    
    def save(self):
        with open(self.meta_path, "wb") as f:
            pickle.dump(self, f)

    def iterator(self, max_chars: Optional[int] = None):
        nb_chars = 0
        with open(self.corpus_dir, "r", errors="ignore") as f:
            for line in f:
                yield line.strip()
                nb_chars += len(line)
                if max_chars is not None and nb_chars >= max_chars:
                    break

    @classmethod
    def from_path(cls, path: Union[Path, str]):
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix == ".txt":
            path = path.with_suffix(".meta.pkl")
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def write_from_sources(
            cls,
            corpus_dir: Union[str, Path],
            sources: Optional[dict] = None, # dict ds_name: weight,
            chars_per_doc: int = 10_000,
            max_chars: int = 1_000_000_000,
            random_seed: int = RANDOM_SEED,
        ):
        if isinstance(corpus_dir, str):
            corpus_dir = Path(corpus_dir)
        if not corpus_dir.suffix == ".txt":
            corpus_dir = corpus_dir.with_suffix(".txt")
        char_count, doc_count = write_corpus_sample(
            sources=sources,
            chars_per_doc=chars_per_doc,
            max_chars=max_chars,
            out_path=corpus_dir,
            random_seed=random_seed,
        )
        meta = cls(
            corpus_dir=corpus_dir,
            total_chars=char_count,
            total_docs=doc_count,
            sources=sources,
        )
        meta.save()
        return meta


def write_corpus_sample(
        sources = None, # dict ds_name: weight
        chars_per_doc=10_000,
        max_chars=1_000_000_000,
        per_dataset_normalizer=None,
        out_path: Path = Path(".data/corpus.txt"),
        split: str = "train",
        show_progress: bool = True,
        random_seed: int = RANDOM_SEED,
    ):

    if not sources:
        sources = [
            { "path": "HuggingFaceFW/fineweb-edu", "weight": 0.7 },
            { "path": "HuggingFaceTB/finemath", "weight": 0.15, "name": "finemath-4plus" },
            { "path": "codeparrot/codeparrot-clean", "weight": 0.15 },
        ]
    ds = load_datasets(sources, split=split)
    
    r = random.Random(random_seed)
    if max_chars == -1:
        max_chars = sum(len(text) for subset in ds.values() for text in subset["text"])
        print(f"Calculated max_chars from datasets: {max_chars}")
    char_count = 0
    doc_count = 0
    it = { name: iter(subset) for name, subset in ds.items() }

    with open(out_path, "w", encoding="utf-8") as fout:
        with tqdm(total=max_chars, desc="Writing corpus", disable=not show_progress) as pbar:
            while char_count < max_chars:
                p = r.random()
                try:
                    for src in sources:
                        weight = src.get("weight", 1.0)
                        if p < weight:
                            s = next(it[src["path"]])
                            break
                        else:
                            p -= weight
                except StopIteration:
                    break
                text = s.get("text") or s.get("content") or ""
                if not text.strip():
                    continue
                doc_count += 1
                if src.get("path") == "codeparrot/codeparrot-clean":
                    text = clean_codeparrot_example(text)

                text = text[-chars_per_doc:] # arbitrary truncation
                if per_dataset_normalizer:
                    text = per_dataset_normalizer(text)
                if not text.strip():
                    continue

                fout.write(text)
                char_count += len(text)
                pbar.update(len(text))
    return char_count, doc_count