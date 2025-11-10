import my_gpt.tokenizer.rustbpe as rust_bpe

print(rust_bpe.__dict__)
tokenizer = rust_bpe.bpe("vocab.json", "merges.txt")

ids = tokenizer.encode("hello world")
print(ids)
