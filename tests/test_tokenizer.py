import pytest
import random
import string


def make_dummy_dataset(size: int, max_seq_len: int):
    """Creates a dummy dataset of random token sequences for testing purposes.
    
    Args:
        size (int): The number of samples in the dataset (in MB).
        max_seq_len (int): The maximum sequence length for each sample.
    Returns:
        Iterator[str]: An Iterator of random strings representing token sequences.
    """
    num_samples = size * 1024 * 1024 // max_seq_len  # Approximate number of samples based on size and sequence length

    for _ in range(num_samples):
        seq_len = random.randint(1, max_seq_len)
        sample = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=seq_len))
        yield sample


@pytest.fixture(scope="module")
def dummy_small():
    return make_dummy_dataset(size=10, max_seq_len=16)

@pytest.fixture(scope="module")
def dummy_large():
    return make_dummy_dataset(size=100, max_seq_len=16)
