import multiprocessing as mp
import pickle
from collections import Counter

import numpy as np
import tqdm


def dump_counter(counter: Counter, name: str) -> None:
    with open(f"{name}.p", "wb") as f:
        pickle.dump(counter, f)


def load_counter(name) -> Counter:
    with open(f"{name}.p", "rb") as f:
        return pickle.load(f)


def dump_results(results: list, name: str) -> None:
    with open(f"{name}.p", "wb") as f:
        pickle.dump(results, f)


bigrams_counts = load_counter(name="bigrams_counter")
bigrams_filtered_counts = load_counter(name="bigrams_filtered_counter")
tokens_count = load_counter(name="tokens_counter")


def p_bigram(bigram: tuple[str, str]) -> float:
    tok1, tok2 = bigram
    N = sum(bigrams_counts.values())
    return bigrams_filtered_counts[(tok1, tok2)] / N


def p_token(tok: str) -> float:
    N = sum(bigrams_counts.values())
    return tokens_count[tok] / N


def pmi_bigram(bigram: tuple[str, str]) -> tuple[tuple[str, str], float]:
    tok1, tok2 = bigram
    return bigram, np.log2(p_bigram(bigram) / (p_token(tok1) * p_token(tok2)))


if __name__ == "__main__":

    mp.freeze_support()

    pmi_results = []

    with mp.Pool(processes=10) as pool:
        with tqdm.tqdm(total=len(bigrams_filtered_counts)) as pbar:
            for bigram, pmi in pool.imap_unordered(pmi_bigram, bigrams_filtered_counts):
                pmi_results.append((bigram, pmi))
                pbar.update()

    dump_results(results=pmi_results, name="pmi_results")
