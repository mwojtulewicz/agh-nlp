import multiprocessing as mp
import pickle
from collections import Counter

import numpy as np
import tqdm


def load_counter(name) -> Counter:
    with open(f"{name}.p", "rb") as f:
        return pickle.load(f)


def dump_results(results: list, name: str) -> None:
    with open(f"{name}.p", "wb") as f:
        pickle.dump(results, f)


def p_bigram(bigram: tuple[str, str]) -> float:
    return bigrams_filtered_counter[bigram] / bigrams_counter.total()


def p_token(tok: str) -> float:
    return tokens_counter[tok] / tokens_counter.total()


def pmi_bigram(bigram: tuple[str, str]) -> tuple[tuple[str, str], float]:
    a, b = bigram
    p_a = p_token(tok=a)
    p_b = p_token(tok=b)
    p_ab = p_bigram(bigram=bigram)
    pmi = np.log2(p_ab / (p_a * p_b))
    return bigram, pmi


if __name__ == "__main__":

    mp.freeze_support()

    for prefix in ["", "tag_"]:

        bigrams_counter = load_counter(name=f"{prefix}bigrams_counter")
        bigrams_filtered_counter = load_counter(
            name=f"{prefix}bigrams_filtered_counter"
        )
        tokens_counter = load_counter(name=f"{prefix}tokens_counter")

        pmi_results = []
        with mp.Pool(processes=10) as pool:
            with tqdm.tqdm(total=len(bigrams_filtered_counter)) as pbar:
                for bigram, pmi in pool.imap_unordered(
                    pmi_bigram, bigrams_filtered_counter
                ):
                    pmi_results.append((bigram, pmi))
                    pbar.update()

        dump_results(results=pmi_results, name=f"{prefix}bigram_pmi_results")
