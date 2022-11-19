import multiprocessing as mp
import pickle
from collections import Counter
from itertools import repeat

import numpy as np
from tqdm import tqdm

N_PROCESSES = 4


def load_counter(name) -> Counter:
    with open(f"saved/{name}.p", "rb") as f:
        return pickle.load(f)


def dump_results(results: list, name: str) -> None:
    with open(f"saved/{name}.p", "wb") as f:
        pickle.dump(results, f)


def p_bigram(bigram: tuple[str, str], all_bigrams: Counter) -> float:
    return all_bigrams[bigram] / all_bigrams.total()


def p_token(tok: str, all_tokens: Counter) -> float:
    return all_tokens[tok] / all_tokens.total()


def pmi_bigram(
    bigram: tuple[str, str], all_bigrams: Counter, all_tokens: Counter
) -> tuple[tuple[str, str], float]:
    a, b = bigram
    p_a = p_token(tok=a, all_tokens=all_tokens)
    p_b = p_token(tok=b, all_tokens=all_tokens)
    p_ab = p_bigram(bigram=bigram, all_bigrams=all_bigrams)
    pmi = np.log2(p_ab / (p_a * p_b))
    return bigram, pmi


if __name__ == "__main__":

    for prefix in ["", "tag_"]:

        bigrams_counter = load_counter(name=f"{prefix}bigrams_counter")
        bigrams_filtered_counter = load_counter(
            name=f"{prefix}bigrams_filtered_counter"
        )
        tokens_counter = load_counter(name=f"{prefix}tokens_counter")

        with mp.Pool(processes=N_PROCESSES) as pool:
            inputs = zip(
                bigrams_filtered_counter,
                repeat(bigrams_counter),
                repeat(tokens_counter),
            )
            pmi_results = pool.starmap(
                pmi_bigram,
                tqdm(
                    inputs,
                    total=len(bigrams_filtered_counter),
                    desc=f"Calculating PMI [{prefix}bigrams]",
                ),
            )

        dump_results(results=pmi_results, name=f"{prefix}bigram_pmi_results")
