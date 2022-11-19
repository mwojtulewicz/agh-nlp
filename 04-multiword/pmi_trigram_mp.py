import multiprocessing as mp
import pickle
from collections import Counter
from itertools import repeat

import numpy as np
import tqdm

N_PROCESSES = 4


def load_counter(name) -> Counter:
    with open(f"saved/{name}.p", "rb") as f:
        return pickle.load(f)


def dump_results(results: list, name: str) -> None:
    with open(f"saved/{name}.p", "wb") as f:
        pickle.dump(results, f)


def p_trigram(trigram: tuple[str, str, str], all_trigrams: Counter) -> float:
    return all_trigrams[trigram] / all_trigrams.total()


def p_token(tok: str, all_tokens: Counter) -> float:
    return all_tokens[tok] / all_tokens.total()


def pmi_trigram(
    trigram: tuple[str, str, str], all_trigrams: Counter, all_tokens: Counter
) -> tuple[tuple[str, str, str], float]:
    a, b, c = trigram
    p_a = p_token(tok=a, all_tokens=all_tokens)
    p_b = p_token(tok=b, all_tokens=all_tokens)
    p_c = p_token(tok=c, all_tokens=all_tokens)
    p_abc = p_trigram(trigram=trigram, all_trigrams=all_trigrams)
    pmi = np.log2(p_abc / (p_a * p_b * p_c))
    return trigram, pmi


if __name__ == "__main__":

    for prefix in ["", "tag_"]:

        trigrams_counter = load_counter(name=f"{prefix}trigrams_counter")
        trigrams_filtered_counter = load_counter(
            name=f"{prefix}trigrams_filtered_counter"
        )
        tokens_counter = load_counter(name=f"{prefix}tokens_counter")

        with mp.Pool(processes=N_PROCESSES) as pool:
            inputs = zip(
                trigrams_filtered_counter,
                repeat(trigrams_counter),
                repeat(tokens_counter),
            )
            pmi_results = pool.starmap(
                pmi_trigram,
                tqdm.tqdm(
                    inputs,
                    total=len(trigrams_filtered_counter),
                    desc=f"Calculating PMI [{prefix}trigrams]",
                ),
            )

        dump_results(results=pmi_results, name=f"{prefix}trigram_pmi_results")
