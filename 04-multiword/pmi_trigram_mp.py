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


def p_trigram(trigram: tuple[str, str, str]) -> float:
    return trigrams_filtered_counter[trigram] / trigrams_counter.total()


def p_token(tok: str) -> float:
    return tokens_counter[tok] / tokens_counter.total()


def pmi_trigram(trigram: tuple[str, str, str]) -> tuple[tuple[str, str, str], float]:
    a, b, c = trigram
    p_a = p_token(tok=a)
    p_b = p_token(tok=b)
    p_c = p_token(tok=c)
    p_abc = p_trigram(trigram=trigram)
    pmi = np.log2(p_abc / (p_a * p_b * p_c))
    return trigram, pmi


if __name__ == "__main__":

    mp.freeze_support()

    for prefix in ["", "tag_"]:

        trigrams_counter = load_counter(name=f"{prefix}trigrams_counter")
        trigrams_filtered_counter = load_counter(
            name=f"{prefix}trigrams_filtered_counter"
        )
        tokens_counter = load_counter(name=f"{prefix}tokens_counter")

        pmi_results = []
        with mp.Pool(processes=10) as pool:
            with tqdm.tqdm(total=len(trigrams_filtered_counter)) as pbar:
                for bigram, pmi in pool.imap_unordered(
                    pmi_trigram, trigrams_filtered_counter.keys()
                ):
                    pmi_results.append((bigram, pmi))
                    pbar.update()

        dump_results(results=pmi_results, name=f"{prefix}trigram_pmi_results")
