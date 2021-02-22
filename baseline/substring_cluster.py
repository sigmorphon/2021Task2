from argparse import ArgumentParser
from typing import Dict, List, Set, Tuple


def read_bible_text(fn: str) -> Tuple[List, Set]:
    """Read in the bible text file, returning the lines in the corpus,
    and the unique vocab Set"""
    print(f"Reading corpus {fn.split('/')[-1]}... ")
    corpus = []
    vocab = set()
    with open(fn) as f:
        for line in f:
            line = [w.lower() for w in line.strip().split()]
            corpus.append(line)
            for w in line:
                vocab.add(w)
    print(f"Read corpus with {len(corpus)} lines, with vocab of size {len(vocab)}.\n")

    return corpus, vocab


def make_ngram_dicts(vocab, n) -> Tuple[Dict, Dict]:
    ngram2words, words2ngrams = {}, {}
    for word in vocab:
        # Words < n-gram size should just be their own cluster
        if len(word) < n:
            ngram2words[word] = set([word])
            words2ngrams[word] = set([word])

        for i in range(len(word)-n+1):
            n_gram = word[i:i+n]
            ngram2words.setdefault(n_gram, set()).add(word)
            words2ngrams.setdefault(word, set()).add(n_gram)

    return ngram2words, words2ngrams


def main(bible_fn: str, output_fn: str, n: int):
    corpus, vocab = read_bible_text(bible_fn)

    # 1. Build dictionaries based on n-grams
    #    {word: [ngrams, for all word ngram combos]}
    #    {ngram: [words] for all words with that ngram}
    ngram2words, words2ngrams = make_ngram_dicts(vocab, n)

    # 2. Sample down to remove duplicate clusters of words
    #    (words that share >1 ngram and thus form multiple clusters
    #    with exactly those words)
    #
    # For quick lookup
    unique_clusters_set = set(tuple(x) for x in ngram2words.values())
    unique_clusters = {}
    for ngram, words in ngram2words.items():
        if words not in unique_clusters_set:
            unique_clusters[ngram] = words
        else:
            # Remove the ngrams from the word dict for later checking in step 3
            for word in words:
                words2ngrams[word].remove(ngram)

    # 3. Remove clusters of single words, for words that belong to other clusters.
    #    This is to avoid useless extra clusters determined by the many
    #    unique ngrams a word might have
    for word, ngrams in words2ngrams.items():
        # Check if the word belongs to multiple ngrams
        if len(ngrams) > 1:
            for ngram in ngrams:
                # remove the ngrams whose cluster is exactly 1
                # Ensuring every word still belongs to at least 1 ngram cluster
                if ngram in unique_clusters \
                and len(unique_clusters[ngram]) == 1 \
                and len(words2ngrams[word]) > 1:
                    unique_clusters.pop(ngram)

    # Format as strings, and write results
    paradigms = []
    for s, words in unique_clusters.items():
        paradigms.append("\n".join(words))

    print(f"Writing paradigm predictions to {output_fn}...")
    with open(output_fn, "w") as out:
        out.write("\n\n".join(paradigms))


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--bible-text', type=str,
                        help="The bible.txt file")
    parser.add_argument('--output', type=str,
                        help="The file to write results to")
    parser.add_argument('--n', type=int,
                        help="The lowest number of chars two words must have in common.")
    args = parser.parse_args()

    main(args.bible_text, args.output, args.n)
