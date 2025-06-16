#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from collections import Counter
from math import log2
import re
import os


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sents = sent_tokenize(text)
    words = [w for sent in sents for w in word_tokenize(sent) if w.isalpha()]

    total_sents = len(sents)
    total_words = len(words)
    total_syls = sum(count_syl(w, d) for w in words)

    if total_sents == 0 or total_words == 0:
        return 0.0
    return 0.39 * (total_words/total_sents) + 11.8 * (total_syls/total_words) - 15.59

    

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        return [len([ph for ph in pron if ph[-1].isdigit()]) for pron in d[word]][0]
    else:
        syls = re.findall(r'[aeiouy]+', word)
        return max(1, len(syls))


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    for file_path in path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        filename = file_path.stem
        try:
            nov_title, nov_author, nov_year = filename.split("-")
        except ValueError:
            continue

        data.append({
            "title": nov_title,
            "author": nov_author,
            "year": nov_year,
            "text": text
        })

        df = pd.DataFrame(data)

    return df.sort_values(by="year").reset_index(drop=True)


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    os.makedirs(store_path, exist_ok=True)
    df['parsed'] = df['text'].apply(lambda x: nlp(x))
    df.to_pickle(store_path / out_name)
    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = word_tokenize(text)
    words = [token.lower() for token in tokens if token.isalpha()]
    num_tokens = len(words)
    num_types = len(set(words))
    return num_types/num_tokens if num_tokens > 0 else 0


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subj_count = Counter()
    verb_count = Counter()
    pair_count = Counter()

    all_lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
    counted_lemmas = Counter(all_lemmas)

    pairs = []
    for token in doc: 
        if token.lemma_ == target_verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    subj = child.lemma_.lower()
                    verb = target_verb
                    pairs.append((subj, verb))
                    pair_count[(subj, verb)] += 1
    
    sum_pairs = sum(pair_count.values())
    results = []

    for (subj, verb), count in pair_count.items():
        p_subj = counted_lemmas[subj] / len(all_lemmas)
        p_verb = counted_lemmas[verb] / len(all_lemmas)
        p_pair = count / sum_pairs

        if p_subj > 0 and p_verb > 0 and p_pair > 0: 
            pmi = log2(p_pair / (p_subj * p_verb))
            results.append((subj, pmi))
    
    return sorted(results, key=lambda x: x[1], reverse=True)[:10]


def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []

    for token in doc: 
        if token.lemma_ == verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    subjects.append(child.text.lower())

    top_10 = Counter(subjects).most_common(10)
    return top_10

def objects_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    results = {}

    for i, row in df.iterrows():
        doc = row['parsed']
        objects = []
        for token in doc:
            if token.dep_ in ("obj", "dobj", "pobj"):
                objects.append(token.text.lower())
    
        top_10 = Counter(objects).most_common(10)
        results[row['title']] = top_10
    
    return results


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # path = Path.cwd() / "p1-texts" / "novels"
    # print(path)
    # df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    # print(df.head())
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    #print(objects_counts(df))

    '''
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")'''
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

