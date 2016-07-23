# -*- coding: utf-8 -*-
"""
Help preprocess the text.

@author: Nirmalya Ghosh
"""

import Queue
import gc
import itertools
import json
import multiprocessing as mp
import os
import os
import re
import string
import sys
import tarfile
import time
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from urlparse import urlparse

import numpy as np
import pandas as pd
import pandas as pd
from functools32 import lru_cache
from gensim import parsing, utils
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.en import English


def count_num_features(dfx, column_name):
    return len(set(list(itertools.chain(*dfx[column_name].values.tolist()))))


def _remove_email_addresses(text, email_regex):
    tmp_txt = text
    try:
        email_addresses = []
        for email_tuple in re.findall(email_regex, tmp_txt):
            email_addr = email_tuple[0]
        email_addresses.append(email_addr)
        for email_addr in email_addresses:
            tmp_txt = tmp_txt.replace(email_addr, " ")
    except:
        pass
    return unicode(str(tmp_txt), errors="ignore")


def _remove_punct(token):
    return re.sub(r"\W+", " ", token).strip()


def _remove_repeated_chars(token):
    # Credit : http://stackoverflow.com/a/10072826
    return re.sub(r'(.)\1+', r'\1\1', token)


def _remove_urls(text):
    processed_text = text
    t = str(text).split()
    t = [el for el in t if
         el.startswith("https") == False and el.startswith("http") == False]
    try:
        t = [el for el in t if not urlparse(el).scheme]
        t = [i for i in t if i.startswith("www") == False]
    except Exception as exc:
        exc_mssg = str(exc)
        print(exc_mssg)
    processed_text = " ".join(t)
    return unicode(str(processed_text), errors="ignore")


def _tokenize_text(q_in, q_out):
    pid = os.getpid()
    print("Process ID {} starting".format(pid))
    # Initialize Spacy - takes a few seconds and ~3GB of RAM
    from spacy.en import English
    _parser_ = English()
    print("Process ID {} ready".format(pid))

    _wnl_ = WordNetLemmatizer()
    _lemmatize_ = lru_cache(maxsize=150000)(_wnl_.lemmatize)

    # Create a set of stop words + symbols, and ensure they all are unicode
    excluded_tokens = set(list(ENGLISH_STOP_WORDS) + ["urllink"])
    excluded_tokens.update(set(set(" ".join(string.punctuation).split(" "))))
    excluded_tokens = set([unicode(word) for word in list(excluded_tokens)])

    # For removing email addresses
    email_regex = re.compile(
        "([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
        "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
        "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)")
    # Regex credit : https://gist.github.com/dideler/5219706

    while True:
        if q_in.empty() == True:
            print("Process ID {} - queue empty".format(pid))
            break
        _tuple_ = q_in.get()
        _id, _text = _tuple_[0], unicode(_tuple_[1], errors="ignore")

        # First, do some basic preprocessing, prior to using Spacy
        # - this is done because using Spacy alone, is slow
        tokens = _text.split()

        # Remove tokens appearing in stop_words or stop_symbols
        tokens = [t for t in tokens if t not in excluded_tokens]

        # Next,
        tokens = [_remove_email_addresses(token, email_regex) for token in
                  tokens]
        tokens = [_remove_urls(token) for token in tokens]
        tokens = [_remove_repeated_chars(token) for token in tokens]
        tokens = [_lemmatize_(token) for token in tokens]

        # Next, reconstructing the text prior to sending to Spacy
        _text = " ".join(tokens)

        # Next, using spaCy to tokenize and lemmatize text
        tokens = _parser_(_text)
        lemmas = []
        for t in tokens:
            lemmas.append(
                t.lemma_.lower().strip() if t.lemma_ != "-PRON-" else t.lower_)
        tokens = lemmas

        # Remove tokens appearing in stop_words or stop_symbols
        # NOTE : Repeating this because spaCy expands tokens
        # Example, I'm ==> I to be
        tokens = [t for t in tokens if t not in excluded_tokens]

        # Remove repeated dots
        tokens = [token for token in tokens if token.count(".")!=len(token)]

        # Next, remove large strings of whitespace, new line characters
        while "" in tokens:
            tokens.remove("")
        while " " in tokens:
            tokens.remove(" ")

        q_out.put((_id, tokens))
        q_in.task_done()


def tokenize_text(data_frame, num_processes=4):
    # Tokenizes text in the "all_posts_text" column
    # Expects columns : "blogger_id", "gender", "all_posts_text"
    ts = time.time()
    input_q = mp.JoinableQueue()
    output_q = mp.JoinableQueue()

    for index, row in data_frame.iterrows():
        input_q.put((row["blogger_id"], row["all_posts_text"]))

    # Setup (then start) list of processes responsible for tokenizing text
    # NOTE : I noticed each instance of Spacy English parser takes up ~3GB
    # of RAM (also verified it, https://github.com/spacy-io/spaCy/issues/100),
    # so increase the number of processes prudently
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=_tokenize_text, args=(input_q, output_q))
        p.daemon = True
        processes.append(p)
    for p in processes:
        p.start()
    print "Starting {} processes for tokenize_text".format(len(processes))

    # Read the items from the output queue
    # - each item is a tuple (id, tokenized text)
    items = []
    while True:
        try:
            item = output_q.get(True, 60)
            items.append(item)
        except Queue.Empty:
            break  # Work done
        output_q.task_done()

    # Construct a DataFrame based on this list of tuples and merge later
    col_name = "tokenized_text"
    _df = pd.DataFrame(items, columns=["blogger_id", col_name])
    data_frame = pd.merge(data_frame, _df, how="left")

    # Fix for Only first 100 entries of list appear when saving as csv with
    # utf-8 encoding
    data_frame[col_name] = map(unicode, data_frame[col_name])

    print "Time taken : {:.2f} seconds".format(time.time() - ts)
    return data_frame


if __name__ == '__main__':
    dir = r"C:\work\projects\ipython_notebooks\misc\word_embeddings\w2v_gender"
    df = pd.read_csv(dir + "\df_before_tokenized_text.txt", sep="\t",
                     index_col=False)
    df.columns = ["blogger_id", "gender", "all_posts_text"]
    df = tokenize_text(df)
    df.to_csv(dir + "\df_with_tokenized_text.txt", encoding="utf-8-sig",
              index=False, sep="\t")
