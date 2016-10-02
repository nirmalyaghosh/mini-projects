# -*- coding: utf-8 -*-
"""
Summarizes text.

@author: Nirmalya Ghosh (and where indicated, https://gist.github.com/shlomibabluki/5473521)
"""

import spacy

_spacy_parser = None  # Initialize later, as it takes time + ~3GB of RAM


class SimpleSummaryTool(object):
    # Based on the idea proposed (with some code modifications) in the article,
    # https://thetokenizer.com/2013/04/28/build-your-own-summary-tool/
    paragraph_separator = None

    def _build_sentences_dictionary(self, text):
        # Builds the sentences dictionary (key = sentence, value = sentence rank)

        # Calculate the intersection of each sentence with every other sentence
        sentences = self._split_into_sentences(text)
        n = len(sentences)
        values = [[0 for x in iter(range(n))] for x in iter(range(n))]
        for i in range(0, n):
            for j in range(0, n):
                values[i][j] = self._sentence_intersection(sentences[i],
                                                           sentences[j])

        # Build the sentences dictionary
        # The score of a sentences is the sum of all its intersection
        sentences_dict = {}
        for i in range(0, n):
            score = 0
            for j in range(0, n):
                if i == j:
                    continue
                score += values[i][j]
            # sentences_dic[self.format_sentence(sentences[i])] = score
            sentences_dict[sentences[i].strip()] = score

        return sentences_dict

    def _get_best_sentence(self, paragraph, sentences_dict):
        # Get the best sentence in given paragraph (best = highest rank)
        best_sentence = None
        sentences = self._split_into_sentences(paragraph)
        paragraph_wcnt = len(paragraph.split())
        if len(sentences) < 3 and paragraph_wcnt < 20:
            # Most paragraphs have 2-3 sentences, but more than 20 words
            return best_sentence

        max_value = 0
        for sentence in sentences:
            if sentence not in sentences_dict:
                # This shouldn't happen, unless there's a bug in the text split
                continue
            if sentences_dict[sentence] > max_value:
                max_value = sentences_dict[sentence.strip()]
                best_sentence = sentence

        return best_sentence

    def _init_spacy_parser(self):
        global _spacy_parser
        if _spacy_parser is None:
            _spacy_parser = spacy.load("en")  # Takes few seconds + ~3GB of RAM

    def _sentence_intersection(self, sent1, sent2):
        # Credit : https://gist.github.com/shlomibabluki/5473521
        s1 = set(sent1.split(" "))
        s2 = set(sent2.split(" "))
        if (len(s1) + len(s2)) == 0:
            return 0
        return len(s1.intersection(s2)) / ((len(s1) + len(s2)) / 2)

    def _split_into_sentences(self, text):
        self._init_spacy_parser()
        doc = _spacy_parser(text)
        sentences = []
        for s in list(doc.sents):
            sentence = s.text.strip()
            # Paragraphs in the article text are separated by custom separator
            if self.paragraph_separator:
                ss = sentence.split(self.paragraph_separator)
                sentences.extend(ss)
            else:
                sentences.append(sentence)
        return sentences

    def summarize(self, text, custom_paragraph_separator=None):
        # Extract key sentences, rank them, then figure out which sentences
        # best represent the original text
        self.paragraph_separator = custom_paragraph_separator
        sentences_dict = self._build_sentences_dictionary(text)

        # Split the text into paragraphs
        # [The article text is already separated by a custom separator]
        paragraphs = text.split(custom_paragraph_separator)

        # Select the best sentence from each paragraph (best = highest rank)
        selected = []
        for paragraph in paragraphs:
            best_sentence = self._get_best_sentence(paragraph, sentences_dict)
            if best_sentence:
                selected.append(best_sentence)

        summarized_text = " ".join(selected)
        return summarized_text


def summarize(text, method="simple"):
    if method == "simple":
        sst = SimpleSummaryTool()
        paragraph_separator = "  "
        return sst.summarize(text, paragraph_separator)
    else:
        return None
