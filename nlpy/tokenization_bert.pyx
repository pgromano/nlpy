# coding=utf-8
# This is a modification of the following code.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import unicode_literals, print_function
import numpy as np
cimport numpy as np
from cymem.cymem cimport Pool
from .utils import prepare_vocabulary, is_punctuation

cdef class BertTokenizer:
    """ BERT Tokenization

    Constructs the onepiece tokenization scheme from BERT 2019 [1].

    Arguments
    ---------
    vocab : file-like str or container, optional
        The vocabulary which defines non-splitable tokens. This can be a path
        to a vocabulary text or json file, or a set, tuple, list, dictionary
        etc. 
    lowercase : bool, optional
        Whether or not tokens should be lowercased on tokenization.
    pad_punctuation : str or bool, optional
        Whether or not punctuations should be "padded" with whitespaces. For
        tokenization this separates punctuations away from traditional word-like
        tokens. If True then ".,?!" are padded and if False nothing is padded.
        A string of characters can be passed as well to provided more specific
        padding. In such cases, remove_punctuation should be set to False or
        a string of remove characters should be provided to avoid overlap
        between the two parameters.
    remove_punctuation : str or bool, optional
        Whether or not punctuations should be removed. If True and 
        pad_punctuation is True or a string, then 
        '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~' are removed. If pad_punctuation is 
        False and remove_punctuation is True then all punctuations are removed. 
        Special care should be taken to ensure that pad_punctuation and 
        remove_punctuation do not overlap!
    special_tokens : container, optional
        A secondary list of special tokens that are neither default special
        tokens (unknown, mask, etc) nor present in the vocabulary. This can
        be beneficial when starting from a predefined vocabulary that needs
        additional tokens for niche language.
    unknown_token : str, optional
        The special token to handle out of vocabulary tokens. Defaults to [UNK].
    sep_token : str, optional
        Separation token. Defaults to [SEP].
    pad_token : str, optional
        The special token to handle encode padding tokens. Defaults to [PAD].
    cls_token : str, optional
        Classifiction token. Defaults to [CLS].
    mask_token : str, optional
        The special token to handle masked tokens. Defaults to [MASK].
    max_char_per_token : int, optional
        The number of characters in a token before onepiece tokenization ignores
        and returns unknown_token.
    piece_buffer : str, optional
        The buffer type to identify splits in word pieces. Defaults to ##, e.g.
        'testing' --> ['test', '##ing'].
    pad_length : int, optional
        The number of tokens per document. If less than pad_length, the document
        is padded with pad_token. If -1, then the lenght of longest document in 
        the corpus passed to `encode` is used.
    pad_type : str, optional {'left', 'right'}
        Whether or not padding should be to the left or right of the document,
        i.e. before or after respectively.
    trunc_type : str, optional {'left', 'right'}
        In the case when document length is longer than pad_length, this 
        determiens whether or not documents should be truncated to the left or
        right of the document.

    Examples
    --------
    After initializing the tokenizer, tokenization can be performed by either
    running the `tokenize` method or by directly calling from the object itself.
    The BERT tokenization scheme also uses the onepiece open-vocabulary method
    to handle out-of-vocabulary tokens.

    ```python
    from nlpy import BertTokenizer

    tokenizer = BertTokenizer(
        lowercase=True, 
        pad_punctuation='.',
        remove_punctuation='!'
    )

    tokenizer("TESTING a testerful TeStaStIc TEST!")
    >>> ['testing', 'a', 'test', '##er', '##ful', 'test', '##astic', 'test']

    The mehtod can then be encoded or decoded using the pad tokens

    ```python
    tokenizer = BERTTokenizer(
        vocab = {'a', 'b', 'c', "##a"},
        lowercase=True, 
        pad_length=10,
        pad_type='right'
    )

    tokenizer.encode("a b c aa bb cc!")
    >>> [0, 1, 2, 0, 3, 4, 4, 4, 5, 5, 5, 5]

    tokenizer.decode([0, 1, 2, 0, 3, 4, 4, 4, 5, 5, 5, 5])
    >>> ["a", "b", "c", "a", "##a", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]
    ```
    ```
    """

    cdef public object vocab
    cdef public bint lowercase
    cdef public bint basic_tokenize
    cdef public object special_tokens
    cdef public object unknown_token
    cdef public object sep_token
    cdef public object pad_token
    cdef public object cls_token
    cdef public object mask_token
    cdef public int max_char_per_token
    cdef public object piece_buffer
    cdef public int pad_length
    cdef public object pad_type
    cdef public object trunc_type
    cdef public mem

    def __init__(self, vocab=None, lowercase=False, special_tokens=None, 
                 unknown_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', 
                 cls_token='[CLS]', mask_token='[MASK]', basic_tokenize=True, 
                 piece_buffer="##", max_char_per_token=100, pad_length=-1, 
                 pad_type='right', trunc_type='right'):
        
        self.lowercase = lowercase
        self.basic_tokenize = basic_tokenize
        self.special_tokens = special_tokens
        self.unknown_token = unknown_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.max_char_per_token = max_char_per_token
        self.piece_buffer = piece_buffer
        self.pad_length = pad_length
        self.pad_type = pad_type
        self.trunc_type = trunc_type
        self.vocab = prepare_vocabulary(
            vocab,
            unknown_token = self.unknown_token,
            sep_token = self.sep_token,
            pad_token = self.pad_token,
            cls_token = self.cls_token,
            mask_token = self.mask_token,
            special_tokens = self.special_tokens
        )
        self.mem = Pool()

    def tokenize(self, object doc):
        cdef object split_tokens
        if self.basic_tokenize:
            split_tokens = []
            for token in self._basic_tokenize(doc):
                for sub_token in self._wordpiece_tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self._wordpiece_tokenize(doc)
        return split_tokens

    def encode(self, object corpus):
        
        # Prepare corpus
        if isinstance(corpus, str):
            corpus = [corpus]
        cdef int n_documents = len(corpus)

        # Tokenize corpus
        cdef object tokenized_corpus = [
            self.tokenize(document)
            for document in corpus
        ]

        # Determine maximum number of tokens in corpus
        cdef np.ndarray n_tokens = np.array([
            len(tokens)
            for tokens in tokenized_corpus
        ])
        cdef int max_tokens = n_tokens.max()
        
        # Set pad-length to max_tokens if -1
        cdef int pad_length = self.pad_length
        if pad_length == -1:
            pad_length = max_tokens
            
        # Create sequence array of pad_token
        cdef np.ndarray sequence = np.empty((n_documents, pad_length), dtype=int)
        sequence.fill(self.vocab.encode(self.pad_token))

        cdef int i = 0
        while i < n_documents:
            if self.trunc_type == 'left':
                tokens = tokenized_corpus[i][-pad_length:]
            elif self.trunc_type == 'right':
                tokens = tokenized_corpus[i][:pad_length]

            if n_tokens[i] >= pad_length:
                sequence[i] = [
                    self.vocab.encode(token)
                    for token in tokens
                ]
            else:
                if self.pad_type == 'left':
                    sequence[i, -n_tokens[i]:] = [
                        self.vocab.encode(token)
                        for token in tokens
                    ]
                elif self.pad_type == 'right':
                    sequence[i, :n_tokens[i]] = [
                        self.vocab.encode(token)
                        for token in tokens
                    ]
            i += 1
        return sequence

    def decode(self, object sequence):
        if isinstance(sequence[0], (tuple, list, np.ndarray)):
           return [
               self.vocab.decode(key)
               for keys in sequence
               for key in keys
           ]
        return [self.vocab.decode(key) for key in sequence]

    def pipe(self, object corpus):
        for document in corpus:
            yield self.tokenize(document)

    def _basic_tokenize(self, object document):
        cdef int i
        cdef bint start_new_word
        cdef object output
        
        cdef int n_characters
        cdef object split_tokens = []
        for token in document.split():
            if self.lowercase and token not in self.vocab:
                token = token.lower()
            
            split_tokens.extend(self._punct_tokenize(token))
        output_tokens = (" ".join(split_tokens)).split()
        return output_tokens

    def _punct_tokenize(self, object text):
        if text in self.vocab:
            return [text]

        cdef object character
        cdef object characters = list(text)
        cdef int n_characters = len(characters)
        cdef int i = 0
        cdef bint start_new_word = True
        cdef object output = []

        while i < n_characters:
            character = characters[i]
            if is_punctuation(character):
                output.append([character])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(character)
            i += 1
        return ["".join(x) for x in output]

    def _wordpiece_tokenize(self, object text):
        
        cdef object output_tokens = []
        cdef object characters
        cdef int n_characters
        cdef object cur_substr
        cdef object substr
        cdef bint is_bad
        cdef int start
        cdef int end
        cdef object sub_tokens 

        for token in text.split():
            characters = list(token)
            n_characters = len(characters)
            if n_characters > self.max_char_per_token:
                output_tokens.append(self.unknown_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < n_characters:
                end = n_characters
                cur_substr = None
                while start < end:
                    substr = "".join(characters[start:end])
                    if start > 0:
                        substr = self.piece_buffer + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unknown_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def __call__(self, document):
        return self.tokenize(document)