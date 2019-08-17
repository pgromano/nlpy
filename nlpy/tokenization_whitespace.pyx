
from __future__ import unicode_literals, print_function
import numpy as np
cimport numpy as np
from cymem.cymem cimport Pool
from .utils import prepare_vocabulary

cdef class WhitespaceTokenizer:

    """ Simple Whitespace Tokenizer

    Tokenizer splits on whitespaces as well as punctuations.

    Arguments
    ---------
    lowercase : bool, optional
        Whether or not the Tokenizer should change casing of all text to lower
    special_tokens : set
        An iterable container of special tokens that should not be split. Note
        this will only prevent splitting with regards to punctuations. The 
        tokenizer first splits on whitespaces and then splits on punctuations.
        Special tokens like "[ EXAMPLE ]" would still tokenize to ["[", 
        "EXAMPLE", "]"]. While the tokenizer will accept any iterable container,
        it is highly recommended to use a `set` or `dict` container as the 
        lookup is O(1) compared to at least O(N) for `list` and `tuple`.
    punctuations : str
        A string (or iterable container) of all punctuatinos to separate in 
        tokenization. For example given, punctuations="@" the text "$1.23@test"
        would return ["$1.23", "@", "test"].

    Examples
    --------
    After initializing the tokenizer, tokenization can be performed by either
    running the `tokenize` method or by directly calling from the object itself.

    ```python
    from nlpy import WhitespaceTokenizer

    tokenizer = WhitespaceTokenizer(lowercase=True, punctuations=".")

    >>> tokenizer.tokenize("THIS IS A TEST...")
    ["this", "is", "a", "test", ".", ".", "."]

    >>> tokenizer("THIS IS A TEST...")
    ["this", "is", "a", "test", ".", ".", "."]
    """

    cdef public object vocab
    cdef public bint lowercase
    cdef public object unknown_token
    cdef public object pad_token
    cdef public int pad_length
    cdef public object pad_type
    cdef public object trunc_type
    cdef public mem
    
    def __init__(self, vocab=None, lowercase=False, unknown_token='[UNK]',
                 pad_token='[PAD]', pad_length=-1, pad_type='right', 
                 trunc_type='right'):
        
        # Prepare vocabulary and Vocabulary instance
        self.lowercase = lowercase
        self.unknown_token = unknown_token
        self.pad_token = pad_token
        self.pad_length = pad_length
        self.pad_type = pad_type
        self.trunc_type = trunc_type
        self.vocab = prepare_vocabulary(
            vocab,
            unknown_token = self.unknown_token,
            bos_token = self.bos_token,
            eos_token = self.eos_token,
            sep_token = self.sep_token,
            pad_token = self.pad_token,
            cls_token = self.cls_token,
            mask_token = self.mask_token,
        )
        self.mem = Pool()

    def tokenize(self, document):
        return document.split()
    
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
                    sequence[i, -(pad_length - n_tokens[i]):] = [
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

    def pipe(self, corpus):
        for document in corpus:
            yield self.tokenize(document)

    def __call__(self, document):
        return self.tokenize(document)