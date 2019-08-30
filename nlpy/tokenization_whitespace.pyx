
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
    vocab : file-like str or container, optional
        The vocabulary which defines non-splitable tokens. This can be a path
        to a vocabulary text or json file, or a set, tuple, list, dictionary
        etc. 
    lowercase : bool, optional
        Whether or not tokens should be lowercased on tokenization.
    unknown_token : str, optional
        The special token to handle out of vocabulary tokens. Defaults to [UNK].
    pad_token : str, optional
        The special token to handle encode padding tokens. Defaults to [PAD].
    pad_length : int, optional
        The number of tokens per document. If less than pad_length, the document
        is padded with pad_token. If -1, then the lenght of longest document in 
        the corpus passed to `encode` is used. This is only for use with the 
        encode/decode methods and as such is only intended when vocab is not 
        None.
    pad_type : str, optional {'left', 'right'}
        Whether or not padding should be to the left or right of the document,
        i.e. before or after respectively. This is only for use with the 
        encode/decode methods and as such is only intended when vocab is not 
        None.
    trunc_type : str, optional {'left', 'right'}
        In the case when document length is longer than pad_length, this 
        determiens whether or not documents should be truncated to the left or
        right of the document. This is only for use with the encode/decode 
        methods and as such is only intended when vocab is not None.
        
    Examples
    --------
    After initializing the tokenizer, tokenization can be performed by either
    running the `tokenize` method or by directly calling from the object itself.

    ```python
    from nlpy import WhitespaceTokenizer

    tokenizer = WhitespaceTokenizer(
        lowercase=True
    )

    tokenizer("a b c aa bb cc!")
    >>> ["a", "b", "c", "aa", "bb", "cc!"]
    ```

    The mehtod can then be encoded or decoded using the pad tokens

    ```python
    tokenizer = WhitespaceTokenizer(
        vocab = {'a', 'b', 'c'},
        lowercase=True, 
        pad_length=10,
        pad_type='right'
    )

    tokenizer.encode("a b c aa bb cc!")
    >>> [0, 1, 2, 3, 3, 3, 4, 4, 4, 4]

    tokenizer.decode([0, 1, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> ["a", "b", "c", "[UNK]", "[UNK]", "[UNK]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"]
    ```
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