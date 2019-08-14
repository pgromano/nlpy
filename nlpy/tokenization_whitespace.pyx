
from __future__ import unicode_literals, print_function
from cymem.cymem cimport Pool
from .vocabulary import Vocabulary
import pathlib
from collections import OrderedDict

cdef class WhitespaceTokenizer:

    """ Whitespace Tokenizer

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
    cdef public mem
    
    def __init__(self, vocab=None, lowercase=False, unknown_token='[UNK]', 
                 pad_token='[PAD]', pad_length=-1):
        
        # Prepare vocabulary and Vocabulary instance
        self.lowercase = lowercase
        self.unknown_token = unknown_token
        self.pad_token = pad_token
        self.vocab = self._prepare_vocab(vocab)
        self.pad_length = pad_length
        self.mem = Pool()

    def tokenize(self, object document):
        return document.split()

    def encode(self, object document):
        cdef object tokens = self.tokenize(document)
        cdef int n_tokens = len(tokens)
        cdef int pad_length = self.pad_length
        if pad_length == -1:
            pad_length = n_tokens
        sequence = [
            self.vocab.encode(tokens[i]) if i < n_tokens
            else self.vocab.encode(self.pad_token)
            for i in range(pad_length)
        ]
        return sequence

    def decode(self, object sequence):
        tokens = [
            self.vocab.decode(key)
            for key in sequence
        ]
        return tokens

    def pipe(self, object corpus):
        cdef int n_docs = len(corpus)
        cdef int doc_index = 0
        while doc_index < n_docs:
            yield self.tokenize(corpus[doc_index])
            doc_index += 1

    def _prepare_vocab(self, vocab):
        if isinstance(vocab, Vocabulary):
            return vocab
        elif isinstance(vocab, (list, tuple, set, dict, OrderedDict)) or vocab is None:
            return Vocabulary(
                vocab, 
                unknown_token=self.unknown_token,
                pad_token=self.pad_token
            )
        elif isinstance(vocab, str):
            vocab_path = pathlib.Path(vocab)

            # Load vocab if file given
            if vocab_path.is_file():
                return Vocabulary.from_file(
                    vocab_path, 
                    unknown_token=self.unknown_token,
                    pad_token=self.pad_token
                )

            # Maaaybe load vocab if directory given (with appropriate file inside)
            if vocab_path.is_dir():
                if (vocab_path/"vocab.txt").exists():
                    vocab_file = str(vocab_path/"vocab.txt")
                elif (vocab_path/"vocab.json").exists():
                    vocab_file = str(vocab_path/"vocab.json")
                else:
                    error = "Expected file but directory given. Assumed vocab.txt/json but none found"
                    raise ValueError(error)
                return Vocabulary.from_file(
                    vocab_file, 
                    unknown_token=self.unknown_token,
                    pad_token=self.pad_token
                )
        else:
            raise ValueError("Unable to interpret vocabulary")

    def __call__(self, document):
        return self.tokenize(document)

    def __repr__(self):
        return "Tokenizer(lowercase={}, punctuations={}, special_tokens={}".format(
            self.lowercase
        )