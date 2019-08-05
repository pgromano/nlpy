from __future__ import unicode_literals, print_function
from cymem.cymem cimport Pool

_PUNCT = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"

cdef class Tokenizer:

    """ Simple Tokenizer

    Tokenizer to split on whitespaces as well as separate or remove punctuations
    in the process. Additionally, special tokens can be invoked to prevent the
    splitting.

    Arguments
    ---------
    lowercase : bool, optional
        Whether or not the Tokenizer should change casing of all text to lower.
    special_tokens : set
        An iterable container of special tokens that should not be split. Note
        this will only prevent splitting with regards to punctuations. The 
        tokenizer first splits on whitespaces and then splits on punctuations.
        It's highly recommended to use a `set` or `dict` container as the lookup
        is O(1) compared to at least O(N) for `list` and `tuple`.
    punctuations : 
    """

    cdef public bint lowercase
    cdef public object special_tokens
    cdef public object punctuations
    cdef public bint remove_punctuations
    cdef public bint split_punctuations
    cdef public mem

    def __init__(self, lowercase=False, special_tokens=None, punctuations=None,
                remove_punctuations=False, split_punctuations=True):
        self.lowercase = lowercase
        if special_tokens is None:
            special_tokens = set()
        self.special_tokens = special_tokens
        if punctuations is None:
            self.punctuations = _PUNCT
        self.remove_punctuations = remove_punctuations
        if remove_punctuations:
            split_punctuations = True
        self.split_punctuations = split_punctuations
        self.mem = Pool()

    def tokenize(self, object document):
        cdef object tokens = []
        for token in document.split():
            if token in self.special_tokens:
                tokens.append(token)
            else:
                if self.lowercase:
                    token = token.lower()
                
                if self.split_punctuations:
                    token = self._split_punctuations(token)
                    tokens.extend(token)
                else:
                    tokens.append(token)
        return tokens

    def _split_punctuations(self, object document):

        cdef object tokens = []
        cdef object current_token = ""
        cdef int n_char = len(document)
        for i in range(n_char):
            char = document[i]

            if char in self.punctuations:
                if current_token != "":
                    tokens.append(current_token)
                if not self.remove_punctuations:
                    tokens.append(char)
                current_token = ""
            else:
                current_token += char
        if current_token != "":
            tokens.append(current_token)
        return tokens

    def __call__(self, document):
        return self.tokenize(document)

    def __repr__(self):
        return f"Tokenizer(lowercase={self.lowercase}, punctuations={self.punctuations}, special_tokens={self.special_tokens})"

