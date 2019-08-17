from __future__ import unicode_literals, print_function
import numpy as np
cimport numpy as np
from cymem.cymem cimport Pool
from .utils import prepare_vocabulary

cdef class Tokenizer:

    """ Simple Tokenizer

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
    cdef public object special_tokens
    cdef public object pad_punctuation
    cdef public object remove_punctuation
    cdef public object unknown_token
    cdef public object bos_token
    cdef public object eos_token
    cdef public object sep_token
    cdef public object pad_token
    cdef public object cls_token
    cdef public object mask_token
    cdef public int pad_length
    cdef public object pad_type
    cdef public object trunc_type
    cdef public mem
    
    def __init__(self, vocab=None, lowercase=False, special_tokens=None, 
                 pad_punctuation=True, remove_punctuation=False,
                 unknown_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]',
                 sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', 
                 mask_token='[MASK]', pad_length=-1, pad_type='right', 
                 trunc_type='right'):
        
        # Prepare vocabulary and Vocabulary instance
        self.lowercase = lowercase
        self.unknown_token = unknown_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
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
            special_tokens = self.special_tokens
        )
        self.mem = Pool()

        # Handle special tokens and ensure handled as set for O(1) lookup
        if special_tokens is None:
            special_tokens = set()
        if not isinstance(special_tokens, set):
            special_tokens = set(special_tokens)
        self.special_tokens = special_tokens

        # If both pad_punctuation and remove_punctuation are True
        if pad_punctuation == True and remove_punctuation == True:
            pad_punctuation = ".,?!"
            remove_punctuation = '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'

        # Set punctuation removal
        if pad_punctuation == True and remove_punctuation != True:
            pad_punctuation = ".,?!"

        if pad_punctuation != True and remove_punctuation == True:
            remove_punctuation = '.,?!"#$%&\'()*+-/:;<=>@[\\]^_`{|}~'

        if pad_punctuation == False:
            pad_punctuation = ""

        if remove_punctuation == False:
            remove_punctuation = ""

        intersection = set(pad_punctuation) & set(remove_punctuation)
        if len(intersection) > 0:
            raise AttributeError(
                "{} is/are marked for padding and removal.".format(intersection)
            )

        self.pad_punctuation = pad_punctuation
        self.remove_punctuation = remove_punctuation

    def tokenize(self, object document):

        cdef object split_tokens = []
        for token in document.split():
            if token in self.vocab:
                split_tokens.append(token)
            else:
                if self.lowercase:
                    token = token.lower()
                
                if self.pad_punctuation or self.remove_punctuation:
                    token = self._process_punctuations(token)
                    split_tokens.extend(token)
                else:
                    split_tokens.append(token)
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

    def pipe(self, object corpus):
        for document in corpus:
            yield self.tokenize(document)
    
    def _process_punctuations(self, object token):
        
        cdef object sub_tokens = []
        cdef object current_sub_token = ''
        
        cdef int n_char = len(token)
        for i in range(n_char):
            char = token[i]
            # Handle punctuations
            if char in self.pad_punctuation:
                if current_sub_token != '':
                    sub_tokens.append(current_sub_token)
                sub_tokens.append(char)
                current_sub_token = ''
            elif char in self.remove_punctuation:
                if current_sub_token != '':
                    sub_tokens.append(current_sub_token)
                current_sub_token = ''
            else:
                current_sub_token += char
        if current_sub_token != '':
            sub_tokens.append(current_sub_token)
        return sub_tokens

    def __call__(self, document):
        return self.tokenize(document)