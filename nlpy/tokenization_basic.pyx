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
    bos_token : str, optional
        Beginning of sentence token to identify sentence origin. Defaults to
        [BOS].
    eos_token : str, optional
        End of sentence token to identify sentence origin. Defaults to [EOS].
    sep_token : str, optional
        Separation token. Defaults to [SEP].
    pad_token : str, optional
        The special token to handle encode padding tokens. Defaults to [PAD].
    cls_token : str, optional
        Classifiction token. Defaults to [CLS].
    mask_token : str, optional
        The special token to handle masked tokens. Defaults to [MASK].
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
    from nlpy import BasicTokenizer

    tokenizer = BasicTokenizer(
        lowercase=True, 
        pad_punctuation='.',
        remove_punctuation='!'
    )
    
    tokenizer.tokenize("THIS! IS A TEST...")
    >>> ["this", "is", "a", "test", ".", ".", "."]

    tokenizer("THIS! IS A TEST...")
    >>> ["this", "is", "a", "test", ".", ".", "."]

    The mehtod can then be encoded or decoded using the pad tokens

    ```python
    tokenizer = BasicTokenizer(
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
    ```
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