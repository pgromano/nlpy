
from __future__ import unicode_literals, print_function
from cymem.cymem cimport Pool
from .vocabulary import Vocabulary
import pathlib
from collections import OrderedDict

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
    cdef public mem
    
    def __init__(self, vocab=None, lowercase=False, special_tokens=None, 
                 pad_punctuation=True, remove_punctuation=False,
                 unknown_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]',
                 sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', 
                 mask_token='[MASK]'):
        
        # Prepare vocabulary and Vocabulary instance
        self.lowercase = lowercase
        self.unknown_token = unknown_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.vocab = self._prepare_vocab(vocab)
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
            
        # Ignore tokens for splitting
        cdef object ignore_split = self.vocab + \
            self.special_tokens + \
            {
                self.unknown_token,
                self.bos_token,
                self.eos_token,
                self.sep_token,
                self.pad_token,
                self.cls_token,
                self.mask_token
            }

        cdef object split_tokens = []
        for token in document.split():
            if token in ignore_split:
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

    def pipe(self, object corpus):
        cdef int n_docs = len(corpus)
        cdef int doc_index = 0
        while doc_index < n_docs:
            yield self.tokenize(corpus[doc_index])
            doc_index += 1
    
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

    def _prepare_vocab(self, vocab):
        if isinstance(vocab, Vocabulary):
            return vocab
        elif isinstance(vocab, (list, tuple, set, dict, OrderedDict)) or vocab is None:
            return Vocabulary(
                vocab,
                unknown_token = self.unknown_token,
                bos_token = self.bos_token,
                eos_token = self.eos_token,
                sep_token = self.sep_token,
                pad_token = self.pad_token,
                cls_token = self.cls_token,
                mask_token = self.mask_token,
            )
        elif isinstance(vocab, str):
            vocab_path = pathlib.Path(vocab)

            # Load vocab if file given
            if vocab_path.is_file():
                return Vocabulary.from_file(
                    vocab_path,
                    unknown_token = self.unknown_token,
                    bos_token = self.bos_token,
                    eos_token = self.eos_token,
                    sep_token = self.sep_token,
                    pad_token = self.pad_token,
                    cls_token = self.cls_token,
                    mask_token = self.mask_token,
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
                    unknown_token = self.unknown_token,
                    bos_token = self.bos_token,
                    eos_token = self.eos_token,
                    sep_token = self.sep_token,
                    pad_token = self.pad_token,
                    cls_token = self.cls_token,
                    mask_token = self.mask_token,
                )
        else:
            raise ValueError("Unable to interpret vocabulary")

    def __call__(self, document):
        return self.tokenize(document)

    def __repr__(self):
        return "Tokenizer(lowercase={}, punctuations={}, special_tokens={}".format(
            self.lowercase, self.punctuations, self.special_tokens
        )