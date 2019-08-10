from __future__ import unicode_literals, print_function
from cymem.cymem cimport Pool

cdef class BertTokenizer:
    """ BERT Tokenization

    Constructs the onepiece tokenization scheme from BERT 2019 [1].

    Arguments
    ---------
    """

    cdef public bint lowercase
    cdef public bint basic_tokenize
    cdef public object special_tokens
    cdef public object unknown_token
    cdef public object sep_token
    cdef public object pad_token
    cdef public object cls_token
    cdef public int max_char_per_token
    cdef public object piece_buffer
    cdef public mem

    def __init__(self, vocab=None, lowercase=False, special_tokens=None, 
                 unknown_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]',
                 sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', 
                 basic_tokenize=True, max_char_per_token=100, piece_buffer="##"):
        
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

    def _basic_tokenize(self, doc):
        pass

    def _wordpiece_tokenize(self, doc):
        pass

    def __call__(self, doc):
        return self.tokenize(doc)

