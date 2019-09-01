from collections import OrderedDict
import copy as copytool
import json
from pathlib import Path
import re
import warnings


class Vocabulary:
    """ Vocabulary Object
 
    Instance to create a token vocabulary
 
    Arguments
    ---------
    vocab : iterable, optional
        An iterable of string tokens to initialize the vocabulary object.
    unknown_token : str, optional
        The output string to use if a token is not in the vocabulary. Mostly
        for handling in encoding and decoding.
    bos_token : str, optional
        Beginning of sentence token to identify sentence origin. 
    eos_token : str, optional
        End of sentence token to identify sentence origin. 
    sep_token : str, optional
        Separation token
    pad_token : str, optional
        The special token to handle encode padding tokens. 
    cls_token : str, optional
        Classifiction token
    mask_token : str, optional
        The special token to handle masked tokens. 
    special_tokens : container, optional
        A secondary list of special tokens that are neither default special
        tokens (unknown, mask, etc) nor present in the vocabulary. This can
        be beneficial when starting from a predefined vocabulary that needs
        additional tokens for niche language.

    Examples
    --------

    The Vocabulary can be thought of as a Python ordered dictionary with methods  
    for handling forward and backward mappings.

    ```python
    from nlpy import Vocabulary

    A = Vocabulary({'a', 'b', 'c'})
    B = Vocabulary({'c', 'd', 'e'})
    print(A, B)
    >>> Vocabulary(size=3) Vocabulary(size=3)

    A.encode('c'), A.decode(2)
    >>> 2, 'c'

    B.encode('c'), B.encode(0)
    >>> 0, 'c'
    ```

    It also provides set operations, allowing for adding and subtracting 
    vocabularies which reproduces set join and differences.

    ```python
    C = A + B
    [(letter, letter in C) for letter in 'abcde']
    >>> [('a', True), ('b', True), ('c', True), ('d', True), ('e', True)]

    D = A - B
    [(letter, letter in D) for letter in 'abcde']
    >>> [('a', True), ('b', True), ('c', False), ('d', False), ('e', False)]

    E = B - A
    [(letter, letter in E) for letter in 'abcde']
    >>> [('a', False), ('b', False), ('c', False), ('d', True), ('e', True)]
    ```
    """
 
    def __init__(self, vocab=None, unknown_token=None, 
                 bos_token=None, eos_token=None, sep_token=None, 
                 pad_token=None, cls_token=None, mask_token=None,
                 special_tokens=None):
        
        # Set attributes
        if isinstance(vocab, Vocabulary):
            self.unknown_token = vocab.unknown_token
            self.unknown_index = vocab.unknown_index
            self.bos_token = vocab.bos_token
            self.eos_token = vocab.eos_token
            self.sep_token = vocab.sep_token
            self.pad_token = vocab.pad_token
            self.cls_token = vocab.cls_token
            self.mask_token = vocab.mask_token
            self.special_tokens = vocab.special_tokens
            self._encoder = vocab._encoder
            self._decoder = vocab._decoder
        else:
            self.unknown_token = unknown_token
            self.unknown_index = None
            self.bos_token = bos_token
            self.eos_token = eos_token
            self.sep_token = sep_token
            self.pad_token = pad_token
            self.cls_token = cls_token
            self.mask_token = mask_token
            self.special_tokens = special_tokens
            self._reset(vocab)

    def copy(self, deep=True):
        if deep:
            return copytool.deepcopy(self)
        return copytool.copy(self)

    def add(self, token):
        """ Add token to vocabulary

        Given a string token add to vocabulary if does not present.

        Arguments
        ---------
        token : str
            The token to add into vocabulary 
        """

        if token not in self._encoder:
            index = self.size
            self._encoder[token] = index
            self._decoder[index] = token

    def remove(self, token):
        """ Remove token from vocabulary

        Given a string token, remove it from the vocabulary and reprocess the
        decoder map to re-adjust the indices.

        Arguments
        ---------
        token : str
            The token to remove from the vocabulary
        """

        if token in self._encoder:
            self._encoder.pop(token)
            self._decoder = {index: token for token, index in self._encoder.items()}
 
    def encode(self, token):
        """ Encode Token
        
        Given a string token return the associated index in the vocabulary
        
        Arguments
        ---------
        token : str
            The token to encode into vocabulary index
        """
        
        if not isinstance(token, str):
            raise ValueError("Token must be a string type")
        
        return self._encoder.get(token, self.unknown_index)
 
    def decode(self, index):
        """ Decode Index
        
        Given an integer index return the associated token in the vocabulary
        
        Arguments
        ---------
        index : int
            The index to decode into vocabulary token
        """
        
        #if not isinstance(index, int):
        #    raise ValueError("Index must be integer type")
            
        return self._decoder.get(index, self.unknown_token)

    @classmethod
    def from_file(cls, vocab_path, unknown_token=None, bos_token=None, 
                  eos_token=None, sep_token=None, pad_token=None, 
                  cls_token=None, mask_token=None, special_tokens=None):
        
        if isinstance(vocab_path, str):
            vocab_path = vocab_path.replace("~", str(Path.home()))

        # Check file path
        vocab_path = Path(vocab_path)
        assert vocab_path.exists(), "Cannot find path to file"
        
        with open(vocab_path, 'r') as vocab_file:
            if vocab_path.suffix == '.json' or vocab_path.suffix == '':
                vocab_dict = json.load(vocab_file)
            else:
                vocab_dict = {
                    key.strip(): val
                    for val, key in enumerate(vocab_file)
                }
        return cls(
            vocab_dict, 
            unknown_token = unknown_token,
            bos_token = bos_token,
            eos_token = eos_token,
            sep_token = sep_token,
            pad_token = pad_token,
            cls_token = cls_token,
            mask_token = mask_token,
            special_tokens = special_tokens
        )
    
    #TODO: to_file does not write to file by index!!!
    def to_file(self, vocab_path, overwrite=False):
        if isinstance(vocab_path, str):
            vocab_path = vocab_path.replace("~", str(Path.home()))

        vocab_path = Path(vocab_path)
        if vocab_path.exists() and vocab_path.is_file():
            if overwrite:
                warnings.warn(f"OVERWRITING FILE: {vocab_path}")
            else:
                raise ValueError("File already exists. Either move, rename, or set overwrite=True")

        with open(vocab_path, 'w+') as vocab_file:
            if vocab_path.suffix == '.json' or vocab_path.suffix == '':
                json.dump(self._encoder, vocab_file)
            else:
                for i in range(self.size):
                    token = self.decode(i)
                    index = self.encode(token)
                    if index != i:
                        raise ValueError(f"Error saving token \"{token}\". Inconsistent index!")
                    vocab_file.write(f"{token}\n")
 
    def replace(self, old_token, new_token, duplicate='raise'):
        """ Replace tokens in vocabulary

        Given an old and new token, replace the encode/decode mappings to the
        new token at the same index. This is useful for pre-trained models
        where old tokens are not relevant and fine-tuning with a new token
        adds niche language capture.

        Arguments
        ---------
        old_token : str
            The token to be replaced
        new_token : str
            The token to replace with
        duplicate : str, optional {'raise', 'warn', 'ignore'}
            How to handle errors when new_token is already present in the
            vocabulary.
        """

        if old_token not in self._encoder:
            raise ValueError(f"\"{old_token}\" not found in vocabulary")

        if new_token in self._encoder:
            if duplicate == 'raise':
                # Break it all
                raise ValueError(f"\"{new_token}\" already in vocabulary!")
            elif duplicate == 'warn':
                # Warn the user
                warnings.warn(f"\"{new_token}\" already in vocabulary! Skipping replacement")
            elif duplicate == 'ignore':
                # Do nothing, hope the user knows what they're doing
                pass
        else:

            # Get original index value
            val = self._encoder[old_token]
            
            # Remove old token from vocabulary
            self._encoder.pop(old_token)
            
            # Add new token in vocabulary and udpate decoder
            self._encoder[new_token] = val
            self._decoder[val] = new_token

    @property
    def size(self):
        return len(self._encoder)

    def search(self, pattern, regex=False):
        if regex:
            return (token for token in self if re.match(pattern, token))
        return (token for token in self if pattern in token)

    def sort(self, ascending=True):
        sorted_vocab = sorted(self._encoder, reverse=not ascending)
        self._reset(sorted_vocab)
 
    @property
    def tokens(self):
        for key in self._encoder.keys():
            yield key
 
    @property
    def index(self):
        for key in self._decoder.keys():
            yield key
            
    def _reset(self, vocab=None):
        
        self._encoder = OrderedDict()
        self._decoder = OrderedDict()

        if isinstance(vocab, (dict, OrderedDict)):
            for token, index in vocab.items():
                self._encoder[token] = index
                self._decoder[index] = token
        elif isinstance(vocab, (list, set, tuple)):
            index = len(self._encoder)
            for token in vocab:
                if token not in self._encoder:
                    self._encoder[token] = index
                    self._decoder[index] = token
                    index += 1

        if self.unknown_token is not None:
            index = len(self._encoder)
            self.unknown_index = index
            if self.unknown_token not in self._encoder:
                self._encoder[self.unknown_token] = index
                self._decoder[index] = self.unknown_token

        if self.bos_token is not None:
            index = len(self._encoder)
            if self.bos_token not in self._encoder:
                self._encoder[self.bos_token] = index
                self._decoder[index] = self.bos_token

        if self.eos_token is not None:
            index = len(self._encoder)
            if self.eos_token not in self._encoder:
                self._encoder[self.eos_token] = index
                self._decoder[index] = self.eos_token

        if self.sep_token is not None:
            index = len(self._encoder)
            if self.sep_token not in self._encoder:
                self._encoder[self.sep_token] = index
                self._decoder[index] = self.sep_token

        if self.pad_token is not None:
            index = len(self._encoder)
            if self.pad_token not in self._encoder:
                self._encoder[self.pad_token] = index
                self._decoder[index] = self.pad_token

        if self.cls_token is not None:
            index = len(self._encoder)
            if self.cls_token not in self._encoder:
                self._encoder[self.cls_token] = index
                self._decoder[index] = self.cls_token

        if self.mask_token is not None:
            index = len(self._encoder)
            if self.mask_token not in self._encoder:
                self._encoder[self.mask_token] = index
                self._decoder[index] = self.mask_token

        if self.special_tokens is not None:
            index = len(self._encoder)
            for token in self.special_tokens:
                if token not in self._encoder:
                    self._encoder[token] = index
                    self._decoder[index] = token
                    index += 1
            
    def __getitem__(self, index):
        return self.decode(index)
    
    def __call__(self, token):
        return self.encode(token)
 
    def __contains__(self, token):
        return token in self._encoder

    def __len__(self):
        return self.size

    def __iter__(self):
        for token in self._encoder:
            yield token
 
    def __eq__(self, vocab):
        if hasattr(vocab, '_encoder'):
            vocab = vocab._encoder

        if len(self) != len(vocab):
            return False

        equality = all([
            token in vocab
            and self[token] == vocab[token]
            for token in self
        ])
        return all(equality)

    def __ne__(self, vocab):
        return not self.__eq__(vocab)

    def __add__(self, vocab):
        add_vocab = self.copy(deep=True)

        if hasattr(vocab, '_encoder'):
            vocab = vocab._encoder

        for token in vocab:
            add_vocab.add(token)
        return add_vocab

    def __sub__(self, vocab):
        sub_vocab = self.copy(deep=True)

        if hasattr(vocab, '_encoder'):
            vocab = vocab._encoder

        for token in vocab:
            if token in sub_vocab._encoder:
                sub_vocab._encoder.pop(token)
        sub_vocab._reset(sub_vocab._encoder)
        return sub_vocab

    def __and__(self, vocab):
        and_vocab = {}
        index = 0
        for token in self + vocab:
            if token in self and token in vocab:
                and_vocab[token] = index
                index += 1
        output = self.copy(deep=True)
        output._reset(and_vocab)
        return output

    def __or__(self, vocab):
        
        or_vocab = {}
        index = 0
        for token in self + vocab:
            if token in self and token not in vocab:
                or_vocab[token] = index
                index += 1
            elif token not in self and token in vocab:
                or_vocab[token] = index
                index += 1
        output = self.copy(deep=True)
        output._reset(or_vocab)
        return output

    def __str__(self):
        return "Vocabulary(size={})".format(self.size)
 
    def __repr__(self):
        return self.__str__()