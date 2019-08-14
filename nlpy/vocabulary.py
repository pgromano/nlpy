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
    """
 
    def __init__(self, vocab=None, unknown_token=None, unknown_index=None, 
                 bos_token=None, eos_token=None, sep_token=None, 
                 pad_token=None, cls_token=None, mask_token=None):
        
        # Set attributes
        self.unknown_token = unknown_token
        self.unknown_index = unknown_index
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        
        # Initialize model
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
        
        if not isinstance(index, int):
            raise ValueError("Index must be integer type")
            
        return self._decoder.get(index, self.unknown_token)

    @classmethod
    def from_file(cls, vocab_path, unknown_token=None, bos_token=None, 
                  eos_token=None, sep_token=None, pad_token=None, 
                  cls_token=None, mask_token=None):
        
        if isinstance(vocab_path, str):
            vocab_path = vocab_path.replace("~", str(Path.home()))

        # Check file path
        vocab_path = Path(vocab_path)
        assert vocab_path.exists(), "Cannot find path to file"
        assert vocab_path.is_file(), "Path provide paht to file"
        
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
            unknown_token,
            bos_token,
            eos_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
        )
    
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
                for token in self:
                    vocab_file.write(f"{token}\n")
 
    def replace(self, old_token, new_token):
        if old_token not in self._encoder:
            raise ValueError(f"\"{old_token}\" not found in vocabulary")

        # Get original index value
        val = self._encoder[old_token]
        
        # Remove old token from vocabulary
        self._encoder.pop(old_token)
        
        # Add new token in vocabulary
        self._encoder[new_token] = val

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

        index = len(self._encoder)
        if self.unknown_token not in self._encoder and self.unknown_token is not None:
            self.unknown_index = index
            self._encoder[self.unknown_token] = index
            self._decoder[index] = self.unknown_token
            index += 1

        if self.bos_token not in self._encoder and self.bos_token is not None:
            self._encoder[self.bos_token] = index
            self._decoder[index] = self.bos_token
            index += 1

        if self.eos_token not in self._encoder and self.eos_token is not None:
            self._encoder[self.eos_token] = index
            self._decoder[index] = self.eos_token
            index += 1

        if self.sep_token not in self._encoder and self.sep_token is not None:
            self._encoder[self.sep_token] = index
            self._decoder[index] = self.sep_token
            index += 1

        if self.pad_token not in self._encoder and self.pad_token is not None:
            self._encoder[self.pad_token] = index
            self._decoder[index] = self.pad_token
            index += 1

        if self.cls_token not in self._encoder and self.cls_token is not None:
            self._encoder[self.cls_token] = index
            self._decoder[index] = self.cls_token
            index += 1
        
        if self.mask_token not in self._encoder and self.mask_token is not None:
            self._encoder[self.mask_token] = index
            self._decoder[index] = self.mask_token
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

    def __str__(self):
        return "Vocabulary(size={})".format(self.size)
 
    def __repr__(self):
        return self.__str__()