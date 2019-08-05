import json
from pathlib import Path
import warnings

class Vocabulary:
    """ Vocabulary Object

    Instance to create and manage token-level vocabulary

    Arguments
    ---------
    vocab : iterabl, optional
        An iterable of string tokens to initialize the vocabulary object
    unknown_token : str, optional
        The output string to use if a token is not in the vocabulary. Mostly for
        handling in encoding and decoding of unknown tokens.
    """

    def __init__(self, vocab=None, unknown_token=None):
        
        # Handle unknown attributes
        self.unknown_token = unknown_token
        self.unknown_index = 0 if unknown_token else None

        # Initialize Model
        self._reset(vocab)

    def add(self, token):
        """ Add token to vocabulary

        Given a string token add to vocabulary if not currently present.

        Arguments
        ---------
        token : str
            The token to add into vocabulary
        """

        if token not in self._token_lookup:
            index = self.size
            self._token_lookup[token] = index

    def remove(self, token):
        """ Remove token from vocabulary

        Given a string token, remove it from the vocabulary and re-process the
        token-lookup table to re-adjust the ordering.

        Arguments
        ---------
        token : str
            The token to remove from vocabulary
        """

        if token in self._token_lookup:
            self._token_lookup.pop(token)
            self._reset(self._token_lookup)

    @classmethod
    def from_file(cls, vocab_path, unknown_token=None):
        """ Initialize Vocabulary from File

        Arugments
        ---------
        vocab_path : str or path-like
            Path to vocabulary file. If the extension of the file is json or 
            there is not extension, this method will load it with the json 
            package and assume the given structure is the vocabulary. Otherwise, 
            the file will be read line by line into a dictionary stripping 
            whitespace.

        TODO: Add yaml and csv (predefined index?) support
        """

        # Convert ~ reference to home
        if isinstance(vocab_path, str):
            vocab_path = vocab_path.replace("~", str(Path.home()))

        # Check file path
        vocab_path = Path(vocab_path)
        assert vocab_path.exists(), "Cannot find path to file"
        assert vocab_path.is_file(), "Must provide path to file, not directory"

        with open(vocab_path, 'r') as vocab_file:
            if vocab_path.suffix == '.json' or vocab_path.suffix == '':
                vocab_dict = json.load(vocab_file)
            else:
                vocab_dict = {
                    key.strip(): val 
                    for val, key in enumerate(vocab_file)
                }

        return cls(vocab_dict, unknown_token)

    def to_file(self, vocab_path, overwrite=False):
        """ Save Vocabulary to File

        Arguments 
        ---------
        vocab_path : str or path-like
            Path to where vocabulary will be saved. If the extension of the file
            is json or no extension provided will load the vocabulary as a
            json file assuming the structure is the vocabulary. Otherwise, the
            file will be as a raw string which each token given per line.
        """

        # Convert ~ reference to home
        if isinstance(vocab_path, str):
            vocab_path = vocab_path.replace("~", str(Path.home()))

        # Check file path
        vocab_path = Path(vocab_path)
        if vocab_path.exists() and vocab_path.is_file():
            if overwrite:
                warnings.warn(f"OVERWRITING FILE: {vocab_path}")
            else:
                raise ValueError("File already exists. Either move or rename the file or set overwrite=True")
        
        with open(vocab_path, 'w+') as vocab_file:
            if vocab_path.suffix == '.json' or vocab_path.suffix == '':
                json.dump(self._token_lookup, vocab_file)
            else:
                for token in self:
                    vocab_file.write(f"{token}\n")
    @property
    def size(self):
        return len(self._token_lookup)

    def sort(self, ascending=True):
        sorted_vocab = sorted(self._token_lookup, reverse=not ascending)
        self._reset(sorted_vocab)

    def _reset(self, vocab=None):
        self._token_lookup = {}
        if self.unknown_token is not None:
            self._token_lookup[self.unknown_token] = self.unknown_index
        
        if vocab is not None:
            index = len(self._token_lookup)
            for token in vocab:
                if token not in self._token_lookup:
                    self._token_lookup[token] = index
                    index += 1

    def __contains__(self, token):
        return token in self._token_lookup

    def __iter__(self):
        for token in self._token_lookup:
            yield token

    def __add__(self, vocab):
        if hasattr(vocab, '_token_lookup'):
            add_vocab = set(self._token_lookup) | set(vocab._token_lookup)
        else:
            add_vocab = set(self._token_lookup) | set(vocab)
        return Vocabulary(add_vocab, self.unknown_token)

    def __sub__(self, vocab):
        if hasattr(vocab, '_token_lookup'):
            sub_vocab = set(self._token_lookup) - set(vocab._token_lookup)
        else:
            sub_vocab = set(self._token_lookup) - set(vocab)
        return Vocabulary(sub_vocab, self.unknown_token)

    def __len__(self):
        return self.size

    def __str__(self):
        return f"Vocabulary({self.size})"

    def __repr__(self):
        return self.__str__()