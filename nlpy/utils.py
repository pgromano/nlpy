from __future__ import unicode_literals, print_function
import pathlib
from collections import OrderedDict
import string
from .vocabulary import Vocabulary

def is_punctuation(character):
    if len(character) > 1:
        raise ValueError("Must be character")
    return character in string.punctuation

def has_punctuation(text):
    return any(punct in text for punct in string.punctuation)

def is_whitespace(character):
    if len(character) > 1:
        raise ValueError("Must be character")
    return character in string.whitespace

def has_whitespace(text):
    return any(space in text for space in string.whitespace)

def prepare_vocabulary(vocab, unknown_token=None, bos_token=None, 
                       eos_token=None, sep_token=None, pad_token=None, 
                       cls_token=None, mask_token=None, special_tokens=None):

        if isinstance(vocab, Vocabulary):
            return vocab
        elif isinstance(vocab, (list, tuple, set, dict, OrderedDict)) or vocab is None:
            return Vocabulary(
                vocab,
                unknown_token = unknown_token,
                bos_token = bos_token,
                eos_token = eos_token,
                sep_token = sep_token,
                pad_token = pad_token,
                cls_token = cls_token,
                mask_token = mask_token,
                special_tokens = special_tokens
            )
        elif isinstance(vocab, str):
            vocab_path = pathlib.Path(vocab)

            # Load vocab if file given
            if vocab_path.is_file():
                return Vocabulary.from_file(
                    vocab_path,
                    unknown_token = unknown_token,
                    bos_token = bos_token,
                    eos_token = eos_token,
                    sep_token = sep_token,
                    pad_token = pad_token,
                    cls_token = cls_token,
                    mask_token = mask_token,
                    special_tokens = special_tokens
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
                    unknown_token = unknown_token,
                    bos_token = bos_token,
                    eos_token = eos_token,
                    sep_token = sep_token,
                    pad_token = pad_token,
                    cls_token = cls_token,
                    mask_token = mask_token,
                    special_tokens = special_tokens
                )
        else:
            raise ValueError("Unable to interpret vocabulary")