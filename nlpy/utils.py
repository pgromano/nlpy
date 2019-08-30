from __future__ import unicode_literals, print_function
import pathlib
from collections import OrderedDict
import string
import requests
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


def download_url(url, path, overwrite=False, chunk_size=1024*1024, timeout=4, 
                 retries=5):
    """ Download `url` to `dest` unless it exists and not `overwrite`.
    
    """
    
    if pathlib.Path(path).exists() and not overwrite: 
        return None

    session = requests.Session()
    session.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    url = session.get(url, stream=True, timeout=timeout)
    try: 
        file_size = int(url.headers["Content-Length"])
    except: 
        show_progress = False

    with open(path, 'wb') as file:
        for chunk in url.iter_content(chunk_size=chunk_size):
            file.write(chunk)

def prepare_vocabulary(vocab, unknown_token=None, bos_token=None, 
                       eos_token=None, sep_token=None, pad_token=None, 
                       cls_token=None, mask_token=None, special_tokens=None):

    """ Method to prepare various containers for nlpy.Vocabulary 
    
    Arguments
    ---------
    vocab : file-like str or container, optional
        The vocabulary which defines non-splitable tokens. This can be a path
        to a vocabulary text or json file, or a set, tuple, list, dictionary
        etc. 
    unknown_token : str, optional
        The special token to handle out of vocabulary tokens. 
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
    """

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