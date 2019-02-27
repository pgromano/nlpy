from __future__ import division
cimport cython 


#from torch.utils import data
#
#class Dataset(data.Dataset):
#  'Characterizes a dataset for PyTorch'
#  def __init__(self, files):
#        'Initialization'
#
#        self.files = files
#
#  def __len__(self):
#        'Denotes the total number of samples'
#
#        return len(self.files)
#
#  def __getitem__(self, index):
#        'Generates one sample of data'
#
#        # Select sample
#        file = self.files[index]
#
#        # Load data and get label
#
#        return open(file, 'r').read()


def ngrams(text, int n=2):
    assert n > 1, "N must be positive integer of at least 2"

    # Get number of tokens within document
    cdef int n_tokens = len(text)

    # Return output
    return [text[k:k + n] for k in range(n_tokens - n + 1)]


def left_context(text, int window_size=1):
    """ Generate left context and target

    Arguments
    ---------
    text : list(str)
        The tokenized text, or document.
    window_size : int
        The amount of tokens to keep within the context left of the target 
        word.

    Returns
    -------
    context
    """

    # Get number of tokens within document
    cdef int n_tokens = len(text)

    # Return output
    return [(text[k:k + window_size], text[k + window_size]) for k in range(n_tokens - window_size)]


def right_context(text, int window_size=1):
    """ Generate right context and target

    Arguments
    ---------
    text : list(str)
        The tokenized text, or document.
    window_size : int
        The amount of tokens to keep within the context right of the target 
        word.

    Returns
    -------
    context
    """

    # Get number of tokens within document
    cdef int n_tokens = len(text)

    # Return output
    return [(text[k + 1:k + window_size + 1], text[k]) for k in range(n_tokens - window_size)]


def context(text, int window_size=1):
    """ Generate right context and target

    Arguments
    ---------
    text : list(str)
        The tokenized text, or document.
    window_size : int
        The amount of tokens to keep within the context on both the left and
        right of the target respectively. Note the true window size is 
        actually two times the size of the window size.

    Returns
    -------
    context
    """

    # Get number of tokens within document
    cdef int n_tokens = len(text)

    # Return output
    return [(text[k:k + window_size] + text[k + window_size + 1:k + 2 * window_size + 1], text[k + window_size]) for k in range(n_tokens - 2 * window_size)]