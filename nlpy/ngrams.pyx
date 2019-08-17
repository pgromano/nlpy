from __future__ import division
cimport cython 


def ngrams(document, int n=2):
    assert n > 1, "N must be positive integer of at least 2"

    # Get number of tokens within document
    cdef int n_tokens = len(document)

    # Return output
    return [document[k:k + n] for k in range(n_tokens - n + 1)]


def left_context(document, int window_size=1):
    """ Generate left context and target

    Arguments
    ---------
    document : list(str)
        The tokenized document, or document.
    window_size : int
        The amount of tokens to keep within the context left of the target 
        word.

    Returns
    -------
    context
    """

    # Get number of tokens within document
    cdef int n_tokens = len(document)

    # Return output
    return [(document[k:k + window_size], document[k + window_size]) for k in range(n_tokens - window_size)]


def right_context(document, int window_size=1):
    """ Generate right context and target

    Arguments
    ---------
    document : list(str)
        The tokenized document, or document.
    window_size : int
        The amount of tokens to keep within the context right of the target 
        word.

    Returns
    -------
    context
    """

    # Get number of tokens within document
    cdef int n_tokens = len(document)

    # Return output
    return [(document[k + 1:k + window_size + 1], document[k]) for k in range(n_tokens - window_size)]


def context(document, int window_size=1):
    """ Generate right context and target

    Arguments
    ---------
    document : list(str)
        The tokenized document, or document.
    window_size : int
        The amount of tokens to keep within the context on both the left and
        right of the target respectively. Note the true window size is 
        actually two times the size of the window size.

    Returns
    -------
    context
    """

    # Get number of tokens within document
    cdef int n_tokens = len(document)

    # Return output
    return [(document[k:k + window_size] + document[k + window_size + 1:k + 2 * window_size + 1], document[k + window_size]) for k in range(n_tokens - 2 * window_size)]