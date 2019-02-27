from __future__ import division
import re
cimport cython 

cdef str RE_CHAR = "\w+"
cdef str RE_CONTR = "\'\w+"
cdef str RE_PUNCT = "[.,!?\"']"
cdef str RE_DEFAULT = \
    RE_CHAR + "|" + \
    RE_CONTR + "|" + \
    RE_PUNCT 

def tokenize(str text, str pattern=None, str method='match'):
    """ Tokenize Text

    Arguments
    ---------
    text : str
        Text string to be tokenized
    pattern : str
        Regular expression pattern to tokenize. See method for
        further details.
    method : str, {'match', 'split'}
        Whether or not `pattern` should define token matches
        or if define characters on which to split.

    Returns
    -------
    List of tokens
    """
    if pattern is None:
        pattern = RE_DEFAULT

    # Compile regex tokenizer
    tokenizer = re.compile(pattern)
    
    # Tokenize by pattern match
    if method == 'match':
        return tokenizer.findall(text)

    # Tokenize by pattern split
    elif method == 'split':
        return tokenizer.split(text)
