from . import tokenize
import re


IS_ALPHA = lambda text: re.compile('\d+').findall(text) != []


class Token(object):
    def __init__(self, text, index=None, start=False, is_stop=False):
        self.text = text
        self.index = index
        self.start = start
        self.is_stop = is_stop


class Document(object):
    def __init__(self, text, pattern=None, method='match', stopwords=None):
        self.text = text
        self.pattern = pattern
        self.method = method
        if stopwords is None:
            stopwords = set()
        self.stopwords = set(stopwords)

    def __iter__(self):
        tokens = tokenize(self.text, pattern=self.pattern, method=self.method)
        for index, token in enumerate(tokens):
            yield Token(
                token, 
                index=index, 
                is_stop=(token in self.stopwords),
                start=(index == 0), 
                stop=(index == len(tokens))
            )