cimport cython
from . import tokenize

cdef class TextFileReader(object):
    
    cdef public str encoding
    cdef public list files
    cdef public list filenames

    def __init__(self, str encoding='UTF-8'):
        self.encoding = encoding
        self.files = []
        self.filenames = []

    def load(self, list filenames):
        self.filenames = filenames
        self.files = [
            open(filename, 'r', encoding=self.encoding) 
            for filename in filenames
        ]
        return self

    @property
    def lines(self):
        for file in self.files:
            for line in file:
                yield line

    @property
    def tokens(self):
        for line in self.lines:
            for token in tokenize(line):
                yield token

    def close(self):
        for file in self.files:
            file.close()

    def _reset_files(self):
        self.files = [
            file.seek(0)
            for file in self.files
        ]
