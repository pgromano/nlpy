from .vocabulary import Vocabulary
from .tokenizer import Tokenizer


cdef class Encoder:

    cdef public object _tokenizer
    cdef public object _vocab
    cdef public int max_size
    cdef public object _encoder
    cdef public object _decoder

    def __init__(self, tokenizer=None, vocab=None, max_size=-1):
        self._reset(vocab)
        if tokenizer is None:
            tokenizer = Tokenizer()
        self._tokenizer = tokenizer
        self.max_size = max_size

    def fit(self, corpus, is_tokenized=False, ignore=set()):
        """
        """
        
        self._reset()
        return self.partial_fit(corpus, is_tokenized, ignore)
    
    def partial_fit(self, corpus, is_tokenized=False, ignore=None):
        """
        """

        if self.size >= self.max_size and self.max_size > 0:
            raise AttributeError("Cannot partial fit when max_size set.")

        # Initialize counter to truncate vocab to max_size (if set)
        cdef object counter = {}

        # Create ignore list for COUNTER ONLY
        if ignore is not None:
            ignore = set(ignore)

        # Build vocab
        cdef int index = self.size
        for document in corpus:

            # Tokenize if corpus is not tokenized
            if not is_tokenized:
                document = self._tokenizer(document)
            
            for token in document:
                if token not in ignore:
                    counter[token] = counter.get(token, 0) + 1

                if token not in self._encoder:
                    self._encoder[token] = index
                    self._decoder[index] = token
                    index += 1
        if self.max_size > 0:
            self._clip(counter, shift_index=len(self.special_tokens))
        return self

    def transform(self, corpus):
        enc = []
        for document in corpus:
            enc.append([
                self.encode(token)
                for token in self._tokenizer(document)
            ])
        return enc

    def inverse_transform(self, corpus):
        dec = []
        for document in corpus:
            dec.append([
                self.decode(index)
                for index in document
            ])
        return dec

    def encode(self, token):
        return self._encoder.get(token, self._vocab.unknown_index)

    def decode(self, index):
        return self._decoder.get(index, self._vocab.unknown_token)

    def get_vocab(self):
        return self._vocab

    def get_tokenizer(self):
        return self._tokenizer

    @property
    def size(self):
        return len(self._encoder)

    def _clip(self, counter, shift_index=0):

        sorted_encoder = {
            token: index + shift_index
            for index, token in enumerate(sorted(counter, key=lambda k: counter[k], reverse=True))
            if index < self.max_size
        }

        self._reset(sorted_encoder)

    def _reset(self, vocab=None):
        if not isinstance(vocab, Vocabulary):
            vocab = Vocabulary(vocab)
        self._vocab = vocab
        self._encoder = self._vocab._token_lookup
        self._decoder = {val: key for key, val in self._encoder.items()}

    def __call__(self, document):
        enc_doc = [
            self.encode(token)
            for token in self._tokenizer(document)
        ]

        if len(enc_doc) == 1:
            return enc_doc[0]
        return enc_doc

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.decode(index)
        elif hasattr(index, '__iter__') and not isinstance(index, str):
            return [self.decode(i) for i in index]
        raise ValueError("Unable to interpret index")

    def __len__(self):
        return self.size

    def __str__(self):
        return f"Encoder(size={self.size})"

    def __repr__(self):
        return self.__str__()