import numpy as np

class TokenVector:
    def __init__(self, model, token=None, vector=None):
        if vector is None:
            if token is None:
                raise ValueError("Must supply either a token or vector")
            else:
                index = model.encode(token)
                vector = model.vectors[index]
                vector_norm = model.vector_norms[index]
        else:
            if token is None:
                index = None
                token = '[UNK]'
            else:
                index = model.encode(token)
            vector_norm = np.linalg.norm(vector)

        self._model = model
        self.token = token
        self.index = index
        self.vector = vector
        self.vector_norm = vector_norm

    def similarity(self, X):
        if isinstance(X, TokenVector):
            X = X.vector
        return self._model.similarity(self.vector, X)

    def most_similar(self, topk=1, ignore=None, return_score=False):
        return self._model.most_similar(self.vector, topk, ignore, return_score)

    def __add__(self, X):
        if isinstance(X, str):
            X = self._model.transform(X)
        elif isinstance(X, TokenVector):
            X = X.vector
        return TokenVector(self._model, vector=self.vector + X)

    def __sub__(self, X):
        if isinstance(X, str):
            X = self._model.transform(X)
        elif isinstance(X, TokenVector):
            X = X.vector
        return TokenVector(self._model, vector=self.vector - X)

    def __mul__(self, X):
        return self.similarity(X)

    def __repr__(self):
        return self.token

    def __str__(self):
        return self.__repr__()
            