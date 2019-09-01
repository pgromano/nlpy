import torch
import torch.nn as nn
import numpy as np
import pathlib
import json
cimport numpy as np
from .vocabulary import Vocabulary
from .utils import download_url
from .token import TokenVector

cdef object S3_BUCKET = "https://s3-us-west-2.amazonaws.com/nerd.ai-datasets-nlp/word2vec/"

cdef object VOCAB_MAP = {
    "word2vec": S3_BUCKET + "vocab.json",
}

cdef object VECTOR_MAP = {
    "word2vec": S3_BUCKET + "vectors.npy",
}

cdef class Word2Vec:

    """ Word2Vec: 

    

    Arguments
    ---------
    vocab : array-like
        The vocabulary container that is used to map all tokens in the 
        vocabulary to a unique index, and reverse.
    vectors : array-like
        The matrix of shape (vocab_size, embedding_size) that contains the
        embedding vector for each token within the vocabulary.

    Examples
    --------

    ```python
    from nlpy import Word2Vec

    # Load Word2Vec
    w2v = Word2Vec.from_file("word2vec")

    # Get word-embedding
    X = w2v.transform('frog')

    # Find most similar
    w2v.most_similar("frog")
    >>> ['lizard']
    ```

    References
    ----------
    """

    cdef public object vocab
    cdef public object vectors
    cdef public object vector_norms
    cdef public int embedding_size
    cdef public int vocab_size

    def __init__(self, vocab=None, vectors=None):
        self.vocab = Vocabulary(vocab)
        self.vectors = np.squeeze(vectors).astype(np.float32)
        self.vector_norms = np.linalg.norm(vectors, axis=1)
        self.vocab_size = self.vectors.shape[0]
        self.embedding_size = self.vectors.shape[1]

    def add(self, token, vector=None, init='normal'):
        """ Add new token to vocabulary
        
        Arguments
        ---------
        token : str
            The token to add to vocabulary
        vector : array-like, optional
            The embedding vector to assign to the new token. If None, then 
            initialization is determined by the init parameters.
        init : str, optional, {'normal', 'zero'}
            Determines how to initialize new embedding vectors. If 'normal' then
            the vector is sampled from a normal distribution. If 'zero', then 
            all values are set to zero.
        """
        if token in self.vocab:
            return ValueError("Token \"{}\" already in vocabulary".format(token))
        
        if vector is None:
            if init == 'normal':
                vector = np.random.normal(size=self.embedding_size)
            elif init == 'zero':
                vector = np.zeros(size=self.embedding_size)
            else:
                raise ValueError("Unable to interpret init \"{}\"".format(init))
        self.vocab.add(token)
        self.vectors = np.vstack([self.vectors, vector])
        self.vector_norms = np.linalg.norm(self.vectors, axis=1)
        self.vocab_size = self.vectors.shape[0]
        return self

    def compute(self, formula):
        """ Compute from formula """
        
        formula = formula.split()
        operation = 'positive'

        pos = []
        neg = []
        for token in formula:
            if token == '+':
                operation = 'positive'
                continue
            elif token == '-':
                operation = 'negative'
                continue

            if operation == 'positive':
                pos.append(token)
            elif operation == 'negative':
                neg.append(token)

        vector = np.sum([self.transform(token) for token in pos], axis=0) - \
                np.sum([self.transform(token) for token in neg], axis=0)

        token = self.most_similar(vector, ignore=pos + neg)[0]
        return TokenVector(self, token, vector)

    def encode(self, token):
        """ Return vocabulary index for a given token in vocabulary """
        return self.vocab.encode(token)

    def decode(self, index):
        """ Return vocabulary token for a given index in vocabulary """
        return self.vocab.decode(index)

    def transform(self, token):
        """Return embedding vector for a given token in vocabulary """
        index = self.encode(token)
        if index is None:
            return np.zeros(self.embedding_size, dtype=float)
        return self.vectors[index]

    def similarity(self, a, b):
        """ Cosine Similarity Score
        
        Returns the cosine similarity score between two tokens, a and b, within
        the Word2Vec vocabulary. Cosine similarity scores are re-scaled to be from 
        0 to 1.

        Arguments
        ---------
        a, b : str or array-like
            If a or b are strings, then the corresponding embedding vector is
            used, otherwise they must be a embedding vector. 

        Returns
        -------
        score : float
            The cosine similarity score between embeddings a and b rescaled to
            be from 0 to 1. 
        """
        if isinstance(a, str):
            a = self.transform(a)

        if isinstance(b, str):
            b = self.transform(b)

        score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        score = (1 + score) / 2
        return score

    def most_similar(self, X, topk=1, ignore=None, return_score=False):
        """ Find most similar tokens

        Scan all tokens in the vocabulary and calculate the euclidean distance
        in the embedding space by using cosine-similarity. Cosine similarity
        scores are re-scaled to be from 0 to 1.

        Arguments
        ---------
        X : str or array-like
            If given a string, then will return embedding vector if token exists
            in vocabulary. If given a vector, then the vector is used directly.
        topk : int, optional
            The number of similar tokens to return.
        ignore : set, optional
            A list of tokens to ignore in the comparison. For example, in the
            standard "king - man + woman" example, generally all three tokens
            are ignored in the comparison (otherwise, king is the most similar).
            If ignore is None, then no tokens are ignored in the comparison.
        return_score : bool, optional
            Whether or not the similarity scores should be returned with the
            most similar tokens. 
        
        Returns
        -------
        output : list
            If return_score is False, then tokens are returned as a flat list. 
            Otherwise, each element in the output list is a tuple of 
            (token, score).
        """

        # Tokens to ignore
        if ignore is None:
            ignore = set()
        ignore = set(ignore)

        # Convert tokens to vectors
        if isinstance(X, str):
            ignore.add(X)
            X = self.transform(X)

        # Calcualte scores and rank-order
        scores = self.vectors.dot(X) / (self.vector_norms * np.linalg.norm(X))
        scores = (scores + 1) / 2
        order = scores.argsort()[::-1]
        scores = scores[order]

        output = []
        for i in range(scores.shape[0]):
            token = self.decode(order[i])
            if token not in ignore:
                if return_score:
                    token = (token, scores[i])
                output.append(token)
            if len(output) >= topk:
                break
        return output

    @classmethod
    def from_file(cls, path, vocab_path=None):
        """ Create a Word2Vec Method from file
        
        Assumes that the file follows the conventions used in the original 
        Stanford NLP release where each line in the file contains the token
        and then the vector. 

        Arguments
        ---------
        path : str
            The path to load the file into memory.
        """

        if path in VOCAB_MAP:

            # Check for nlpy data folder
            output_path = pathlib.Path('').home() / 'nlpy_data/word2vec'
            if not output_path.exists():
                output_path.mkdir()

            # Download vocab if not already present
            vocab_output = output_path / 'vocab.json'
            if not vocab_output.exists():
                download_url(VOCAB_MAP[path], str(vocab_output.absolute()))
            file = vocab_output.open()
            vocab = json.load(file)
            file.close()

            vectors_output = output_path / f'vectors.npy'
            if not vectors_output.exists():
                download_url(VECTOR_MAP[path], str(vectors_output.absolute()))
            vectors = np.load(str(vectors_output.absolute()))

        else:
            path = pathlib.Path(path)
            if path.exists():
                if path.suffix == '.txt':
                    vectors = np.loadtxt(path)
                elif path.suffix == '.npy':
                    vectors = np.load(path)
                elif path.suffix == '.bin':
                    raise NotImplementedError("The developer is too lazy to repeat gensim's implementation. Sorry!")
                else:
                    raise ValueError("Unable to interpet vectors")
            
            if vocab_path is not None:
                vocab = Vocabulary.from_file(vocab_path)
    
        return cls(vocab, vectors)

    def to_torch(self, path=None, requires_grad=True):
        """ Convert Word2Vec Embeddings to PyTorch Embeddings

        Arguments
        ---------
        path : str or path-like, optional
            If not None, then the state dictionary of the torch embedding is 
            saved to file.
        requires_grad : bool, optional
            Whether or not the embedding layer should require auto-gradients.
        """

        # Initialize pytorch embedding object
        torch_embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size
        )
        
        # Load state weights
        torch_embedding.load_state_dict(
            {'weight': torch.tensor(self.vectors)}
        )

        # Set gradients
        torch_embedding.requires_grad = requires_grad
        
        # Save a torch checkpoint file
        if path is not None:
            torch.save(torch_embedding.state_dict(), path)
            return None
        return torch_embedding

    def __call__(self, token):
        vector = self.transform(token)
        return TokenVector(self, token=token, vector=vector)

    def __getitem__(self, index):
        return self.vectors[index]

    def __repr__(self):
        return "Word2Vec({}, {})".format(
            self.vocab_size,
            self.embedding_size
        )

    def __str__(self):
        return self.__repr__()