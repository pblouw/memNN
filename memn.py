import numpy as np

from simplequestion import Query


class MemoryNetwork(object):
    def __init__(self, vocab_dim, embedding_dim, vectorizer):
        self.vectorizer = vectorizer
        self.U = np.random.random((embedding_dim, vocab_dim)) * 0.2 - 0.1
        self.facts = []

    def fit(self, stories, n_iter):
        for _ in range(n_iter):
            self.fit_story(np.random.choice(stories))

    def fit_story(self, story):
        self.facts = []
        for sentence in story:
            if isinstance(sentence, Query):
                embed_q = np.dot(self.vectorizer(sentence.sentence), self.U.T)
                # TODO
            else:
                self.facts.append(sentence.sentence)

