import sys
import random
import numpy as np

from simplequestion import Query


class MemoryNetwork(object):
    # Note that this is a WIP and will be fixed to generalize across all FB20 tasks
    def __init__(self, vocab_dim, embedding_dim, vectorizer, k=2):
        self.vectorizer = vectorizer
        self.Uf = np.random.random((embedding_dim, vocab_dim+3)) * 0.2 - 0.1 # 3 is for time features
        self.Uq = np.random.random((embedding_dim, vocab_dim)) * 0.2 - 0.1
        
        self.margin = 0.2
        self.vocab_dim = vocab_dim

    def fit(self, stories, n_iter):
        for _ in range(n_iter):
            self.fit_story(random.choice(stories))

    def fit_story(self, story):
        self.facts = []
        self.mvecs = np.zeros((0, self.vocab_dim+3))

        Uf_grad = np.zeros_like(self.Uf)
        Uq_grad = np.zeros_like(self.Uq)

        for sentence in story:
            bow = self.vectorizer(sentence.sentence).flatten()

            if isinstance(sentence, Query):
                embed_q = np.dot(self.Uq, bow)

                # Compute gradients iteratively
                for x in range(len(self.facts)):
                    if self.facts[x].index == sentence.support:
                        for y in range(len(self.facts)):
                            if y != x and self.margin + self.score(sentence, y, x) > 0:
                                embed_f = np.dot(self.Uf, self.get_phi(y, x))
                                Uq_grad += np.outer(embed_f, bow)
                                Uf_grad += np.outer(embed_q, self.get_phi(y, x))                                
                    
                    else:
                        indices = [f.index for f in self.facts]
                        idx = indices.index(sentence.support)

                        if self.margin - self.score(sentence, idx, x) > 0:
                            embed_f = np.dot(self.Uf, self.get_phi(idx, x))
                            Uq_grad += -np.outer(embed_f, bow)
                            Uf_grad += -np.outer(embed_q, self.get_phi(idx, x))

            else:
                bow = np.concatenate((bow, np.zeros(3))) # for time features
                self.facts.append(sentence)
                self.mvecs = np.vstack([self.mvecs, bow])

        self.Uf += -0.001 * Uf_grad # somewhat arbitrary learning rate
        self.Uq += -0.001 * Uq_grad   


    def score(self, query, f1, f2):
        embed_f = np.dot(self.Uf, self.get_phi(f1, f2))
        embed_q = np.dot(self.Uq, self.vectorizer(query.sentence).flatten())
        return np.dot(embed_f, embed_q)

    def predict(self, stories):
        # Currently returns prediction accuracy on supporting mems.
        correct = 0
        incorrect = 0

        for story in stories:
            self.facts = []
            self.mvecs = np.zeros((0, self.vocab_dim+3))

            for sentence in story:
                bow = self.vectorizer(sentence.sentence).flatten()

                if isinstance(sentence, Query):
                    t = 0
                    for i in range(len(self.facts)):
                        if i < 1: 
                            continue
                        score = self.score(sentence, i, t)
                        if score > 0:
                            t = i

                    if self.facts[t].index == sentence.support:
                        correct += 1
                    else:
                        incorrect += 1
                else:
                    bow = np.concatenate((bow, np.zeros(3))) # for time features
                    self.facts.append(sentence)
                    self.mvecs = np.vstack([self.mvecs, bow])
                    
        return correct / float(incorrect+correct)


    def get_phi(self, f1, f2):
        phi = self.mvecs[f1,:]-self.mvecs[f2,:]
        if f1 < f2:
            phi[-1] = 1
        phi[-2], phi[-3] = 1, 1
        return phi






