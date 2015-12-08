import random
import sys
import numpy as np

from simplequestion import Query, Fact

class MemoryNetwork(object):
    """
    Note that this is a WIP and will be extended to handle other FB20 tasks.
    Currently, QA with 1 supporting fact is performed with perfect accuracy,
    and QA with 2 supporting facts is performed with moderate accuracy.
    Increasing performance on this latter task should suffice to achieve
    reasonable performance on many of the tasks in FB20, since the model 
    architecture generalized to k=2 is constant across the tasks 
    (i.e. the data is the only thing that changes once the model works for k=2)
    """
    def __init__(self, vocab_dim, embedding_dim, vectorizer, margin=0.2, k=1):
        self.vocab_dim = vocab_dim
        self.temp_dim = vocab_dim + 3
        self.vectorizer = vectorizer
        self.margin = margin
        self.k = k
        
        # Initialize O parameter matrices
        self.Om = np.random.random((embedding_dim, self.temp_dim))*0.2-0.1
        self.Oq = []

        for _ in range(k): 
            self.Oq.append(np.random.random((embedding_dim, vocab_dim))*0.2-0.1)

        # Initialize R parameter matrices
        self.Rm = np.random.random((embedding_dim, self.vocab_dim))*0.2-0.1
        self.Rq = []

        for _ in range(k+1): 
            self.Rq.append(np.random.random((embedding_dim, vocab_dim))*0.2-0.1)


    def phi_x(self, x):
        bow = self.vectorizer(x.sentence).flatten()
        return bow


    def phi_y(self, y):
        bow = self.vectorizer(y.sentence).flatten()
        bow = np.concatenate((bow, np.zeros(3)))
        return bow


    def phi_r(self, r):
        bow = self.vectorizer(r).flatten()
        return bow


    def phi_t(self, x, y1, y2):
        tx = x[-1].index if isinstance(x, list) and len(x) > 1 else None
        t1 = y1.index
        t2 = y2.index

        # Logic for setting time features
        bow = self.phi_y(y1) - self.phi_y(y2)
        bow[-1] = 1 if t1 < t2 else 0
       
        if not tx:
            bow[-2] = 1
            bow[-3] = 1
        else:
            bow[-2] = 1 if tx < t2 else 0 
            bow[-3] = 1 if tx < t1 else 0

        return bow


    def o_embed_x(self, x):
        # For handling variable numbers of items conditioning mem selection.
        if isinstance(x, list):
            xs = [np.dot(self.Oq[_], self.phi_x(x[_])) for _ in range(len(x))]
            embedding = np.sum(xs, axis=0) if len(xs) > 1 else xs.pop()
        else:
            embedding = np.dot(self.Oq[0], self.phi_x(x))
        return embedding


    def r_embed_x(self, x):
        # For handling variable numbers of items conditioning resp. selection.
        if isinstance(x, list):
            xs = [np.dot(self.Rq[_], self.phi_x(x[_])) for _ in range(len(x))]
            embedding = np.sum(xs, axis=0) if len(xs) > 1 else xs.pop()
        else:
            embedding = np.dot(self.Rq[0], self.phi_x(x))
        return embedding


    def o_embed_y(self, x, y1, y2):
        embedding = np.dot(self.Om, self.phi_t(x, y1, y2))
        return embedding


    def r_embed_y(self, y):
        embedding = np.dot(self.Rm, self.phi_r(y))
        return embedding


    def score_o(self, x, y1, y2):
        return np.dot(self.o_embed_x(x), self.o_embed_y(x, y1, y2))


    def score_r(self, x, y):
        return np.dot(self.r_embed_x(x), self.r_embed_y(y))


    def reset_memory(self):
        self.memory = []
   

    def reset_o_gradients(self):
        self.Om_grad = np.zeros_like(self.Om)
        self.Oq_grad = []
        
        for _ in range(self.k):
            self.Oq_grad.append(np.zeros_like(self.Oq[_]))


    def reset_r_gradients(self):
        self.Rm_grad = np.zeros_like(self.Rm)
        self.Rq_grad = []
        
        for _ in range(self.k+1):
            self.Rq_grad.append(np.zeros_like(self.Rq[_]))


    def update_o_gradients(self, query, n):
        mo = [m for m in self.memory if query.support[n] == m.index].pop()
        xs = [query]+[m for m in self.memory if m.index in query.support[:n]]
        xs = [xs[0]] if len(xs) < 2 else xs

        for m in self.memory:
            embed_x = self.o_embed_x(xs)

            if m != mo and self.margin + self.score_o(xs, m, mo) > 0:
                embed_y = self.o_embed_y(xs, m, mo)
                bow_y = self.phi_t(xs, m, mo)
                
                for _ in range(n+1):
                    bow_x = self.phi_x(xs[_])
                    self.Oq_grad[_] += np.outer(embed_y, bow_x)

                self.Om_grad += np.outer(embed_x, bow_y)       
            
            elif m != mo and self.margin - self.score_o(xs, mo, m) > 0:
                embed_y = self.o_embed_y(xs, mo, m)
                bow_y = self.phi_t(xs, mo, m)
            
                for _ in range(n+1):
                    bow_x = self.phi_x(xs[_])
                    self.Oq_grad[_] += -np.outer(embed_y, bow_x)

                self.Om_grad += -np.outer(embed_x, bow_y) 


    def update_r_gradients(self, query):
        ans = query.answer
        xs = [m for m in self.memory if m.index in query.support]
        xs = [query] + xs

        embed_xs = self.r_embed_x(xs)

        for word in self.vectorizer.vocab:
            embed_r = self.r_embed_y(word)
            bow_r = self.phi_r(word)

            # Note these are slightly hacky margins for computing loss 
            # Training w/ a simple margin didn't work well for some reason
            if word != ans and self.score_r(xs, word) > 0.1:
                for _ in range(len(xs)):
                    bow_x = self.phi_x(xs[_])
                    self.Rq_grad[_] += np.outer(embed_r, bow_x)

                self.Rm_grad += np.outer(embed_xs, bow_r)

            elif word == ans and self.score_r(xs, word) < 0.9:
                for _ in range(len(xs)):
                    bow_x = self.phi_x(xs[_])
                    self.Rq_grad[_] += -np.outer(embed_r, bow_x)

                self.Rm_grad += -np.outer(embed_xs, bow_r)


    def fit(self, stories, n_iter, rate=0.001):
        self.rate = rate
        for _ in range(n_iter):
            self.fit_story(random.choice(stories))


    def fit_story(self, story):
        self.reset_memory()
        self.reset_o_gradients()
        self.reset_r_gradients()

        # Compute O gradients
        for line in story:            
            if isinstance(line, Query):  
                for n in range(self.k):
                    if len(line.support)-1 >= n:
                        self.update_o_gradients(line, n)   

            else:
                self.memory.append(line)

        # Update O parameters
        self.Om += -self.rate * self.Om_grad        
        for _ in range(self.k):
            self.Oq[_] += -self.rate * self.Oq_grad[_]
 
        # Compute R gradients
        for line in story:            
            if isinstance(line, Query):  
                self.update_r_gradients(line)
        
        # Update R parameters
        self.Rm += -self.rate * self.Rm_grad        
        for _ in range(self.k):
            self.Rq[_] += -self.rate * self.Rq_grad[_]


    def select_support(self, xs):
        mems = []
        for _ in range(self.k):
            t = 0
            for i in range(len(self.memory)):
                if i < 1: 
                    continue
                score = self.score_o(xs, self.memory[i], self.memory[t])
                if score > 0:
                    t = i
            mems.append(self.memory[t])
            xs += mems

        return mems


    def select_response(self, xs):
        embed_xs = self.r_embed_x(xs)
        scores = np.dot(embed_xs.T, self.r_embeddings)
        top = np.argmax(scores)
        return self.vectorizer.vocab[top]


    def predict_answers(self, stories):
        correct = 0
        incorrect = 0

        # For easily computing argmax on response scores
        r_bags = np.zeros((0, self.vocab_dim))
        for word in self.vectorizer.vocab:
            r_bags = np.vstack([r_bags, self.phi_r(word)])
        self.r_embeddings = np.dot(self.Rm, r_bags.T)

        
        for story in stories:
            self.reset_memory()

            for line in story:
                if isinstance(line, Query):
                    mems = self.select_support([line])
                    mems = [line] + mems

                    response = self.select_response(mems)
                    if response == line.answer:
                        correct += 1
                    else:
                        incorrect += 1
                   
                else:
                    self.memory.append(line)
       
        print ''
        print 'EXAMPLE PREDICTION:'
        print '___________________'
        print 'Query:', line
        print 'Support:', mems
        print 'Predicted Answer:', response
        print ''

        return correct / float(incorrect+correct)


    def predict_support(self, stories):
        # Returns returns prediction accuracy on supporting mems.
        correct = 0
        incorrect = 0

        for story in stories:
            self.reset_memory()

            for line in story:
                if isinstance(line, Query):
                    mems = self.select_support([line])
                    mems = [m.index for m in mems]
                    if mems == line.support:
                        correct += 1
                    else:
                        incorrect += 1
                else:
                    self.memory.append(line)

        print ''
        print 'EXAMPLE PREDICTION:'
        print '___________________'
        print 'Query: ', line
        print 'Predicted Support: ', mems
        print ''

        return correct / float(incorrect+correct)




