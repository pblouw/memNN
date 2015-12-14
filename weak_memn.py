import numpy as np
import modules
import gensim
import string
import cPickle as pickle

from sklearn.feature_extraction.text import CountVectorizer

stopwords = ['the','a','of','and','this','that','to']
roles = ['A'+str(x) for x in range(20)]

class BagOfWords(object):
    # Modified slightly from version used for standard memnet
    def __init__(self, word_list):
        self._vectorizer = CountVectorizer(binary=True)
        self._vectorizer.fit(word_list)

    def __call__(self, sentence):
        array = self._vectorizer.transform([sentence]).toarray().T
        return array

    @property
    def vocab(self):
        return self._vectorizer.vocabulary_.keys()

    @property
    def vocab_dim(self):
        return len(self.vocab)



class WeakMemoryNetwork(object):
    # Weakly supervised memory network
    def __init__(self, embedding_dim, stories):
        self.embedding_dim = embedding_dim
        self.vocab = None
        self.vectorizer = None

        self.build_vocab(stories)

        self._input = modules.Input(self)
        self._output = modules.Output(self)
        self._response = modules.Response(self)


    @staticmethod
    def normalize(z):    
        norm = np.linalg.norm(z)
        if norm > 0:
            return z / float(norm)
        else:
            return z

    def build_vocab(self, stories):
        word_list = set()

        for story in stories:
            # Add words from story text to word list
            for line in story.text:
                word_list.update([w.lower() for w in line])

            # Add words from each query text to word list
            for query in story.queries:
                for choice in query.choices:
                    word_list.update(choice.split())

        word_list = sorted(list(word_list))
        word_list = [w for w in word_list if not w.isdigit()]
        word_list = [w.lower() for w in word_list if w not in stopwords]
        word_list = [w.translate(None, string.punctuation) for w in word_list]
        word_list = [w for w in word_list if len(w) > 1]
        word_list = word_list + roles

        for _ in range(len(word_list)):
            if word_list[_][0] == 'a':
                idx = _
                break

        self.word_list = word_list[idx:]

        # Build BoW vectorizer from word list
        self.vectorizer = BagOfWords(word_list)

        # Build word2vec vocab
        with open('word2vec.pickle', 'rb') as pickle_file:
            self.word_vecs = pickle.load(pickle_file)

        # Assign random vectors to agent roles
        for role in roles:
            self.word_vecs[role] = self.new_vec(role)

        for word in self.word_list:
            if np.linalg.norm(self.word_vecs[word]) == 0:
                self.word_vecs[word] = self.new_vec(word)


    def new_vec(self, word, threshold=0.15):
        count = 0
        while count < 200:
            vec = np.random.randn(self.embedding_dim)
            vec = self.normalize(vec)

            for w2v in self.word_vecs .values():
                if np.dot(vec, w2v) > threshold:
                    count += 1
                    break
                else:
                    return vec
        print 'Could not make a good vector'
        return vec


    def train(self, story):
        # Currently uses 


        self._output.build_memory(story)

        for query in story.queries:
            self._response.build_choices(query)
            
            q_embed = self._input.encode_question(query)
            o_embed = self._output.encode_output_features(q_embed)

            prediction = self._response.predict(o_embed)
            
            target = np.array([1 if c == query.answer else 0 
                               for c in query.choices])
            
            # Error gradient on 4-way softmax output
            r_grad = prediction - target      

            # Update R based on this gradient
            # self._response.update_parameters(r_grad, q_embed+o_embed)  

            # Error gradient wrt input to R module
            r_input_grad = np.dot(self._response.choices.T, r_grad)    
            

            m_grad = np.dot(self._output.memory, r_input_grad) 
            q_grad = np.dot(self._output.memory.T, m_grad)

            # soft_in_grad = -np.outer(self._output.softmax_dist, self._output.softmax_dist)
            # diag = self._output.softmax_dist * (1 - self._output.softmax_dist)
            # indices = np.diag_indices(len(self._output.softmax_dist))
            # soft_in_grad[indices] += diag
            # m1_grad = self._output.softmax_dist - m2_grad


            # Gradient checks to avoid explosions
            if np.linalg.norm(q_grad) > 5:
                print 'Big grad'    
                q_grad = 5 * q_grad / np.linalg.norm(q_grad)

            self._input.update_parameters(q_grad, query)


    def predict_answer(self, story):
        correct = 0
        incorrect = 0 

        self._output.build_memory(story)

        for query in story.queries:          

            self._response.build_choices(query)
            q_embed = self._input.encode_question(query)
            o_embed = self._output.encode_output_features(q_embed)

            prediction = self._response.predict(q_embed+o_embed)
            prediction = query.choices[np.argmax(prediction)]

            if query.answer == prediction:
                correct += 1 
            else:
                incorrect += 1

        if len(story.queries) < 1:
            score = 0
        else:
            score = correct / float(correct+incorrect)
        return score














