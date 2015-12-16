import numpy as np
import modules
import os
import os.path
import string
import cPickle as pickle

from vocabulary import TextVocabulary
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
    def __init__(
        self, embedding_dim, timetag_dim, stories, word2vec=False, roles=False,
        timetags=False, coref=False, path='', pos_file='pos.pkl'):

        self.embedding_dim = embedding_dim
        self.timetag_dim = timetag_dim
        self.vocab = None
        self.vectorizer = None
        self.word2vec = word2vec
        self.timetags = timetags
        self.roles = roles
        self.coref = coref
        self.path = path

        if word2vec and embedding_dim != 300 and not timetags:
            raise Exception('Word2Vec embeddings are 300 dimensions')

        self.build_vocab(stories)

        with open(os.path.join(path, pos_file), 'rb') as f:
            pos_dict = pickle.load(f)

        self._input = modules.Input(self)
        self._output = modules.Output(self, pos_dict=pos_dict)
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

        # Build a word list
        for story in stories:
            # Add words from story text to word list
            for line in story.text:
                if self.coref:
                    word_list.update([w.lower() for w in line])
                else:
                     word_list.update(line.split())

            # Add words from each query text to word list
            for query in story.queries:
                for choice in query.choices:
                    word_list.update(choice.split())

        word_list = sorted(list(word_list))
        word_list = [w for w in word_list if not w.isdigit()]
        if self.coref:
            word_list = [w.lower() for w in word_list 
                         if w not in stopwords]
            word_list = [w.translate(None,string.punctuation)
                         for w in word_list]
        word_list = [w for w in word_list if len(w) > 1]

        if self.roles:
            word_list += roles

        for _ in range(len(word_list)):
            if word_list[_][0] == 'a':
                idx = _
                break

        self.word_list = word_list[idx:]

        # Build BoW vectorizer from word list
        self.vectorizer = BagOfWords(word_list)

        # Build word2vec vocab
        if self.word2vec:
            word2vec_path = os.path.join(self.path, 'word2vec.pickle')
            with open(word2vec_path, 'rb') as pickle_file:
                self.word_vecs = pickle.load(pickle_file)

            if self.roles:
                for role in roles:
                    self.word_vecs[role] = self.new_vec(role)

            # Build vectors for words not modelled by word2vec
            for word in self.word_list:
                if word not in self.word_vecs:
                    self.word_vecs[word] = self.new_vec(word)

                if np.linalg.norm(self.word_vecs[word]) == 0:
                    self.word_vecs[word] = self.new_vec(word)

            if self.timetags:
                self.vocab = TextVocabulary(self.embedding_dim,
                                            max_similarity=0.2)
                for word in self.word_list:
                    self.vocab[word]

        # Use random hrr vocabulary instead of word2vec
        else:
            self.vocab = TextVocabulary(self.embedding_dim, max_similarity=0.2)

            for word in self.word_list:
                self.vocab[word]


    def new_vec(self, word, threshold=0.15):
        count = 0
        while count < 500:
            vec = np.random.randn(300)
            vec = self.normalize(vec)
            for w2v in self.word_vecs.values():
                if np.dot(vec, w2v) > threshold:
                    count += 1
                    break
            else:
                return vec
        print 'Could not generate a good vector'       
        return vec


    def train(self, story):
        self._output.build_memory(story)

        for query in story.queries:
            
            self._response.build_choices(query)
            q_embed = self._input.encode_question(query)
            o_embed = self._output.encode_output_features(q_embed)

            prediction = self._response.predict(q_embed + o_embed)
            target = np.array([1 if c == query.answer else 0 
                               for c in query.choices])
            
            # Error gradient on 4-way softmax output
            r_grad = prediction - target   
            r_input_grad = np.dot(self._response.choices.T, r_grad)    
               
            def softmax_jacobian(o):
                a = modules.Module.softmax(o)
                return np.diag(a) - np.outer(a, a)

            J = softmax_jacobian(
                np.dot(self._response.choices, q_embed + o_embed))


            self._response.embedder += -self._response.rate * np.outer(
                q_embed + o_embed,
                np.sum(r_grad[:, None] * np.dot(
                    J, self._response.choices_bow), axis=0))

            # v^T W^T = q_embed.T
            a = np.dot(q_embed.T, self._output.memory.T)
            J1 = softmax_jacobian(a)
            b = np.dot(self._output.memory, self._response.choices.T)
            c = np.dot(np.dot(b.T, J1), self._output.memory)
            d = np.dot(self._response.choices, q_embed + o_embed)
            J2 = softmax_jacobian(d)
            e = np.sum(r_grad[:, None] * np.dot(J2, c), axis=0)
            self._input.update_parameters(
                e, query)


    def predict_answer(self, story):
        correct = 0
        incorrect = 0 

        self._output.build_memory(story)

        for query in story.queries:
            self._response.build_choices(query)
            q_embed = self._input.encode_question(query)
            o_embed = self._output.encode_output_features(q_embed)

            prediction = self._response.predict(q_embed + o_embed)
            prediction = query.choices[np.argmax(prediction)]

            if query.answer == prediction:
                correct += 1 
            else:
                incorrect += 1

        return correct / float(correct+incorrect)

  






