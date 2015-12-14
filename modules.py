import numpy as np
import sys
import nltk
import string


class Module(object):
    stopwords = ['the','a','of','and','this','that','to']
    roles = ['A'+str(x) for x in range(20)]
    agents = []

    # Things common to all modules
    def __init__(self, rate=0.03):
        self.rate = rate

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def sigmoid(z):
        return 1.0/(1+np.exp(-z))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    @staticmethod
    def rectified(z, threshold=0):    
        indices = z < threshold
        z[indices] = 0
        return z

    @staticmethod
    def extract_agents(sentence):
        agents = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(sentence)):
            if isinstance(chunk, nltk.tree.Tree):
                agents.append(chunk.leaves()[0][0])
        return agents

    def clean(self, words):
        words = [w if w not in Module.agents else Module.roles[Module.agents.index(w)] 
                 for w in words]
        words = [w.lower() if w not in Module.roles else w for w in words]
        words = [w.translate(None, string.punctuation) for w in words]
        words = [w for w in words if w not in Module.stopwords]
        return words

class Response(Module):
    
    def __init__(self, net):
        super(Response, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                         len(net.vectorizer.vocab)))*0.2-0.1

    def build_choices(self, query):
        # Generates 4 choice embeddings using encode_choice 
        self.choices = np.zeros((0, self.net.embedding_dim))
        self.choices_bow = np.zeros((0, len(self.net.vectorizer.vocab)))

        for choice in query.choices:
            embedding = self.encode_choice(choice)
            self.choices = np.vstack([self.choices, embedding])

            bow = self.net.vectorizer(choice).flatten()
            self.choices_bow = np.vstack([self.choices_bow, bow])

    def encode_choice(self, choice):
        words = choice.split()
        words = self.clean(words)

        words = ' '.join(words)
        bow = self.net.vectorizer(choice).flatten()
        return np.dot(self.embedder, bow)

    def predict(self, output_features):
        scores = np.dot(self.choices, output_features)
        return self.softmax(scores)

    def update_parameters(self, gradient, activation):
        partial = np.dot(self.choices_bow.T, gradient)
        self.embedder += -self.rate * np.outer(activation, partial)


class Output(Module):

    def __init__(self, net):
        super(Output, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                 len(net.vectorizer.vocab)))*0.2-0.1

    # def build_map(self, story):
    #     self.map = np.zeros((0, self.net.embedding_dim))
    #     self.map_bow = np.zeros((0, len(self.net.vectorizer.vocab)))

    #     for sentence in story.text:
    #         agents = self.extract_agents(sentence)
    #         for agent in Module.agents:
    #             if agent not in Module.agents:
    #                 Module.agents.append(agent)

    #         string = ' '.join(sentence)
    #         string = string.lower()
    #         bow = self.net.vectorizer(string).flatten()
    #         embedding = np.dot(self.embedder, bow)
            
    #         self.map = np.vstack([self.map, embedding])
    #         self.map_bow = np.vstack([self.map_bow, bow])

    def update_parameters(self, gradient):
        partial = np.dot(self.map_bow.T, self.softmax_dist)
        self.embedder += -self.rate * np.outer(gradient, partial)


    def encode_output_features(self, q_embedding, train=True):
        self.softmax_dist = self.sigmoid(np.dot(self.memory, q_embedding))
        output_features = np.dot(self.memory.T, self.softmax_dist)

        return output_features

    def build_memory(self, story):
        # Converts story text into array of HRRs encoding sents
        Module.agents = []
        self.memory = np.zeros((0, self.net.embedding_dim))
        text = story.text

        for sentence in story.text:
            agents = self.extract_agents(sentence)
            for agent in agents:
                if agent not in Module.agents:
                    Module.agents.append(agent)

            vec = self.net.normalize(self.sentence_to_vec(sentence))
            self.memory = np.vstack([self.memory, vec])

    def sentence_to_vec(self, sentence):
        sentence = ['not' if w == "n\'t" else w for w in sentence]
        sentence = self.clean(sentence)
        sentence = [w for w in sentence if len(w) > 1]

        vec = np.zeros(self.net.embedding_dim)
        cache = []
        for word in sentence:
            word = word.translate(None, string.punctuation)
            if word in self.net.word_vecs.keys() and word not in cache:
                vec += self.net.word_vecs[word]
                cache.append(word)
        return vec


class Input(Module):

    def __init__(self, net):       
        super(Input, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                          len(net.vectorizer.vocab)))*0.2-0.1

    def encode_question(self, query):
        words = query.text.split()
        words = self.clean(words)

        words = ' '.join(words)
        bow = self.net.vectorizer(words).flatten()
        return np.dot(self.embedder, bow)

    def update_parameters(self, gradient, query):
        words = query.text.split()
        words = self.clean(words)

        words = ' '.join(words)
        bow = self.net.vectorizer(words).flatten()

        self.embedder += -self.rate * np.outer(gradient, bow)



