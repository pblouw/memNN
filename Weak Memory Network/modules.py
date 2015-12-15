import numpy as np
import sys
import nltk
import string
from nengo import spa

class Module(object):
    roles = ['A'+str(x) for x in range(20)]
    agents = []
    stopwords = ['the','a','of','and','this','that','to']
 
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
    def normalize(z):    
        norm = np.linalg.norm(z)
        if norm > 0:
            return z / float(norm)
        else:
            return z

    @staticmethod
    def extract_agents(sentence):
        agents = []
        for chunk in nltk.ne_chunk(nltk.pos_tag(sentence)):
            if isinstance(chunk, nltk.tree.Tree):
                agents.append(chunk.leaves()[0][0])
        return agents

    @staticmethod
    def clean(words):
        words = [w if w not in Module.agents else 
                 Module.roles[Module.agents.index(w)] for w in words]
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

        # if self.net.word2vec:
        #     vec = np.zeros(300)
        #     for word in words:
        #         if word in self.net.word_vecs.keys():
        #             vec += self.net.word_vecs[word]
        #     return vec

        words = ' '.join(words)
        bow = self.net.vectorizer(choice).flatten()
        return np.dot(self.embedder, bow)

    def predict(self, output_features):
        scores = np.dot(self.choices, output_features)
        return self.softmax(scores)

    # def update_parameters(self, gradient, activation):
    #     partial = np.dot(self.choices_bow.T, gradient)
    #     self.embedder += -self.rate * np.outer(activation, partial)


class Output(Module):

    def __init__(self, net):
        super(Output, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                 len(net.vectorizer.vocab)))*0.2-0.1

    # def update_parameters(self, gradient):
    #     partial = np.dot(self.map_bow.T, self.hidden_dist)
    #     self.embedder += -self.rate * np.outer(gradient, partial)


    def encode_output_features(self, q_embedding):
        self.hidden_dist = self.softmax(np.dot(self.memory, q_embedding))
        output_features = np.dot(self.memory.T, self.hidden_dist)
        
        return output_features


    def build_memory(self, story):
        # Converts story text into array of HRRs encoding sents
        Module.agents = []
        text = story.text

        def build_w2v_memory(story):
            memory = np.zeros((0, 300))
            for sentence in story.text:
                if self.net.roles:
                    agents = self.extract_agents(sentence)
                    for agent in agents:
                        if agent not in Module.agents:
                            Module.agents.append(agent)
                
                vec = self.sentence_to_vec(sentence)
                memory = np.vstack([memory, vec])
            return memory
       
        def build_hrr_memory(story):
            memory = np.zeros((0, self.net.embedding_dim))
            
            if self.net.timetags:
                memory = np.zeros((0, self.net.embedding_dim // 2))

            for sentence in story.text:
                hrr = self.sentence_to_hrr(sentence)
                memory = np.vstack([memory, hrr])
            return memory

        
        if self.net.word2vec:
            self.memory = build_w2v_memory(story)
        else:
            self.memory = build_hrr_memory(story)    

        # If timetags, extend memory to include time vecs
        if self.net.timetags:
            if self.net.word2vec:
                self.memory = np.hstack([self.memory, self.build_timetags(
                    self.memory.shape[0], self.net.embedding_dim - 300)[::-1]])
            else:
                self.memory = np.hstack([self.memory, self.build_timetags(
                    self.memory.shape[0], self.net.embedding_dim // 2)[::-1]])


    def sentence_to_vec(self, sentence):
        if self.net.coref:
            sentence = ['not' if w == "n\'t" else w for w in sentence]
            sentence = self.clean(sentence)
        else:
            sentence = sentence.split()
        
        sentence = [w for w in sentence if len(w) > 1]

        vec = np.zeros(300)
        for word in sentence:
            word = word.translate(None, string.punctuation)
            if word in self.net.word_vecs.keys():
                vec += self.net.word_vecs[word]
        return vec


    def sentence_to_hrr(self, sentence):
        # Modify this to include parsing, time features
        words = sentence.split()
        words = [w for w in words if w not in self.stopwords]

        hrr = np.zeros(self.net.embedding_dim)

        if self.net.timetags:
            hrr = np.zeros(self.net.embedding_dim // 2)

        for word in words:
            if word in self.net.vocab.keys:
                hrr += self.net.vocab[word].v

        return hrr


    def build_timetags(self, n, d):
        base = spa.SemanticPointer(d)
        base.make_unitary()
        base = base.v

        step_size = 0.25

        increment = np.zeros(d)
        increment[1] = 1.
        increment = np.fft.irfft(np.fft.rfft(increment) ** step_size)

        base_pos_v = np.empty((n, d))
        for i in range(n):
            if i <= 0:
                a = np.fft.rfft(base)
            else:
                a = np.fft.rfft(base_pos_v[i - 1])
            b = np.fft.rfft(increment)
            a *= b
            base_pos_v[i, :] = np.fft.irfft(a)

        return base_pos_v


class Input(Module):

    def __init__(self, net):       
        super(Input, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                          len(net.vectorizer.vocab)))*0.2-0.1

    def encode_question(self, query):
        words = query.text.split()
        words = self.clean(words)

        # vec = np.zeros(300)
        # for word in words:
        #     if word in self.net.word_vecs.keys():
        #         vec += self.net.word_vecs[word]
        # return vec

        words = ' '.join(words)
        bow = self.net.vectorizer(words).flatten()
        return np.dot(self.embedder, bow) 

    def update_parameters(self, gradient, query):
        words = query.text.split()
        words = self.clean(words)

        words = ' '.join(words)
        bow = self.net.vectorizer(words).flatten()

        self.embedder += -self.rate*100.0 * np.outer(gradient, bow)


