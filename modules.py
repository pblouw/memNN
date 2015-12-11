import numpy as np



class Module(object):
    # Things common to all modules
    def __init__(self, rate=0.03):
        self.rate = rate

    @staticmethod
    def sigmoid(z):
        return 1.0/(1+np.exp(-z))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)


class Response(Module):
    
    def __init__(self, net):
        super(Response, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                         len(net.vocab.keys)))*0.2-0.1

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
        # Change from linear embedding to make prediction more complex 
        bow = self.net.vectorizer(choice).flatten()
        return np.dot(self.embedder, bow)

    def predict(self, output_features):
        scores = np.dot(self.choices, output_features)
        return self.softmax(scores)

    def update_parameters(self, error, inp_activation):
        temp = np.dot(self.choices_bow.T, error)
        self.embedder += -self.rate * np.outer(inp_activation, temp)


class Output(Module):

    def __init__(self, net):
        super(Output, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                         len(net.vocab.keys)))*0.2-0.1

    def encode_output_features(self, q_embedding):
        weights = self.softmax(np.dot(self.memory, q_embedding))
        output_features = np.dot(self.memory.T, weights)
        
        return output_features

    def build_memory(self, story):
        # Converts story text into array of HRRs encoding sents
        self.memory = np.zeros((0, self.net.embedding_dim))
        text = story.text

        idx = 0
        for sentence in story.text:
            hrr = self.sentence_to_hrr(sentence, idx)
            self.memory = np.vstack([self.memory, hrr])
            idx += 1

    def sentence_to_hrr(self, sentence, idx):
        # Modify this to include parsing, time features
        words = sentence.split()

        hrr = np.zeros(self.net.embedding_dim)
        for word in words:
            if word in self.net.word_vecs:
                hrr += self.net.word_vecs[word]
        return hrr


class Input(Module):

    def __init__(self, net):       
        super(Input, self).__init__()
        self.net = net
        self.embedder = np.random.random((net.embedding_dim, 
                                          len(net.vocab.keys)))*0.2-0.1


    def encode_question(self, query):
        embed = np.zeros(self.net.embedding_dim)

        for word in query.text:
            if word in self.net.word_vecs:
                embed += self.net.word_vecs[word]
        return embed
        # bow = self.net.vectorizer(query.text).flatten()
        # return np.dot(self.embedder, bow)

    def update_parameters(self, error, bow):
        self.embedder += -self.rate * np.outer(error, bow)



