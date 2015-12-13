import numpy as np
import modules
import gensim

from vocabulary import TextVocabulary
from sklearn.feature_extraction.text import CountVectorizer

stopwords = ['the','a','of','and','this','that','to']

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


    def build_vocab(self, stories):
        word_list = set()

        for story in stories:
            # Add words from story text to word list
            for line in story.text:
                word_list.update(line.split())

            # Add words from each query text to word list
            for query in story.queries:
                for choice in query.choices:
                    word_list.update(choice.split())

        word_list = sorted(list(word_list))
        word_list = [w for w in word_list if not w.isdigit()]
        word_list = [w for w in word_list if len(w) > 1]

        # Build BoW vectorizer from word list
        self.vectorizer = BagOfWords(word_list)

        # Build SPA vocab from word list for HRR use
        self.vocab = TextVocabulary(self.embedding_dim, max_similarity=0.2)

        for word in word_list:
            self.vocab[word]

        # Build word2vec vocab
        model = gensim.models.word2vec.Word2Vec.load_word2vec_format(
                'GoogleNews-vectors-negative300.bin', binary=True)

        self.word_vecs = dict()

        for word in word_list:
            if word not in stopwords:
                if word in model.vocab:
                    self.word_vecs[word] = model[word]
                else:
                    self.word_vecs[word] = self.vocab[word].v

    def train(self, story):
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
            
            soft_out_grad = np.dot(self._output.memory, r_input_grad)
            soft_in_grad = -np.outer(self._output.softmax_dist, self._output.softmax_dist)
            
            diag = self._output.softmax_dist * (1 - self._output.softmax_dist)
            np.fill_diagonal(soft_in_grad, diag)
            soft_in_grad = np.sum(soft_in_grad, axis=1)

            soft_grad =  soft_out_grad # * soft_in_grad

            q_grad = np.dot(self._output.memory.T, soft_grad)

            print 'Norm: ', np.linalg.norm(q_grad)

            self._input.update_parameters(q_grad, self.vectorizer(query.text).flatten())


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














