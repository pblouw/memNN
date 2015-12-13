import numpy as np
import modules

from vocabulary import TextVocabulary
from sklearn.feature_extraction.text import CountVectorizer



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
        self.vocab = TextVocabulary(self.embedding_dim // 2, max_similarity=0.2)

        for word in word_list:
            self.vocab[word]

    def train(self, story):
        self._output.build_memory(story)

        for query in story.queries:
            self._response.build_choices(query)
            q_embed = self._input.encode_question(query)
            o_embed = self._output.encode_output_features(q_embed)

            prediction = self._response.predict(q_embed+o_embed)
            target = np.array([1 if c == query.answer else 0 
                               for c in query.choices])
            
            # Error gradient on 4-way softmax output
            r_grad = prediction - target      

            def softmax_jacobian(o):
                a = modules.Module.softmax(o)
                return np.diag(a) - np.outer(a, a)

            J = softmax_jacobian(
                np.dot(self._response.choices, q_embed + o_embed))

            # Update R based on this gradient
            #self._response.update_parameters(r_grad, q_embed+o_embed)  

            self._response.embedder += -self._response.rate * np.outer(
                q_embed + o_embed,
                np.sum(r_grad[:, None] * np.dot(
                    J, self._response.choices_bow), axis=0))

            # Error gradient wrt input to R module


            # v^T W^T = q_embed.T
            a = np.dot(q_embed.T, self._output.memory.T)
            J1 = softmax_jacobian(a)
            b = np.dot(self._output.memory, self._response.choices.T)
            c = np.dot(np.dot(b.T, J1), self._output.memory)
            d = np.dot(self._response.choices, q_embed + o_embed)
            J2 = softmax_jacobian(d)
            e = np.sum(r_grad[:, None] * np.dot(J2, c), axis=0)
            self._input.update_parameters(
                e, self.vectorizer(query.text).flatten())


            #self._response.build_choices(query)
            #q_embed = self._input.encode_question(query)
            #o_embed = self._output.encode_output_features(q_embed)

            #prediction2 = self._response.predict(q_embed+o_embed)
            #r_grad2 = prediction2 - target      
            #print target, prediction, prediction2, np.linalg.norm(r_grad2) - np.linalg.norm(r_grad)
            #assert np.linalg.norm(r_grad2) - np.linalg.norm(r_grad) <= 0

            #r_input_grad = np.dot(self._response.choices.T, r_grad)    
            #self._input.update_parameters(
                    #r_input_grad, self.vectorizer(query.text).flatten())

            # TODO: Figure out update to Input module params based on gradient
            # Right now the R module parameters are the only thing being trained

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














