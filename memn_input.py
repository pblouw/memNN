from sklearn.feature_extraction.text import CountVectorizer


class BagOfWords(object):
    def __init__(self, stories):
        self._vectorizer = CountVectorizer(binary=True)
        self._vectorizer.fit([s.sentence for story in stories for s in story])

    def __call__(self, sentence):
        return self._vectorizer.transform([sentence])
