import cPickle as pickle
import gensim
import numpy as np
import string

"""
This script pickles a dictionary that maps words to
word2vec vectors by extracting a vocabulary from the data
generated by the coreference resolover.
"""


with open('MCTest/mc500.train.coref','rb') as f:
    big_train_stories = pickle.load(f)

with open('MCTest/mc500.test.coref','rb') as f:
    big_test_stories = pickle.load(f)

with open('MCTest/mc500.dev.coref','rb') as f:
    big_dev_stories = pickle.load(f)

with open('MCTest/mc160.test.coref','rb') as f:
    small_test_stories = pickle.load(f)


with open('MCTest/mc160.train.coref','rb') as f:
    small_train_stories = pickle.load(f)

with open('MCTest/mc160.dev.coref','rb') as f:
    small_dev_stories = pickle.load(f)

all_stories = big_train_stories + big_test_stories + \
              big_dev_stories + small_dev_stories +  \
              small_train_stories + small_test_stories


word_list = set()
for story in all_stories:

    # Add words from story text to word list
    for line in story.text:
        word_list.update([w.lower() for w in line])

    # Add words from each query text to word list
    for query in story.queries:
        for choice in query.choices:
            word_list.update(choice.split())

if 'not' not in word_list:
    word_list.append('not')

word_list = sorted(list(word_list))
word_list = [w for w in word_list if not w.isdigit()]
word_list = [w.lower() for w in word_list]
word_list = [w.translate(None, string.punctuation) for w in word_list]
word_list = [w for w in word_list if len(w) > 1]

for _ in range(len(word_list)):
    if word_list[_][0] == 'a':
        idx = _
        break

word_list = word_list[idx:]

model = gensim.models.word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

word_vecs = dict()
for word in word_list:   
    if word in model.vocab:
        word_vecs[word] = model[word]
    else:
        word_vecs[word] = np.zeros(300)


with open('word2vec.pickle', 'wb') as pickle_file:
	pickle.dump(word_vecs, pickle_file)