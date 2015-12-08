from memn import MemoryNetwork
from simplequestion import load
from memn_input import BagOfWords


train_stories = load('data/en/qa1_single-supporting-fact_train.txt')
test_stories = load('data/en/qa1_single-supporting-fact_test.txt')

vectorizer = BagOfWords(train_stories)

memnet = MemoryNetwork(vocab_dim=vectorizer.vocab_dim, embedding_dim=100,
                       vectorizer=vectorizer, k=1)


x = memnet.predict_support(test_stories)
print 'Support accuracy before learning: ', x

y = memnet.predict_answers(test_stories)
print 'Response accuracy before learning: ', y

memnet.fit(train_stories, 800)

x = memnet.predict_support(test_stories)
print 'Support accuracy after Learning: ', x

y = memnet.predict_answers(test_stories)
print 'Response accuracy after Learning: ', y