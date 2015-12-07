from memn import MemoryNetwork
from simplequestion import load
from memn_input import BagOfWords


train_stories = load('data/en/qa1_single-supporting-fact_train.txt')
test_stories = load('data/en/qa1_single-supporting-fact_test.txt')

vectorizer = BagOfWords(train_stories)

memnet = MemoryNetwork(vocab_dim=vectorizer.vocab_dim, embedding_dim=50,
                       vectorizer=vectorizer)



print 'Accuracy before training for selecting best supporting memory on test set:' 
print memnet.predict(test_stories)

memnet.fit(train_stories, 1000)

print 'Accuracy after training for selecting best supporting memory on test set:'
print memnet.predict(test_stories)