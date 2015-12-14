import sys
import cPickle as pickle
from mctest_load import load_stories
from weak_memn import WeakMemoryNetwork

"""
Runs a weakly supervised memory network on MCTest
Currently uses MCTest 160 dev + MCTEST 160 train
as training data.
"""

def compute_accuracy(stories, model):
    accuracy = 0
    for story in stories: 
        score = model.predict_answer(story)
        accuracy += score
    return float(accuracy) / float(len(stories))

with open('MCTest/mc160.dev.coref','rb') as f:
    dev_stories = pickle.load(f)

with open('MCTest/mc160.train.coref','rb') as f:
    train_stories = pickle.load(f)

with open('MCTest/mc160.test.coref','rb') as f:
    test_stories = pickle.load(f)

all_stories = train_stories + test_stories + dev_stories


# initialize with all stories to get full vocab
model = WeakMemoryNetwork(300, all_stories)

print 'Training Accuracy prior to training: ', compute_accuracy(train_stories, model)
print 'Testing Accuracy prior to training: ', compute_accuracy(test_stories, model)


# Train for a certain number of epochs
count = 0
for i in range(30):
    for story in train_stories + dev_stories:
        model.train(story)
    print 'Iteration ', count, ' complete!'
    count += 1


# Test for good generalization 
print 'Training accuracy after training: ', compute_accuracy(train_stories, model)
print 'Testing accuracy after training: ', compute_accuracy(test_stories, model)

