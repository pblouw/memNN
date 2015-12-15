from mctest_load import load_stories
from weak_memn import WeakMemoryNetwork

"""
Runs a baseline test on a weakly supervised memory network.
"""
def clean(stories):
    return [s for s in stories if len(s.queries) > 1]

def compute_accuracy(stories, model):
    accuracy = 0
    for story in stories: 
        score = model.predict_answer(story)
        accuracy += score
    return float(accuracy) / float(len(stories))


train_stories = clean(load_stories('MCTest/mc160.train.tsv','MCTest/mc160.train.ans'))
dev_stories = clean(load_stories('MCTest/mc160.dev.tsv','MCTest/mc160.dev.ans'))
test_stories = clean(load_stories('MCTest/mc160.test.tsv','MCTest/mc160.test.ans'))

all_stories = train_stories + test_stories + dev_stories

# initialize with all stories to get full vocab
model = WeakMemoryNetwork(256, all_stories, timetags=False, word2vec=False)

print 'Training Accuracy prior to training: ', compute_accuracy(train_stories, model)
print 'Testing Accuracy prior to training: ', compute_accuracy(test_stories, model)

# Train for a certain number of epochs
count = 0
for i in range(5):
    for story in train_stories + dev_stories:
        model.train(story)
    print 'Iteration ', count, ' complete!'
    count += 1

# Test for good generalization 
print 'Training accuracy after training: ', compute_accuracy(train_stories, model)
print 'Testing accuracy after training: ', compute_accuracy(test_stories, model)

