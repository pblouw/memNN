from mctest_load import load_stories
from weak_memn import WeakMemoryNetwork


train_stories = load_stories('MCTest/mc160.train.tsv', 'MCTest/mc160.train.ans')
test_stories = load_stories('MCTest/mc160.test.tsv', 'MCTest/mc160.test.ans')
all_stories = train_stories + test_stories

def compute_accuracy(stories, model):
    accuracy = 0
    for story in stories: 
        score = model.predict_answer(story)
        accuracy += score
    return float(accuracy) / float(len(stories))

# initialize with all stories to get full vocab
model = WeakMemoryNetwork(256, all_stories)


print 'Accuracy prior to training: ', compute_accuracy(train_stories, model)


# Train for a certain number of epochs
for i in range(5):
    for story in train_stories:
        model.train(story)


print 'Accuracy after training: ', compute_accuracy(train_stories, model)


# Test generalization on test set
print 'Generalization accuracy: ', compute_accuracy(test_stories, model)

