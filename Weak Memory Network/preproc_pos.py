import cPickle as pickle
from joblib import Parallel, delayed
from modules import Module


def clean(stories):
    return [s for s in stories if len(s.queries) > 1]


def postag(sentence):
    return Module.extract_agents(sentence)


with open('MCTest/mc160.dev.coref', 'rb') as f:
    dev_stories = clean(pickle.load(f))

with open('MCTest/mc160.train.coref', 'rb') as f:
    train_stories = clean(pickle.load(f))

with open('MCTest/mc160.test.coref', 'rb') as f:
    test_stories = clean(pickle.load(f))


all_stories = train_stories + test_stories + dev_stories


agents = Parallel(n_jobs=4)(
    delayed(postag)(sentence)
    for story in all_stories for sentence in story.text)

asdict = {k: v for k, v in zip(
    (sentence for story in all_stories for sentence in story.text), agents)}

with open('coref_pos.pkl', 'wb') as f:
    pickle.dump(asdict, f, 2)
