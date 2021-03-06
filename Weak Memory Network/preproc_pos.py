import cPickle as pickle
from joblib import Parallel, delayed
from mctest_load import load_stories
from modules import Module


def clean(stories):
    return [s for s in stories if len(s.queries) > 1]


def postag(sentence):
    return Module.extract_agents(sentence)


train_stories = clean(load_stories('MCTest/mc500.train.tsv','MCTest/mc500.train.ans'))
dev_stories = clean(load_stories('MCTest/mc500.dev.tsv','MCTest/mc500.dev.ans'))
test_stories = clean(load_stories('MCTest/mc500.test.tsv','MCTest/mc500.test.ans'))


all_stories = train_stories + test_stories + dev_stories

agents = Parallel(n_jobs=6)(
    delayed(postag)(sentence.split())
    for story in all_stories for sentence in story.text)

asdict = {k: v for k, v in zip(
    (tuple(sentence.split())
     for story in all_stories for sentence in story.text), agents)}

with open('pos_500.pkl', 'wb') as f:
    pickle.dump(asdict, f, 2)
