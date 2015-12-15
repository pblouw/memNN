import cPickle as pickle
import os.path
import platform
import sys
path = '/home/jgosmann/Documents/projects/stat946/memNN/Weak Memory Network'
if not os.path.exists(path):
    path = os.pardir
sys.path.insert(0, path)

from psyrun import Param, map_pspace_parallel
from psyrun.scheduler import Sqsub

from mctest_load import load_stories
from weak_memn import WeakMemoryNetwork


variants = Param(
    timetags=[False, True, False, False, False, True],
    word2vec=[False, False, True, False, False, True],
    roles=[False, False, False, True, False, True],
    coref=[False, False, False, False, True, True])

if platform.node().startswith('ctn'):
    pspace = Param(n_epochs=[5]) * variants * Param(trial=range(1))
    mapper = map_pspace_parallel
else:  # assume sharcnet
    pspace = Param(n_epochs=[10]) * variants * Param(trial=range(30))
    workdir = '/work/jgosmann/stat946'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '24h',
        'memory': '1536M'
    }
    n_splits = 90
    min_items = 2


def clean(stories):
    return [s for s in stories if len(s.queries) > 1]


def compute_accuracy(stories, model):
    accuracy = 0
    for story in stories:
        score = model.predict_answer(story)
        accuracy += score
    return float(accuracy) / float(len(stories))


def load_data(coref):
    data_path = os.path.join(path, 'MCTest')
    if coref:
        with open(os.path.join(data_path, 'mc160.train.coref'), 'rb') as f:
            train_stories = clean(pickle.load(f))
        with open(os.path.join(data_path, 'mc160.dev.coref'), 'rb') as f:
            dev_stories = clean(pickle.load(f))
        with open(os.path.join(data_path, 'mc160.test.coref'), 'rb') as f:
            test_stories = clean(pickle.load(f))
    else:
        train_stories = clean(load_stories(
            os.path.join(data_path, 'mc160.train.tsv'),
            os.path.join(data_path, 'mc160.train.ans')))
        dev_stories = clean(load_stories(
            os.path.join(data_path, 'mc160.dev.tsv'),
            os.path.join(data_path, 'mc160.dev.ans')))
        test_stories = clean(load_stories(
            os.path.join(data_path, 'mc160.test.tsv'),
            os.path.join(data_path, 'mc160.test.ans')))
    return train_stories, dev_stories, test_stories


def execute(trial, n_epochs, timetags, word2vec, roles, coref):
    train_stories, dev_stories, test_stories = load_data(coref)
    all_stories = train_stories + test_stories + dev_stories

    if timetags:
        d = 556
    else:
        d = 300
    model = WeakMemoryNetwork(
        d, all_stories, timetags=timetags, word2vec=word2vec, roles=roles,
        coref=coref)

    pre_train_acc = compute_accuracy(train_stories, model)
    pre_test_acc = compute_accuracy(test_stories, model)

    for _ in range(n_epochs):
        for story in train_stories + dev_stories:
            model.train(story)

    post_train_acc = compute_accuracy(train_stories, model)
    post_test_acc = compute_accuracy(test_stories, model)

    return {
        'pre_train_acc': pre_train_acc,
        'pre_test_acc': pre_test_acc,
        'post_train_acc': post_train_acc,
        'post_test_acc': post_test_acc,
    }
