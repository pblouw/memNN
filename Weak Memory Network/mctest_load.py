
"""
NOTE: This works, but it's a mess, so I'm planning on
cleaning it up and refactoring it shortly.
"""
import nltk
import string
from collections import namedtuple
from subprocess import call
import sys

tokenizer = nltk.load('tokenizers/punkt/english.pickle')

Story = namedtuple('Story', ['text', 'queries'])
Query = namedtuple('Query', ['text', 'choices', 'answer'])


def load_answers(filename):
    with open(filename, 'r') as f:
        answers = f.read().split('\n')[:-1]
        answers = [a.split('\t') for a in answers]
        answers = [[c.strip() for c in a] for a in answers]
        answers = {i:j for i, j in enumerate(answers)}
    return answers


def load_stories(filename, answerfile):
    stories = []

    with open(filename, 'r') as f:
        data = f.read()
        raw_stories = data.split('\n')[:-1]
        answers = load_answers(answerfile)

        for ind, raw in enumerate(raw_stories):
            ans = answers[ind]
            story = parse(raw, ans)
            stories.append(story)

        return stories


def parse(raw_story, answers, coref=False):
    items = raw_story.split('\t')[2:]

    if not coref:
            text = tokenizer.tokenize(items.pop(0))
            text = [t.replace('newline', ' ') for t in text]
            text = [preprocess(t) for t in text]

    else:
        text = items.pop(0)
        text = text.replace('\\newline', ' ')

    queries = parse_queries(items, answers, coref)

    return Story(text, queries)


def parse_queries(items, answers, coref):
    query_list = []

    counter = 0
    for index, item in enumerate(items):
        if '?' in item and '?\"' not in item:
            if not coref:
                text = preprocess(strip_tag(item))
                choices = items[index+1:index+5]
                choices = [preprocess(c) for c in choices]

            else:     
                text = strip_tag(item)
                text = text.strip()
                choices = items[index+1:index+5]
                choices = [c.capitalize() for c in choices]
                choices = [c.replace('\r', ' ').strip() for c in choices]

            ans = choices[text_to_ind(answers[counter])]
            query_list.append(Query(text, choices, ans))
            counter += 1

    return query_list


def text_to_ind(answer):
    translation = dict([('A', 0), ('B', 1), ('C', 2), ('D', 3)])
    return translation[answer]

def strip_tag(item):
    index = item.index(':') + 1
    return item[index:]

def preprocess(text):
    text = text.strip()
    text = text.translate(None, string.punctuation)
    return text


if __name__ == '__main__':
    stories = load_stories('MCTest/mc160.train.tsv')
    print stories
