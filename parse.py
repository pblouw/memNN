import sys
import nltk
from collections import namedtuple

from mctest_load import preprocess

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


Reflink = namedtuple('Reflink',['sentence','idx','referent'])


if __name__ == '__main__':
    with open('input.txt','r') as f:
        text = f.read()


    with open('input.txt.out','r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if len(l)>0]


def parse(text, lines):
    delimiter = '<coreference>'

    for line in lines:
        if delimiter in line:
            idx = lines.index(delimiter)
            break

    lines = lines[idx:]
    lines = lines[1:-3]


    def build_entities(lines):
        start, end = 0, 0
        entities = []
        for idx in range(len(lines)):
            if '<coreference>' in lines[idx]:
                start = idx
            elif'</coreference>' in lines[idx]:
                end = idx + 1
                entities.append(lines[start:end])

        return entities


    def build_mentions(entity):
        start, end = 0, 0
        mentions = []
        for idx in range(len(entity)):
            if '<mention' in entity[idx]:
                start = idx
            elif '</mention' in entity[idx]:
                end = idx + 1
                mentions.append(entity[start:end])

        return mentions

    def find_referent(entity):
        for tag in entity:
            if '<mention representative' in tag:
                flag = True
            elif '<text>' in tag and flag == True:
                referent = tag.replace('<text>','')
                referent = referent.replace('</text>','')
                break
        return referent

    def build_reflinks(entity):
        mentions = build_mentions(entity)
        referent = find_referent(entity)

        reflinks = []
        for mention in mentions:
            for item in mention:
                if '<sentence>' in item:
                    sentence = item.replace('<sentence>','')
                    sentence = sentence.replace('</sentence>','')
                    sentence = int(sentence) - 1
                if '<start>' in item:
                    idx = item.replace('<start>','')
                    idx = idx.replace('</start>','')
                    idx = int(idx) - 1
                    reflinks.append(Reflink(sentence, idx, referent))

        return reflinks

    def execute_reflink(reflink, sentences):
        referent = reflink.referent
        sentence = reflink.sentence 
        location = reflink.idx
        
        try:
            if len(referent.split()) < 2:
                sentences[sentence][location] = referent
            # elif len(referent.split()) < 3:
            #     before = sentences[sentence][:location]
            #     insert = referent.split()
            #     after = sentences[sentence][location+len(referent.split()):]
            #     sentences[sentence] = before + insert + after
        except IndexError:
            print 'Indexing error encountered'

        return sentences


    entities = build_entities(lines)

    all_links = []
    for entity in entities:
        links = build_reflinks(entity)
        all_links.append(links)

    text = tokenizer.tokenize(text)
    text = [t.replace('\n',' ') for t in text]
    sentences = [nltk.word_tokenize(t) for t in text]

    for link_set in all_links:
        for link in link_set:
            sentences = execute_reflink(link, sentences)

    return sentences


if __name__ == '__main__':
    print parse(text, lines)
