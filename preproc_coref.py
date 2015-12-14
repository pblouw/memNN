import os
import os.path
import cPickle as pickle
import shutil
import subprocess
from tempfile import mkdtemp

from mctest_load import load_stories, Story
from parse import parse


def process(path):
    stories = load_stories(path + '.tsv', path + '.ans')
    processed = []
    filelist = []
    try:
        tmpdir = mkdtemp()
        for i, story in enumerate(stories):
            infile = os.path.join(tmpdir, str(i) + '.txt')
            with open(infile, 'w') as f:
                f.write(story.text)
            
            filelist.append(infile)

        filelist_filename = os.path.join(tmpdir, 'filelist')
        with open(filelist_filename, 'w') as f:
            for name in filelist:
                f.write(name)
                f.write(os.linesep)

        subprocess.check_call([
            'java', '-cp', '*', '-Xmx2g',
            'edu.stanford.nlp.pipeline.StanfordCoreNLP',
            '-annotators', 'tokenize,ssplit,pos,lemma,ner,parse,dcoref',
            '-filelist', filelist_filename])

        for i, story in enumerate(stories):
            outfile = str(i) + '.txt.out'
            with open(outfile) as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines]
                lines = [l for l in lines if len(l) > 0]
                
            processed.append(Story(parse(story.text, lines), story.queries))    
    finally:
        shutil.rmtree(tmpdir)

    with open(path + '.coref', 'wb') as f:
        pickle.dump(processed, f, 2)

if __name__ == '__main__':
    os.chdir('stanford-corenlp-full-2015-12-09')
    path = os.path.join(os.path.pardir, 'MCTest')
    for filename in os.listdir(path):
        if filename.endswith('.tsv'):
            process(os.path.join(path, os.path.splitext(filename)[0]))
