import string
import os
import sys
import nltk
import numpy as np

with open(os.getcwd()+'/data/en/qa1_single-supporting-fact_train.txt', 'r') as f:
	text = f.readlines()


# Build vocab, dict of memories, and dict of queries 

def preprocess(sen_list):
	sen_list = [s.replace('\n', ' ') for s in sen_list]
	sen_list = [s.replace('\t', ' ') for s in sen_list]
	sen_list = [s.translate(None, string.punctuation) for s in sen_list]
	sen_list = [s.translate(None, '1234567890') for s in sen_list]
	sen_list = [nltk.word_tokenize(s) for s in sen_list]
	sen_list = [[w.lower() for w in s] for s in sen_list]
	return sen_list

def stripline(line):
	line = line.replace('\n', ' ')
	line = line.replace('\t', ' ') 
	line = line.translate(None, string.punctuation)
	line = line.translate(None, '0123456789')
	line = nltk.word_tokenize(line.lower())
	line = ' '.join(line)
	return line

def flatten(lst, acc):
    for item in lst:
        if type(item) == type([]):
            flatten(item, acc)
        else:
            acc.append(item)
    return acc

vocab = list(set(flatten(preprocess(text), [])))
memory = dict()
queries = dict()

old_ind = 0
for line in text:
	new_ind = int(line[0])
	if new_ind < old_ind: # Stop after one story
		break
	elif '?' not in line:
		memory[new_ind] = stripline(line)
	elif '?' in line:
		query = line.split('\t')
		queries[stripline(query[0])] = (stripline(query[1]), query[2].replace('\n',''))
	old_ind = new_ind


# Build embedding models 
dim = 16

Uo = np.random.random((dim, 2*len(vocab))) * 2 * 0.1 - 0.1
Ur = np.random.random((dim, 2*len(vocab))) * 2 * 0.1 - 0.1


wrd_to_ind = {wrd:ind for ind, wrd in enumerate(vocab)}
ind_to_wrd = {ind:wrd for ind, wrd in enumerate(vocab)}

def binvec(item, key):
	vec = np.zeros(len(vocab)*2)
	if key == 'm':
		inds = [wrd_to_ind[w]*2+1 for w in item.split()]
	elif key == 'q':
		inds = [wrd_to_ind[w]*2 for w in item.split()]
	else:
		raise Warning('Invalid Key')
		return None
	vec[inds] = 1
	return vec


q = 'where is mary'

q_embed = np.dot(Uo, binvec(q, key='q'))

# Get best memory 
best = (0, 0)
for x, y in memory.iteritems():
	vec = binvec(y, key='m')
	m_embed = np.dot(Uo, vec)
	score = np.dot(q_embed, m_embed)
	if score > best[1]:
		best = (x, score)


print ''
print 'Memory extraction:'
print q
print (memory[best[0]], best[1])


o_embed = np.dot(Ur, binvec(q, key='q')+binvec(memory[best[0]], key='q'))

# Get best response
best = (0, '')
for x, y in wrd_to_ind.iteritems():
	vec = binvec(x, key='m')
	r_embed = np.dot(Ur, vec)
	score = np.dot(o_embed, r_embed)
	if score > best[0]:
		best = (score, x)

print ''
print 'Response extraction:'
print q
print best

print ''
print memory
print queries

