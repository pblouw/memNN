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

Uom = np.random.random((dim, len(vocab))) * 2 * 0.1 - 0.1
Uoq = np.random.random((dim, len(vocab))) * 2 * 0.1 - 0.1


wrd_to_ind = {wrd:ind for ind, wrd in enumerate(vocab)}
ind_to_wrd = {ind:wrd for ind, wrd in enumerate(vocab)}

def binvec(item):
	vec = np.zeros(len(vocab))
	inds = [wrd_to_ind[w] for w in item.split()]
	vec[inds] = 1
	return vec
	
q = 'where is mary'
m = memory[1]

q_embed = np.dot(Uoq, binvec(q))

print q
print m
print ''

rate = 0.05
gamma = 0.1


# Get scores
for i in range(10):
	error = []
	for x, y in memory.iteritems():
		vec = binvec(y)
		target = np.dot(q_embed, np.dot(Uom, binvec(m)))

		m_embed = np.dot(Uom, vec)
		score = np.dot(q_embed, m_embed)
		loss = max(score + gamma - target, 0)
		if y != m:
			error.append(loss)
		else:
			error.append(0)

		for e in range(len(error)): 

			if error[e] > 0:
				q_grad = np.outer(q_embed * np.dot(Uom, binvec(memory.values()[e])), binvec(q))
				m_grad = np.outer(q_embed, binvec(memory.values()[e]))

				Uoq += -rate * q_grad
				Uom += -rate * m_grad
	print 'Loss:', sum(error)

# error = []
print ''
print 'Final Results'
print ''
# Get scores
for x, y in memory.iteritems():
	vec = binvec(y)
	m_embed = np.dot(Uom, vec)
	score = np.dot(q_embed, m_embed)
	print y, score

