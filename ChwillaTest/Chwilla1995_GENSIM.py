import scipy
from scipy import stats


import numpy as np
import matplotlib as mpl

## agg backend is used to create plot as a .png file
#mpl.use('agg')

import matplotlib.pyplot as plt 


#from wikipedia2vec import Wikipedia2Vec

import gensim
from gensim import models
from gensim.models import Word2Vec

import gensim.downloader as api

import csv

np.random.seed(1)
#vectorDim = 100

#numNode = 100

#inputDataTraining = np.load('./trainingData_averaging/inputDataTraining_4k_average.npy')
#outputDataTraining = np.load('./trainingData_averaging/outputDataTraining_4k_average.npy')



#pretrained_model = "fasttext-wiki-news-subwords-300"
##pretrained_model = "conceptnet-numberbatch-17-06-300"
##pretrained_model = "word2vec-ruscorpora-300"
#pretrained_model = "word2vec-google-news-300"
#pretrained_model = "glove-wiki-gigaword-50"
#pretrained_model = "glove-wiki-gigaword-100"
#pretrained_model = "glove-wiki-gigaword-200"
#pretrained_model = "glove-wiki-gigaword-300"
#pretrained_model = "glove-twitter-25"
#pretrained_model = "glove-twitter-50"
#pretrained_model = "glove-twitter-100"
pretrained_model = "glove-twitter-200"

nlp = api.load(pretrained_model)

print('model loaded')

#wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')

related_1 = ['vein', 'baker', 'leg', 'dark', 'building', 'insane', 'storm', 'left', 'sailor', 'leader', 'ball', 'day', 'coast', 'bed', 'long', 'shark', 'glory', 'animal', 'green', 'law', 'village', 'throw', 'threshold', 'alive','star', 'repent', 'answer', 'chilly', 'nice', 'dog', 'berry', 'empty', 'male', 'fever', 'halt', 'sweet', 'trembling', 'leek', 'paper', 'window']
related_2 = ['blood', 'bread', 'arm', 'light', 'flat', 'crazy', 'wind', 'right', 'ship', 'boss', 'round', 'night', 'sea', 'sheet', 'short', 'fish', 'fame', 'beast', 'grass', 'justice',   'town'   , 'toss',  'door',       'dead','sky',    'regret', 'question', 'cold', 'sweet', 'cat', 'fruit', 'full', 'female', 'ill', 'wrong','honey', 'shaking', 'vegetables', 'pen', 'pane' ]
unrelated_1 = ['sweat', 'stove', 'dust', 'scarf', 'hungry', 'revenge', 'rig', 'eat', 'target', 'hose', 'youth', 'field', 'tight', 'washing', 'quarter', 'package', 'palace', 'times', 'mess', 'monk', 'nation', 'cake', 'platform', 'record', 'panic', 'love', 'hairdresser', 'lock', 'word', 'jelly',   'skull','farmer','rot', 'set',   'rock','put','mist','sound','box','kilo']
unrelated_2 = ['text', 'sports', 'safe', 'ground', 'bush', 'lip', 'wall', 'mail', 'belly', 'failure', 'battery', 'gray', 'point', 'bike', 'family', 'roof', 'monkey', 'child', 'wolf', 'stairs',      'bowl', 'slow', 'sprayer',   'nose',   'blouse', 'sphere', 'more',   'out',  'pocket',  'language', 'taxi', 'cry', 'free', 'purse', 'fit', 'spot','leak','dress','stitch','minute']

for num in range(len(related_1)):
    print(' %i - Related (%s, %s);    Unrelated (%s, %s)' % (num, related_1[num], related_2[num], unrelated_1[num], unrelated_2[num]))


related_data = np.empty(len(related_1))
print(related_data.shape)

unrelated_data = np.empty(len(related_1))
print(unrelated_data.shape)

# calculate cosines for related and unrelated pairs 
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

related_data = []
unrelated_data = []

print('\nRelated:')
for num in range(len(related_1)):
    cos_val = cos_sim(nlp[related_1[num]], nlp[related_2[num]])
    #print('cos(%s, %s)=%f' % (related_1[num], related_2[num], cos_val))
    related_data = np.append(related_data, cos_val)
    
#for num in range(len(related_1)):
#    print('%f' % (cos_sim(nlp[related_1[num]], nlp[related_2[num]])))
    
print(related_data)

print('\nUnRelated:')
for num in range(len(related_1)):
    cos_val = cos_sim(nlp[unrelated_1[num]], nlp[unrelated_2[num]])
    #print('cos(%s, %s)=%f' % (unrelated_1[num], unrelated_2[num], cos_val))
    unrelated_data = np.append(unrelated_data, cos_val)
    
#for num in range(len(related_1)):
#    print('%f' % (cos_sim(nlp[unrelated_1[num]], nlp[unrelated_2[num]])))
    
print(unrelated_data)

data_to_plot = [unrelated_data, related_data]

# Create a figure instance
#fig = plt.figure(1, figsize=(9, 6))
fig = plt.figure(1)
# Create an axes instance
ax = fig.add_subplot(111)
plt.ylabel('Semantic Relatedness')


# Create the boxplot
bp = ax.boxplot(data_to_plot)

ax.set_xticklabels(['Related-Pairs', 'Unrelated-Pairs'])
fig.tight_layout()
plt.savefig('pairs-chwilla1995-GENSIM-'+pretrained_model+'.png')
plt.show()

print(stats.ttest_ind(related_data,unrelated_data))
with open('ttest_ind-GENSIM-'+pretrained_model+'.csv', 'w') as f_ttest_ind:
    writer = csv.writer(f_ttest_ind)
    writer.writerow([stats.ttest_ind(related_data,unrelated_data)[0], stats.ttest_ind(related_data,unrelated_data)[1]])


print(np.mean(related_data))
print(np.mean(unrelated_data))

fig= plt.figure(figsize=(12,5))
# Cut your window in 1 row and 2 columns, and start a plot in the first part
plt.subplot(111)
plt.hist(related_data, 10, alpha=0.5, label='Related Pairs')
plt.hist(unrelated_data, 10, alpha=0.5, label='Unrelated Pairs')

# Add title and axis names
# plt.title('Before Discourse')
plt.xlabel('Semantic Relatedness')
plt.ylabel('Instances')
plt.legend(loc='best')
#plt.xlim(0.40, 0.67)
#plt.ylim(0, 18)
 

 
# Show the graph
# plt.savefig('ChwillaPairsHistogram-GENSIM-'+pretrained_model+'.png')
plt.show()

#fig= plt.figure(figsize=(12,5))

fig = plt.figure(1, figsize=(7,7))
#fig = plt.figure(1)
# Create an axes instance
ax = fig.add_subplot(211)
plt.ylabel('Semantic Relatedness')


# Create the boxplot
bp = ax.boxplot(data_to_plot)

ax.set_xticklabels(['Unrelated-Pairs', 'Related-Pairs'])
fig.tight_layout()
#plt.savefig('pairs-chwilla1995-GENSIM-'+pretrained_model+'.png')
#plt.show()

# Cut your window in 1 row and 2 columns, and start a plot in the first part
plt.subplot(212)

plt.hist(unrelated_data, 10, alpha=0.5, label='Unrelated Pairs')
plt.hist(related_data, 10, alpha=0.5, label='Related Pairs')

# Add title and axis names
# plt.title('Before Discourse')
plt.xlabel('Semantic Relatedness')
plt.ylabel('Instances')
plt.legend(loc='best')
#plt.xlim(0.40, 0.67)
#plt.ylim(0, 18)
 

 
# Show the graph
plt.savefig('ChwillaPairsBOXandHistogram-GENSIM-'+pretrained_model+'.png')
plt.show()

#fig= plt.figure(figsize=(12,5))

fig = plt.figure(1, figsize=(8,4))
#fig = plt.figure(1)
# Create an axes instance
ax = fig.add_subplot(122)
plt.ylabel('Semantic Relatedness')


# Create the boxplot
bp = ax.boxplot(data_to_plot)

ax.set_xticklabels(['Unrelated-Pairs', 'Related-Pairs'])
fig.tight_layout()
#plt.savefig('pairs-chwilla1995-GENSIM-'+pretrained_model+'.png')
#plt.show()

# Cut your window in 1 row and 2 columns, and start a plot in the first part
plt.subplot(121)

plt.hist(unrelated_data, 10, alpha=0.5, label='Unrelated Pairs')
plt.hist(related_data, 10, alpha=0.5, label='Related Pairs')

# Add title and axis names
# plt.title('Before Discourse')
plt.xlabel('Semantic Relatedness')
plt.ylabel('Instances')
plt.legend(loc='best')
#plt.xlim(0.40, 0.67)
#plt.ylim(0, 18)
 

 
# Show the graph
plt.savefig('ChwillaPairsBOXandHistogram-GENSIM-'+pretrained_model+'.png')
plt.show()

A = nlp['vein']
B = nlp['blood']
C = nlp['sweat']
D = nlp['text']
E = nlp['love']
F = nlp['love']

print('Cosine similarity: LOVE IN LOVE blood = %f' % (cos_sim(E,F)))

print(A.shape)

print('Cosine similarity: vein blood = %f' % (cos_sim(A,B)))
print('Cosine similarity: sweat text = %f' % (cos_sim(C, D)))


print(A)

