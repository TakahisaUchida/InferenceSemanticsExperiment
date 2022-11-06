
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import csv

import scipy
from scipy import stats

import spacy


def cos_sim(v1, v2):
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#nlp = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/PeterDell/Downloads/GoogleNews-vectors-negative300.bin', binary=True) #without *norm_only* param
# C:\Users\PeterDell\Google Drive\GoogleWIP\People\Lair\embeddingsExperiments

nlp = spacy.load('../pretrainedData/glove_6B_50d_twitter')

N = 32

A = np.empty(N)

E = np.empty(N)

f = open('McKoon_experiment-twitter-small.csv', 'w')
writer = csv.writer(f, lineterminator='\n')

for i in range(N):
    print('\n############### ' + str(i + 1) + ' ###############')

    # read txt file for data
    f = open('./data_mckoon1986/'+str(i + 1)+'.txt', 'r')
    list = f.readlines()
    discourse_words_1 = list[0].split()
    discourse_words_2 = list[1].split()
    target_word_1 = list[2].lower()

    f.close()

    # large capital -> small capital
    discourse_words_1 = [s.replace(s, s.lower()) for s in discourse_words_1]
    discourse_words_2 = [s.replace(s, s.lower()) for s in discourse_words_2]

    # remove '.' and ',' from word list
    discourse_words_1 = [s.replace('.', '') for s in discourse_words_1]
    discourse_words_2 = [s.replace('.', '') for s in discourse_words_2]
    discourse_words_1 = [s.replace(',', '') for s in discourse_words_1]
    discourse_words_2 = [s.replace(',', '') for s in discourse_words_2]
    discourse_words_1 = [s.replace(';', '') for s in discourse_words_1]
    discourse_words_2 = [s.replace(';', '') for s in discourse_words_2]

    # remove stop words from word list
    stop_words = stopwords.words('english')
    #print(stop_words)
    for stop_word in stop_words:
        while stop_word in discourse_words_1 :
            discourse_words_1.remove(stop_word)
            
        while stop_word in discourse_words_2 :
            discourse_words_2.remove(stop_word)
            
    # remove "'s" and "'" and "-" and "'d" and "'ll" and "'ve" and "re" from word list
    discourse_words_1 = [s.replace("'s", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("'s", '') for s in discourse_words_2]
    discourse_words_1 = [s.replace("'", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("'", '') for s in discourse_words_2]
    discourse_words_1 = [s.replace("-", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("-", '') for s in discourse_words_2]
    discourse_words_1 = [s.replace("'d", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("'d", '') for s in discourse_words_2]
    discourse_words_1 = [s.replace("'ll", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("'ll", '') for s in discourse_words_2]
    discourse_words_1 = [s.replace("'ve", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("'ve", '') for s in discourse_words_2]
    discourse_words_1 = [s.replace("'re", '') for s in discourse_words_1]
    discourse_words_2 = [s.replace("'re", '') for s in discourse_words_2]

    # replace '\n' from target words
    target_word_1 = target_word_1.replace('\n', '')


    print('Data:')
    print('target_word_1: %s' % target_word_1)
   

    print('discourse_words_1:')
    print(discourse_words_1)
    print('discourse_words_2:')
    print(discourse_words_2)


    target_word_1_vector = nlp(target_word_1).vector
 
    '''
    fig, ax = plt.subplots()
    t = np.linspace(1, 2, 2)
    '''

    trajectory_word_1 = np.array([])
 
    print('\nStep1: ')

    for num in range(len(discourse_words_1)):
        #print(discourse_words_1[num])
        if num == 0:
            discourse_vector_1 = nlp(discourse_words_1[num]).vector
        else:
            discourse_vector_1 = (num * discourse_vector_1 + nlp(discourse_words_1[num]).vector) / (num + 1)

    print('cos(discourse_vector_1, %s)=%f' % (target_word_1, cos_sim(discourse_vector_1, target_word_1_vector)))

    A[i] = cos_sim(discourse_vector_1, target_word_1_vector)
    
    trajectory_word_1 = np.append(trajectory_word_1, cos_sim(discourse_vector_1, target_word_1_vector))

    print('\nStep2: ')

    for num in range(len(discourse_words_2)):
        #print(discourse_words_1and2[num])
        if num == 0:
            discourse_vector_2 = nlp(discourse_words_2[num]).vector
        else:
            discourse_vector_2 = (num * discourse_vector_2 + nlp(discourse_words_2[num]).vector) / (num + 1)


    print('cos(discourse_vector_2, %s)=%f' % (target_word_1, cos_sim(discourse_vector_2, target_word_1_vector)))
    
    E[i] = cos_sim(discourse_vector_2, target_word_1_vector)

    writer.writerow([cos_sim(discourse_vector_1, target_word_1_vector),  cos_sim(discourse_vector_2, target_word_1_vector)])

    trajectory_word_1 = np.append(trajectory_word_1, cos_sim(discourse_vector_2, target_word_1_vector))
 

f.close()


data_to_plot = [A,  E]

print(data_to_plot)

# Create a figure instance
#fig = plt.figure(1, figsize=(9, 6))
fig = plt.figure(1)
# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

ax.set_xticklabels(['A', 'E'])
fig.tight_layout()
plt.savefig('McKoon-86-twitter.png')
plt.show()

print('t-test REL: ', stats.ttest_rel(A, E))
print('t-test IND: ', stats.ttest_ind(A, E))

difference = A - E
#print(difference)

