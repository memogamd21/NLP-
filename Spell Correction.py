#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re 
import nltk
import numpy as np
import pandas as pd


# In[3]:


import pickle    #This imports the list of correct vocabulary to compare with
with open('emmasvocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open('modemma.pkl','rb') as fm: #This imports the rawtext that includes the wrong words to be corrected
    rawtext = pickle.load(fm)
vocab.append("'s") #To make an efficient corrector we append "'s" to our vocabulary data because the tokenizer gives "'s" a token so it might affect accuracy 


# In[3]:


rawtext


# In[6]:


from nltk.tokenize import WordPunctTokenizer, TreebankWordTokenizer
words = nltk.regexp_tokenize(rawtext,r'\w+') #this tokenizer uses regular expression to give only words without punctuation
#TreebankWordTokenizer().tokenize(rawtext) 


# In[7]:


def solve(X, Y): #This function is used to calculate minimum edit distance, it takes two words to compute the distance between them
    mem = - np.ones((len(X)+1,len(Y)+1))
    N = len(X)
    M = len(Y)
    #we use the levenshtein table to compute minimum edit distance,
    #so at first we fill the initial row and column 
    for i in range(N+1):
        mem[i,0] = i
    for j in range(M+1):
        mem[0,j] = j
    #here to add a value in a cell we choose the minimum of three options : 
    #moving rightward plus 1 or moving upward plus 1 or diagonally plus either 
    #two or zero depending on the letters existing (this process represents insertion,deletion
    #,substitution costs)
    for i in range(1,N+1):
        for j in range(1,M+1):
            mem[i,j] = min(mem[i-1,j]+1, mem[i,j-1]+1, mem[i-1, j-1]+2 if X[i-1]!=Y[j-1] else mem[i-1, j-1])
    return mem[N,M]
solve("execution","intention")


# In[8]:


corrections = [("wrong","correct","Location")] #Here we create the list of corrections to be 
#formatted as 3 columns: wrong word, correct word and its location in the rawtext
for x in words:
    #here is the core implementation of the algorithm, we compare with vocab, compute dist 
    #and select minimum one and append them to the list of corrections
    if x not in vocab:
        lists = []
        for i in range(len(vocab)):
            dist = solve(x,vocab[i])
            lists.append(dist)
        MED = min(lists)
        correct_word = vocab[lists.index(MED)]
        corrections.append((x,correct_word,words.index(x)))
df = pd.DataFrame(corrections)
df


# In[10]:


wordss = TreebankWordTokenizer().tokenize(rawtext) #here we repeat the exact sameprocess
#but using anothe type of tokenizer just to show the effect of tokenization on our performance
correctionss = [("wrong","correct","Location")]
for xx in wordss:
    if xx not in vocab:
        listss = []
        for ii in range(len(vocab)):
            dist = solve(xx,vocab[ii])
            listss.append(dist)
        MEDd = min(listss)
        correct_wordd = vocab[listss.index(MEDd)]
        correctionss.append((xx,correct_wordd,wordss.index(xx)))
dff = pd.DataFrame(correctionss)
dff


# In[ ]:




