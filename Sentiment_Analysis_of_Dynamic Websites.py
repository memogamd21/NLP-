#!/usr/bin/env python
# coding: utf-8

# Part one 

# In[1]:


import numpy as np 
from bs4 import BeautifulSoup #For html parser
import os #to display the  directory contents (not necessary at all)
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import state_union
from sklearn.naive_bayes import MultinomialNB


# In[2]:


df=pd.read_csv('movie_data.csv')
lists = []
stop_words = list(set(stopwords.words("english")))
for i in range(len(df)):
    sent = BeautifulSoup(df['review'][i],"lxml").text
    words = RegexpTokenizer('\w+').tokenize(sent)
    filtered_sent = []
    for m in range(len(words)):
        if words[m] not in stop_words:
            filtered_sent.append(words[m])
    filteredsent = ' '.join(filtered_sent)
    lists.append((i,filteredsent,df['sentiment'][i]))


# In[ ]:


lists = pd.DataFrame(lists)
lists.columns = ["index","review","sentiment"]
lists


# In[ ]:


def review_to_words(raw_review):
    #stemmer = SnowballStemmer("english")
    Lemmatizer = WordNetLemmatizer()
    words = raw_review.split()
    tagged = nltk.pos_tag(words)
    stemming_words = [Lemmatizer.lemmatize(w) for w in words]#[stemmer.stem(w) for w in words] 
    for m in range(len(words)):
        #if (words[m] == "not" or words[m] == "Not") and m <= len(words)-3:
            #for j in range(2):#case one in the comparison when only 3 words to be negated
                #stemming_words[m+j+1] = "NOT_" + stemming_words[m+j+1]
        if (words[m] == "not" or words[m] == "Not"):
            for jj in range(len(words)-m-2):#case two in the comparison when all the remaining words are negated
                stemming_words[m+jj+1] = "NOT_" + stemming_words[m+jj+1]
    newlist = []
    for i in range (len(words)):
        newlist.append(stemming_words[i]+"_"+tagged[i][1])
    return( ' '.join(newlist))
lists['review'] = lists['review'].apply(review_to_words)
lists 


# In[ ]:


cv = CountVectorizer(min_df = 1, stop_words = 'english')#TfidfVectorizer(min_df = 1, stop_words = 'english')
x_train,x_test,y_train,y_test = train_test_split(lists["review"],lists["sentiment"],test_size = 0.2, random_state=4)
x_traincv = cv.fit_transform(x_train)
y_train = y_train.astype('int')
RFC = RandomForestClassifier()
mnb = MultinomialNB()
mnb.fit(x_traincv, y_train) 
RFC.fit(x_traincv, y_train) 


# In[ ]:


x_testcv = cv.transform(x_test)
pred = RFC.predict(x_testcv)
pred
predds = mnb.predict(x_testcv)
actual = np.array(y_test)
count = 0
countt = 0
for i in range (len(pred)):
    if pred[i] == actual[i]:
        count = count+1
for ii in range (len(predds)):
    if predds[ii] == actual[ii]:
        countt = countt+1
print("Accuracy is " + str(100*count/len(pred)) + " %")
print("Accuracy is " + str(100*countt/len(predds)) + " %")


# Cases For the Different Accuracies : You can test them by commenting/decommenting the other part of the comparison as shown in the code********
# 
# Stemming -->76.52         Lemmatizing -->76.57  -->Comment : Not so much different because the value that is going to be given for both cases by countvectorizer is almost the same
# 
# No Stop Words --> 76.92   With Stop Words --> 76.35  --> Comment : Not so much different because the value that is going to be given for both cases by countvectorizer is almost the same
# 
# CountVectorizer -->77.89  TFIDFvectorizer --> 77.38  --> Comment : Since the only difference is that TFVC returns floats and CV returns ints, it's not expected to have much difference because they work almost the same way
# 
# Not_first_case -->76.71   Not_second_case -->77.26   --> it might be that negating lots of words affect more values in the vectorization step so that it gives better accuracy 
# 
# RandomForest -->77.5      NaiveBayes -->84.82        --> the initial assumptions that were assumed by the naivebayes model were mostly perfect so that it performed better because in normal RFC performs better when the size of the daaset is large like this one but mostly the priori probabilities were given better in the case of the naivebayes

# Second Part --> In the following cell, to load all the comments you have to keep scrolling down 
# 
# in the chrome window of the video that will popup for you. If you make the time.sleep(time = 
# 
# very long one, here is 500 enough for loading all the comments). In addition, you should install 
# 
# selenium at first and make the chrome.exe file in the path of the file(just wait run it as 
# 
# normal and you will get an error to navigate you, follow it :D ). In this section, all comments
# 
# were positive 

# In[24]:


from selenium import webdriver
from nltk.tokenize import sent_tokenize
import time
driver=webdriver.Chrome()
driver.get('https://www.youtube.com/watch?v=iFPMz36std4')
driver.execute_script('window.scrollTo(1, 500);')
#now wait let load the comments
time.sleep(500)
driver.execute_script('window.scrollTo(1, 3000);')
comment_div=driver.find_element_by_xpath('//*[@id="contents"]')
comments=comment_div.find_elements_by_xpath('//*[@id="content-text"]')
commentss = []
for comment in comments:
    commentss.append(comment.text)


# In[25]:


#23ml sentence tokenization llcomments.text w apply elmodel 3leha 
X_testcv = cv.transform(commentss)
predsss = RFC.predict(X_testcv)
predsss


# In[26]:


countrr1 = 0
countrr2 = 0
for i in range (len(predsss)):
    if(predsss[i] == 0):
        countrr1 += 1
    elif(predsss[i] == 1):
        countrr2 += 1
countrr1,countrr2


# In[27]:


commentss


# In[28]:


len(commentss)


# In[ ]:




