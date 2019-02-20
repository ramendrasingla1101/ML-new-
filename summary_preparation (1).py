#!/usr/bin/env python
# coding: utf-8

# In[17]:


# !pip install pdfminer.six
# !
# pdf2txt.py ./data/data1.pdf


# # SUMMARY

# In[17]:


import re
import sys


# In[18]:


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np


# In[29]:


path =sys.argv[1]
rsrcmgr = PDFResourceManager()
retstr = StringIO()
codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
fp = open(path, 'rb')
interpreter = PDFPageInterpreter(rsrcmgr, device)
password = ""
maxpages = 0
caching = True
pagenos=set()

for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
    interpreter.process_page(page)

text = retstr.getvalue()
clean_cont = text.splitlines()
type(clean_cont)
sent_str = ""
for i in clean_cont:
    sent_str += str(i) + " "
sent_str = sent_str[:-1]
from nltk.tokenize import sent_tokenize
clean_cont = sent_tokenize(sent_str)
dubby=[re.sub("[^a-zA-Z]+", " ", s) for s in clean_cont[1:]]
    
##Modelling
    
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
vect=TfidfVectorizer(ngram_range=(1,1),stop_words='english')
dtm=vect.fit_transform(dubby)
result = dtm.toarray()
a = np.array([])
for ix in range(result.shape[0]):
    a=np.append(a,np.sum(result[ix]))
ind = np.argsort(a)
summary = np.array([])
array1 = ind[-2:]
sorted(array1)
for ix in array1:
    summary = np.append(summary,dubby[ix])
summary_str = str(clean_cont[0])+ " "
for ix in summary:
    summary_str+= str(ix)+" "
device.close()
retstr.close()
print(summary_str)


# In[ ]:





# In[ ]:




