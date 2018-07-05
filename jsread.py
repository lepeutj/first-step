from pprint import pprint
import json
import re
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import pyLDAvis.gensim
import gensim
import string

def clean_text(text):
    text = re.sub(r"(?:\@|http?\://)\S+", " ", text)
    text = re.sub(r"(?:\@|https?\://)\S+", " ", text)
    text = re.sub(r"(#)", " ", text)
    text = re.sub(r"(\W)", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = text.lower()
    return text;


p_stemmer = PorterStemmer()

file =  open("tweets.json")
tw = json.load(file)
# tw_indent = json.dumps(tw,indent=4)

# properties=["screen_name","description","followers_count","friends_count","listed_count","lang"]
# user_info = []

#################  USER TESTS
# for t in range(0,10):
#     for i in properties:
#         user_info = np.append(user_info,tw[t]['user'][i])
# for u in xrange(0,9):
#     df = user_info[u:6(u+1)]
#
# print user_info, len(user_info)
################

#################  Creation dataframe a partir du JSON
td = pd.DataFrame.from_records(tw)

stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','rt']) # remove it if you need punctuation


tw_fr = td.loc[td['lang'] == 'fr']
tw_en = td.loc[td['lang'] == 'en']


################# ETUDE CORPUS ENGLISH
english_corpus = pd.DataFrame({'test' : tw_en['text']})
english_corpus.index = xrange(0,len(tw_en))


#########################
text_cleaned = []
new_text_en=[]

for i in xrange(0,len(tw_en)):
    text_cleaned.append(clean_text(english_corpus['test'][i]))


for t in text_cleaned:
    raw = word_tokenize(t)
    stopped_tokens = [i for i in raw if not i in stop]

    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    new_text_en.append(stemmed_tokens)
#######################


dictionary = corpora.Dictionary(new_text_en)
corpus = [dictionary.doc2bow(text) for text in new_text_en]


# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=7, id2word = dictionary, passes=20)
# ldamodel.save("first_lda")
lda=models.LdaModel.load('first_lda')
visu = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(visu, 'LDA_Visualization.html')




# re.sub(r"([^a-z]|#)","",str_test)

#################

##################
# self.header = ['index'] + [ _ for _ in english_corpus.columns ]
# self.values = []
# for i,row in enumerate(english_corpus.values) :
#     row = [ df.index[i] ] + [ x for x in row ]
#     self.values.append(row)

# id_names = []
# id_desc = []
#
# for i in range(0,len(tw)):
#     id_names = np.append(id_names,tw[i]['user']['screen_name'])
#     id_desc = np.append(id_desc,tw[i]['user']['description'])
##################


################## AGAIN USER IDENT
# df = pd.DataFrame({'id' : id_names, 'desc' : id_desc})
# print df.drop_duplicates()

# print id_names,id_desc
# print len(id_names),"Taille vec description :", len(id_desc)
# print id_names[0],id_names[len(id_names)-1]
##################


# for i in user_info.columns:
#     print i
# for i in td.columns:
#   print i
# print td

# 1 random row added to test commit
