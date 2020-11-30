# Example of taking a PDF document and doing basic NLP with SpaCy in Python
# Jeremy Harris / 11/21/2020

import pandas as pd
import textract #need to run: pip install textract on machine
from autocorrect import Speller #need to run: pip install autocorrect on machine
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import spacy
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#Read in the document. In this case, I'm using the NIST AI meeting transcribed document. 
filename = 'NIST_AI_transcript.pdf'
text = textract.process(filename) #the text is a little mess as the pdf font throws a few erros

#convert from bytes to string so that I can actually do something with the results
encoding = 'utf-8'
strText = text.decode(encoding)

#setup spell checking to catch some errors
#for some reason, the pdf font didn't bring over the letters 'v' or 'y'
spell = Speller(lang='en')
spellText = spell(strText)
spellText = spellText.lower()
spellText = re.sub(r'\s+[a-zA-Z]\s+', ' ', spellText) #remove single characters

#remove numbers/digits
no_num = ''.join(c if c not in map(str,range(0,10)) else "" for c in spellText)

#remove single character
no_single = re.sub(r'\s+[a-zA-Z]\s+', ' ', no_num)

#load English language module for nlp
nlp = spacy.load('en_core_web_sm')

#had to add in a few extra stop words because of the poor translation from pdf to work and missing letters
all_stopwords = nlp.Defaults.stop_words
all_stopwords |= {"as", "if", "cut", "ou", "ant", "oka", "apical", "so", "er", "si", "ans", "ie", "gi",
	"ha", "es", "ne", "ing", "ide", "ma", "ell", "ling", "ori", "ho", "ops", "ent", "ith", "sa", "ist",
	"ol", "a.", "yeah", "--", "de", "ill", "ance", "ef", "di", "tr", "ta", "nd", "pri", "ac", "ard", "lea",
	"pa", "ia", "bod", "da", "yup", "yep", "lot", "et", "ed", "mo", "/", "//", "e.", "la", "lo", "e", "-up",
	"dif", "gu", "ste", "al", "ere", "'", ".", "spec", "co", "cult", ""}

#tokenize the information so that we can work with it in spacy
text_tokens = word_tokenize(no_single)
#full_text = nlp(spellText)

#remove stop words
no_stops = [word for word in text_tokens if not word in all_stopwords]

#remove punctuation
punctuations = ['.', '..', ',', '/', '!', '?', ';', ':', '(',')', '[',']', '-', '_', '%']
no_punc = [word for word in no_stops if not word in punctuations]

# lemmatize words
not_token_data = ' '.join(no_punc)

lemmatizer = WordNetLemmatizer()
lemmatized_data = [lemmatizer.lemmatize(word) for word in not_token_data]

# clean data
clean_data = ''.join(lemmatized_data)

### Start to Analyze ###
#create dataframe to analyze data
df = pd.DataFrame([clean_data])
df.columns = ['words']
df.index = ['document']

# create a bag of words and count the frequency of each word
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

corpus = df.words
vect = CountVectorizer(stop_words='english')

data_vect = vect.fit_transform(corpus)

#now let's get the count
tags = vect.get_feature_names()
features = pd.DataFrame(data_vect.toarray(), columns=tags)
features.index = df.index

data_count = features.transpose() #get into column format
data_count.columns = ['freq'] #change name to freq
data_count = data_count.sort_values(by='freq', ascending=False) #sort data

cloud_data = data_count[:50] #extract the first 50 words

print(cloud_data.shape)

###
#Create word cloud from data


#####
# Maybe from here, do a word cloud or some other way to show the sentiment of the document?
#wordcloud = WordCloud(stopwords=STOPWORDS).generate(doc)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plot.show()