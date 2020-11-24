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

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

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

#load English language module for nlp
nlp = spacy.load('en_core_web_sm')

#tokenize the information so that we can work with it in spacy
text_tokens = word_tokenize(spellText)

#pull out stop words and keep remaining text
all_stopwords = nlp.Defaults.stop_words
all_stopwords |= {"as", "if", "cut", "ou", "ant", "oka", "apical", "so", "er", "si", "ans", "ie", "gi",
	"ha", "es", "ne", "ing", "ide", "ma", "ell", "ling", "ori", "ho", "ops", "ent", "ith", "sa", "ist",
	"ol", "a.", "yeah", "--", "de", "ill", "ance", "ef", "di", "tr", "ta", "nd", "pri", "ac", "ard", "lea",}
text_no_stop = [word for word in text_tokens if not word in all_stopwords]

print(text_no_stop)



#process text with spacy and the English module loaded above
#doc = nlp(spellText)



#####
# Maybe from here, do a word cloud or some other way to show the sentiment of the document?
#wordcloud = WordCloud(stopwords=STOPWORDS).generate(doc)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plot.show()