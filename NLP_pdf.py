# Example of taking a PDF document and doing basic NLP with SpaCy in Python
# Jeremy Harris / 11/21/2020

import pandas as pd
import textract #need to run: pip install textract on machine
from autocorrect import Speller #need to run: pip install autocorrect on machine
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import spacy
from spacy.matcher import PhraseMatcher

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

#load English language module for nlp
nlp = spacy.load('en_core_web_sm')

#process text with spacy and the English module loaded above
doc = nlp(spellText)

#####
# Maybe from here, do a word cloud or some other way to show the sentiment of the document?
#####