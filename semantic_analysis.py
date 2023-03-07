import spacy
from spacy_lookup import Entity
from spacy import displacy
ynlp = spacy.load('en_core_web_sm')
doc = 'a fluffy cat with blue eyes standing on a step.'
docy = ynlp(doc)
displacy.render(docy,style ='dep',jupyter = True,options ={'distance':80})

