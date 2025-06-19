import pandas as pd
import re
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from nltk.corpus import stopwords
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

# Part Two
# (a)
original_df = pd.read_csv("p2-texts/hansard40000.csv")
df = original_df

# i. 
df["party"] = df["party"].str.replace('Labour (Co-op)', 'Labour')

# ii. 
'''Top 4 are: 
Conservative                        25079
Labour                               8038
Scottish National Party              2303
(Speaker                               878)
Liberal Democrat                      803'''

df = df[df['party'].isin(['Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat'])]

#iii.
df = df[df['speech_class'].isin(['Speech'])]

#iv.
df = df[df['speech'].str.len() >= 1000]

print(df.shape)

# (b)
vect = TfidfVectorizer(max_features=3000)
X = vect.fit_transform(df['speech'])

# Stratified sampling
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    stratify=y,
    random_state=26 
)

# (c)

# RandomForest
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Evaluation:", f1_score(y_test, rf_pred, average='macro'))
print(classification_report(y_test, rf_pred))

# SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Evaluation:", f1_score(y_test, svm_pred, average='macro'))
print(classification_report(y_test, svm_pred))

# (d)
vect = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
X = vect.fit_transform(df['speech'])
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    stratify=y,
    random_state=26 
)

# RandomForest
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Evaluation:", f1_score(y_test, rf_pred, average='macro'))
print(classification_report(y_test, rf_pred))

# SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Evaluation:", f1_score(y_test, svm_pred, average='macro'))
print(classification_report(y_test, svm_pred))

# (e)
stop_words = set(stopwords.words('english'))
crucial_words = {'scotland', 'scottish', 'independence', 'conservative', 'labour', 'liberal', 'snp', 'tory', 'libdem', 'uk', 'london', 'britain', 'bill', 'parliament', 'tourism', 'nhs', 'economy', 'housing'}

def custom_tokenize(text):
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        lemma = token.lemma_.lower()

        # Always keep crucial words
        if lemma in crucial_words:
            tokens.append(lemma)

        # Replace named entities unless they are crucial
        elif token.ent_type_ and lemma not in crucial_words:
            tokens.append(token.ent_type_)

        # Keep clean non-stopword tokens
        elif token.is_alpha and lemma not in stop_words:
            tokens.append(lemma)

    return tokens

vect = TfidfVectorizer(tokenizer=custom_tokenize, ngram_range=(1, 3), max_features=3000)
X = vect.fit_transform(df['speech'])
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    stratify=y,
    random_state=26 
)

# SVM
svm = SVC(kernel='linear', class_weight='balanced')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Evaluation:", f1_score(y_test, svm_pred, average='macro'))
print(classification_report(y_test, svm_pred))

# #  (f)
''' 
***Intro***
I began with the baseline model that had 0.75 - 0.8 accuracy. I noted that while this was a relatively good accuracy score by itself, the model was only perfoming well on the conservative class, which represented a disproportionatley large amount of the data. 
My goal was therefore to try and increase performance amogst the smaller parties, while maintaining high accuracy. I targeted reaching > 0.7 on all scores and an accuracy > 0.8. 

To begin exploring the data, I checked for the most used words per party, however, these were all stopwords. By disregaring them in the vectoriser, performance increased.  
Next, I experimented with a different number of ngrams, whereby allowing 1 - 3 yielded best results. I also added the parameter to balance the class weights, as the data set was very unbalanced. 

Next, I wanted to work directly with the tokenisation. My hunch was that analysing the topics discussed by each party would improve classification, but at the same time, customising a topic dictionary may introduce too many parameter.
First, I lemmatised the tokens and this helped. As a lighter alternative instead of topic modelling, I explored using named entity recognition to retain meaningful entities instead of a wide range of words, which significanty boosted performance. However, performance significantly dropped for SNP party. I believed this approach may have been too restrictive, and experimented with reintroducing words that would have been "swallowed" by the NER. This included scotland and scottish, but also the parties names. 
I played around with the word I had reintroduced to get a sense for which made the most difference and ultimately decided on a handful of words.

***Tokenizer Function***
The tokenizer loops through all words in a doc and converts them to it's lemma. Next, it will check if any of those lemmas are part of the custom crucial words, and if so, save them to the tokens list. 
Next, if the lemmatised word is not in the crucial words and can be mapped to a named entity mapping, then the mapping will be added to the tokens list (as opposed to the word). 
Any remaining lemmatised words (not spaces or punctuation), will only be passed to the token list if the are not within the nltk stop-words list (I moved stopword removal into the tokenizer to save looping through the words again during the vectorizer step).
The vectorizer then uses my tokenizer to tokenize the doc.

***Performance***
I abandonded the RF clasifier as it did not improve much, howvever the SVM did. It achieved 0.8 total accuracy. I was not able to reach > 0.7 precision for SNP or Lib Dem, but was satified with 0.63 and 0.45 respectively. Recall reached >0.7 for all but Lib Dem (0.43). I decided to stop here, as I was saitfied that to reach a higher degree of accuracy, more in depth tuning would be required.

***Conclusion***
In the essence of balancing parametres with performance, I thought this would be the right point to stop. 
The only logical way I believed to improve the performance would be by analysing the topics and type of topics each party would speak about, and that would require more parametres and in-depth analysis. 
I also wanted to avoid overfitting the model/artificially boosting the performance by increasing sensitivtiy to historical topics that were relevant at the time the data was collected, but would likely do little moving forward (e.g. lockdown, brexit, Covid-19)'''
