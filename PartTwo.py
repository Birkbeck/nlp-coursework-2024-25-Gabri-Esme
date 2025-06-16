import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

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

