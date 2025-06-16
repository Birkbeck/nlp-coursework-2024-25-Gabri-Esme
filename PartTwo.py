import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
tfidf_matrix = vect.fit_transform(df['speech'])

# Stratified sampling
X = df['speech']
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    stratify=y,
    random_state=26 
)


