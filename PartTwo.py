import pandas as pd

# Part Two
# (a)
original_df = pd.read_csv("p2-texts/hansard40000.csv")
df = original_df

# i. 
df["party"] = df["party"].str.replace('Labour (Co-op)', 'Labour')

# ii. 
print(df['party'].value_counts())
'''Top 4 are: 
Conservative                        25079
Labour                               8038
Scottish National Party              2303
(Speaker                               878)
Liberal Democrat                      803'''

df = df[df['party'].isin(['Conservative', 'Labour', 'Scottish National Party', 'Liberal Democrat'])]
print(df['party'].unique())

#iii.
df = df[df['speech_class'].isin(['Speech'])]
print(df['speech_class'].unique())



