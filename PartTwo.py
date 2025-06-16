import pandas as pd

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


