import pandas as pd

df = pd.read_csv('dataset_youtube_processado_modulo2.csv')
df['nicho'] = df['nicho'].str.replace('_', ' ')
df.to_csv('dataset_youtube_processado_modulo2.csv', index=False)

print(df['nicho'].unique())