import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset_youtube_processado_modulo2.csv")

df = pd.read_csv(CSV_PATH)

scaler = MinMaxScaler()
X = scaler.fit_transform(df[['velocidade_views', 'taxa_conversao', 'taxa_discussao']])

pca = PCA(n_components=1)
pca.fit_transform(X)

print("Variância explicada:", pca.explained_variance_ratio_)
print("Pesos (loadings):", pca.components_)