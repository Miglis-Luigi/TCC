import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Carregar os dados com o delimitador correto
dados = pd.read_csv('Dados.csv', delimiter=';')
print("Colunas disponíveis no arquivo:", dados.columns) 

# Preencher valores ausentes com a média da coluna
dados['Tamanho'].fillna(dados['Tamanho'].mean(), inplace=True)
dados['N_Quartos'].fillna(dados['N_Quartos'].mean(), inplace=True)
dados['N_Banheiros'].fillna(dados['N_Banheiros'].mean(), inplace=True)
dados['Distancia_Min'].fillna(dados['Distancia_Min'].mean(), inplace=True)
dados['Distancia_Km'].fillna(dados['Distancia_Km'].mean(), inplace=True)
dados['Piscina_'].fillna(dados['Piscina_'].mean(), inplace=True)
dados['Garagem'].fillna(dados['Garagem'].mean(), inplace=True)

# Separar variáveis independentes e dependentes
X = dados[['Tamanho', 'N_Quartos', 'N_Banheiros', 'Distancia_Min', 'Distancia_Km', 'Piscina_', 'Garagem']]
y = dados['Preco']

# Dividir os dados em conjunto de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Salvar o modelo
joblib.dump(modelo, 'modelo_imoveis.pkl')
