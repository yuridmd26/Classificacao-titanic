# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ----- 1. Carregar os dados -----

# Definindo o caminho dos arquivos (substitua com o caminho correto no seu ambiente)
gender_submission_path = 'gender_submission.csv'
test_file_path = 'test.csv'
train_file_path = 'train.csv'

# Carregar os arquivos CSV em DataFrames
gender_submission = pd.read_csv(gender_submission_path)
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)

# ----- 2. Exploração inicial dos dados -----
def explorar_dados(df, nome):
    print(f'\n----- Dados do arquivo {nome} -----')
    print(df.head())
    print(df.info())
    print(df.describe())

explorar_dados(gender_submission, 'gender_submission.csv')
explorar_dados(test_data, 'test.csv')
explorar_dados(train_data, 'train.csv')

# ----- 3. Limpeza de Dados -----
# Função para preencher valores ausentes
def tratar_valores_ausentes(df, nome_arquivo):
    print(f'\nIdentificando valores ausentes no arquivo {nome_arquivo}...')
    print(df.isnull().sum())

# Tratar dados de teste
tratar_valores_ausentes(test_data, 'test.csv')
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data.dropna(subset=['Cabin'], inplace=True)

# Tratar dados de treino
tratar_valores_ausentes(train_data, 'train.csv')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data.dropna(subset=['Cabin', 'Embarked'], inplace=True)

# Verificar valores ausentes após tratamento
print('\nValores ausentes após o tratamento dos dados de treino:')
print(train_data.isnull().sum())

# ----- 4. Seleção de Variáveis -----

# Converter a coluna 'Sex' para variáveis numéricas usando dummy variables
test_data = pd.get_dummies(test_data, columns=['Sex'], drop_first=True)
train_data = pd.get_dummies(train_data, columns=['Sex'], drop_first=True)

# ----- 5. Preparação dos dados para o modelo -----

# Selecionar as variáveis de entrada (X) e a variável alvo (y)
X = train_data[['Pclass', 'Sex_male', 'Age', 'Fare']]
y = train_data['Survived']

# Dividir os dados em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- 6. Treinamento e Previsão com KNN -----

# Instanciar o modelo KNN com k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(X_test)

# ----- 7. Avaliação de Desempenho -----

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAcurácia: {accuracy:.2f}')

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)