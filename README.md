# ==========================================================
# PROCEDIMENTO / ATIVIDADE Nº 1
# EXPERIMENTO: Criando um modelo de Machine Learning
# OBJETIVO: Classificar espécies de flores Iris
# ==========================================================

# ----------------------------------------------------------
# PASSO 1 - IMPORTAR AS BIBLIOTECAS
# ----------------------------------------------------------

import tensorflow as tf
import pandas as pd

import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# PASSO 1 - CARREGAR O CONJUNTO DE DADOS IRIS
# ----------------------------------------------------------

# Carrega o conjunto de dados Iris
iris = load_iris()

# X contém as características das flores:
# comprimento da sépala
# largura da sépala
# comprimento da pétala
# largura da pétala
X = iris.data

# y contém a espécie de cada flor
y = iris.target

# Criar um DataFrame apenas para visualizar os dados
dados = pd.DataFrame(X, columns=iris.feature_names)

# Adicionar a coluna com a espécie
dados["especie"] = y

print("Primeiras linhas do conjunto de dados:")
print(dados.head())

# ----------------------------------------------------------
# PASSO 2 - DIVIDIR OS DADOS EM TREINO E TESTE
# ----------------------------------------------------------

# 80% dos dados serão usados para treinamento
# 20% serão usados para testar o modelo
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nQuantidade de dados de treinamento:", len(X_train))
print("Quantidade de dados de teste:", len(X_test))

# ----------------------------------------------------------
# PASSO 2 - NORMALIZAR OS DADOS
# ----------------------------------------------------------

# StandardScaler padroniza os valores das características.
# Isso ajuda a rede neural a aprender de maneira mais eficiente.

scaler = StandardScaler()

# Aprende a escala usando somente os dados de treinamento
X_train = scaler.fit_transform(X_train)

# Aplica a mesma transformação nos dados de teste
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# PASSO 3 - CONSTRUIR O MODELO
# ----------------------------------------------------------

# Criamos uma rede neural simples.
# A entrada possui 4 características.
#
# Camada 1: 16 neurônios
# Camada 2: 8 neurônios
# Saída: 3 neurônios, pois existem 3 espécies de Iris.

model = tf.keras.Sequential([

    tf.keras.Input(shape=(4,)),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])

# ----------------------------------------------------------
# CONFIGURAR O MODELO
# ----------------------------------------------------------

model.compile(

    # Adam é o algoritmo utilizado para ajustar os pesos
    optimizer="adam",

    # Como as classes são números 0, 1 e 2,
    # utilizamos sparse_categorical_crossentropy
    loss="sparse_categorical_crossentropy",

    # Métrica utilizada para avaliar o modelo
    metrics=["accuracy"]
)

# Mostrar a estrutura da rede neural
print("\nEstrutura do modelo:")
model.summary()

# ----------------------------------------------------------
# PASSO 4 - TREINAR O MODELO
# ----------------------------------------------------------

print("\nIniciando treinamento...")

historico = model.fit(

    X_train,
    y_train,

    # Número de vezes que o modelo verá os dados
    epochs=100,

    # Quantidade de dados processados por vez
    batch_size=8,

    # Mostra informações durante o treinamento
    verbose=1
)

# ----------------------------------------------------------
# PASSO 5 - AVALIAR O MODELO
# ----------------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n-----------------------------------")
print("RESULTADO DO MODELO")
print("-----------------------------------")

print(f"Perda (Loss): {loss:.4f}")
print(f"Precisão: {accuracy * 100:.2f}%")

# ----------------------------------------------------------
# PASSO 6 - FAZER PREVISÕES
# ----------------------------------------------------------

# O modelo retorna probabilidades para cada uma das 3 classes
previsoes = model.predict(X_test)

# Converte as probabilidades para a classe com maior chance
classes_previstas = tf.argmax(
    previsoes,
    axis=1
).numpy()

print("\nAlgumas previsões do modelo:")

for i in range(10):
    classe_real = iris.target_names[y_test[i]]

    classe_prevista = iris.target_names[classes_previstas[i]]

    print(
        f"Flor {i + 1}: "
        f"Real = {classe_real} | "
        f"Prevista = {classe_prevista}"
    )

# ----------------------------------------------------------
# TESTE COM UMA NOVA FLOR
# ----------------------------------------------------------

# Características:
# comprimento da sépala
# largura da sépala
# comprimento da pétala
# largura da pétala

nova_flor = [[
    5.1,
    3.5,
    1.4,
    0.2
]]

# Precisamos normalizar usando o mesmo scaler
nova_flor_normalizada = scaler.transform(nova_flor)

# Fazer previsão
previsao_nova = model.predict(nova_flor_normalizada)

# Encontrar a classe com maior probabilidade
classe_nova = tf.argmax(
    previsao_nova,
    axis=1
).numpy()[0]

print("\n-----------------------------------")
print("TESTE COM UMA NOVA FLOR")
print("-----------------------------------")

print("Características:", nova_flor[0])

print(
    "Espécie prevista:",
    iris.target_names[classe_nova]
)
