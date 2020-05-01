from dados import carregar_acessos


X, Y = carregar_acessos()

treino_dados = X[:90]
treino_marcacoes = Y[:90]

dados_teste = X[-9:]
marcacoes_teste = Y[-9:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(dados_teste)

diferencas = resultado - marcacoes_teste

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(dados_teste)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)
