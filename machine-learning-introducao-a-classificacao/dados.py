import csv


def carregar_acessos():
    X = []
    Y = []

    with open("acesso.csv", "rb") as arquivo:
        leitor = csv.reader(arquivo)
        next(leitor)
        for home, como_funciona, contato, comprou in leitor:
            dado = [int(home), int(como_funciona), int(contato)]

            X.append(dado)
            Y.append(int(comprou))

        return X, Y
