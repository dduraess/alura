import os
import re
import csv

cnpjs = []
cpfs = []
pasta = input("Informe a pasta: ")
arquivos = [f for f in os.listdir(pasta) if re.match(r'^elementos-carga.*csv$', f)]

for nome_arquivo in arquivos:
    with open(pasta + nome_arquivo, "r") as arquivo:
        csv_reader = csv.reader(arquivo)
        csv_reader.__next__
        for linha in csv_reader:
            if linha[2] == "CNPJ":
                print("({}, {}), ".format(linha[1][:7], linha[1][8:12]), end='')
            elif linha[2] == "CPF":
                print("{}, ".format(linha[1][:8]), end='')