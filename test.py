import os
import re
import csv

cnpjs = []
cpfs = []
pasta = '/home/davison/Downloads/'
arquivos = [f for f in os.listdir(pasta) if re.match(r'^elementos-carga.*csv$', f)]

for nome_arquivo in arquivos:
    with open(pasta + nome_arquivo, "r") as arquivo:
        csv_reader = csv.reader(arquivo)
        for linha in csv_reader:
            if linha[2] == "CNPJ":
                print(linha[1])
            elif linha[2] == "CPF":
                print(linha[1])