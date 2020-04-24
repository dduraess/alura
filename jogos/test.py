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
        for operacao, valor, tipo in csv_reader:
            if tipo == "CNPJ":
                cnpjs.append(valor)
                # print("({}, {}), ".format(valor[:7], valor[8:12]))
            elif tipo == "CPF":
                cpfs.append(valor)
                # print("{}, ".format(valor[:8]))

        for item in cnpjs:
            print("({}, {}), ".format(valor[:7], valor[8:12]))

        for item in cpfs:
            print("{}, ".format(valor[:8]))