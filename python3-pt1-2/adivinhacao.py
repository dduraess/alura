import random

def jogar():
    print("*********************************")
    print("Bem vindo ao jogo de Adivinhação!")
    print("*********************************")

    numero_secreto = random.randrange(1, 101)
    total_tentativas = 0
    pontos = 1000

    print("Qual nível de dificuldade?")
    print("(1) Fácil (2) Médio (3) Difícil")

    nivel = int(input("Define o nível: "))

    if nivel == 1:
        total_tentativas = 20
    elif nivel == 2:
        total_tentativas = 10
    else:
        total_tentativas = 5

    for rodada in range(1, total_tentativas + 1):
        print("Tentativa {} de {}".format(rodada, total_tentativas))
        chute = int(input("Digite um número entre 1 e 100: "))
        print("Você digitou ", chute)

        if chute < 1 or chute > 100:
            print("Você deve digitar um número entre 1 e 100!")
            continue

        acertou = chute == numero_secreto
        maior = chute > numero_secreto
        menor = chute < numero_secreto

        if (acertou):
            print("Você acertou e fez {} pontos!".format(pontos))
            break
        else:
            pontos_perdidos = abs(numero_secreto - chute)
            pontos = pontos - pontos_perdidos
            if (maior):
                print("Você errou! O chute foi maior que o número secreto!")
                if rodada == total_tentativas: print(
                    "O número secreto era {}. Você fez {} pontos".format(numero_secreto, pontos))
            elif (menor):
                print("Você errou! O chute foi menor que o número secreto!")
                if rodada == total_tentativas: print(
                    "O número secreto era {}. Você fez {} pontos".format(numero_secreto, pontos))

    print("Fim do jogo")

if (__name__ == "__main__"):
    jogar()
