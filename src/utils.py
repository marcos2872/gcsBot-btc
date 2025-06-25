#src/utils.py

import csv

def salvar_saldo_atual(saldo_usdt, saldo_btc):
    """Salva o saldo atual da carteira em um arquivo CSV"""
    with open('data/saldo_atual.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Saldo USDT', 'Saldo BTC'])
        writer.writerow([saldo_usdt, saldo_btc])

def recuperar_saldo_atual():
    """Recupera o saldo salvo anteriormente da carteira"""
    try:
        with open('data/saldo_atual.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Pular o cabeçalho
            saldo = next(reader)
            return float(saldo[0]), float(saldo[1])  # Retorna (saldo_usdt, saldo_btc)
    except FileNotFoundError:
        return 0.0, 0.0  # Caso o arquivo não exista, retorna saldo inicial de 0
