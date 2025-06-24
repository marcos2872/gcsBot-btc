import csv
import datetime
import os
from .client import BinanceClient

def registrar_transacao(tipo, simbolo, quantidade, saldo_btc, saldo_usdt):
    # Verificar se o diretório 'data' existe, caso contrário, cria
    if not os.path.exists('data'):
        os.makedirs('data')

    # Obter o preço atual do BTC
    client = BinanceClient()
    preco_atual = client.get_current_price(simbolo)

    # Verificar se o arquivo já existe e adicionar o cabeçalho apenas na primeira vez
    file_exists = os.path.exists('data/transacoes.csv')

    with open('data/transacoes.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Adicionar o cabeçalho se for o primeiro registro
        if not file_exists:
            writer.writerow(['Data e Hora', 'Tipo de Transação', 'Par de Moeda', 'Preço do BTC', 'Quantidade', 'Saldo BTC', 'Saldo USDT'])

        # Registrar a transação com o preço atual
        writer.writerow([datetime.datetime.now(), tipo, simbolo, preco_atual, quantidade, saldo_btc, saldo_usdt])

