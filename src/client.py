import csv
import os
import time
from binance.client import Client
from .config import API_KEY, API_SECRET, TESTNET_API_KEY, TESTNET_API_SECRET, USE_TESTNET
from binance.exceptions import BinanceAPIException

class BinanceClient:
    def __init__(self):
        # Usar Testnet ou Rede Real baseado na variável USE_TESTNET
        if USE_TESTNET:
            self.client = Client(TESTNET_API_KEY, TESTNET_API_SECRET, {"timeout": 60})
            # Modificando a URL base para a Testnet
            self.client.session.base_url = "https://testnet.binance.vision/api"
            print("Conectando à Binance Testnet")
        else:
            self.client = Client(API_KEY, API_SECRET, {"timeout": 60})
            print("Conectando à Binance Rede Real")

        try:
            # Acessando o cliente corretamente
            self.client.ping()
            print("Conexão bem-sucedida com a API da Binance!")
        except Exception as e:
            print(f"Erro ao conectar à API da Binance: {e}")

    def get_local_timestamp(self):
        """Retorna o timestamp local ajustado para 1 minuto atrás"""
        current_timestamp = int(time.time() * 1000)  # Timestamp local em milissegundos
        adjusted_timestamp = current_timestamp - 60 * 1000  # Ajusta para 1 minuto atrás
        return adjusted_timestamp

    def place_order(self, symbol, side, quantity):
        """Faz uma ordem de compra/venda com timestamp ajustado"""
        try:
            # Usa o timestamp local ajustado para 1 minuto atrás
            timestamp = self.get_local_timestamp()  # Ajuste o timestamp para 1 minuto atrás
            
            params = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'timestamp': timestamp
            }
            if side == 'buy':
                return self.client.order_market_buy(**params)
            elif side == 'sell':
                return self.client.order_market_sell(**params)
        except BinanceAPIException as e:
            print(f"Erro ao fazer o pedido: {e}")

    def get_current_price(self, symbol):
        """Retorna o preço atual do ativo"""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])

    def get_balance(self, asset='USDT'):
        """Retorna o saldo disponível do ativo"""
        balance = self.client.get_asset_balance(asset=asset)
        return float(balance['free'])

    def get_historical_data(self, symbol, interval='1h', limit=1000):
        """Obter dados históricos para treinar o modelo de ML"""
        # Aguardar antes de fazer a próxima requisição
        time.sleep(1)  # Aguarda 1 segundo entre as requisições
        
        # Coleta os dados históricos
        klines = self.client.get_historical_klines(symbol, interval, f"{limit} hours ago UTC")

        # Prepara os dados para salvar no CSV
        dados = []
        for kline in klines:
            timestamp = kline[0]
            open_price = float(kline[1])
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            volume = float(kline[5])

            dados.append([timestamp, open_price, high, low, close, volume])
        
        return dados

    def save_historical_data(self, symbol, interval='1h', limit=1000):
        """Salva os dados históricos no arquivo CSV"""
        dados = self.get_historical_data(symbol, interval, limit)

        # Verificar se o diretório 'data' existe, caso contrário, cria
        if not os.path.exists('data'):
            os.makedirs('data')

        # Verificar se o arquivo já existe e adicionar o cabeçalho apenas na primeira vez
        file_exists = os.path.exists('data/treinamento.csv')

        with open('data/treinamento.csv', mode='a', newline='') as file:
            writer = csv.writer(file)

            # Adicionar o cabeçalho se for o primeiro registro
            if not file_exists:
                writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

            # Escrever os dados históricos
            writer.writerows(dados)
