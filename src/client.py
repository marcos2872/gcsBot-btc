# src/client.py
import csv
import os
import time
from binance.client import Client
from .config import API_KEY, API_SECRET, TESTNET_API_KEY, TESTNET_API_SECRET, USE_TESTNET
from binance.exceptions import BinanceAPIException
from .logger import logger

class BinanceClient:
    def __init__(self):
        if USE_TESTNET:
            if not TESTNET_API_KEY or not TESTNET_API_SECRET:
                raise ValueError("Chaves da Testnet não configuradas no arquivo .env")
            
            self.client = Client(
                TESTNET_API_KEY, 
                TESTNET_API_SECRET,
                testnet=True  # Parâmetro correto para testnet
            )
            logger.info("✅ Conectado à Binance Testnet")
        else:
            if not API_KEY or not API_SECRET:
                raise ValueError("Chaves da API real não configuradas no arquivo .env")
            
            self.client = Client(API_KEY, API_SECRET)
            logger.info("✅ Conectado à Binance Rede Real")

        # Testar conexão
        try:
            self.client.ping()
            logger.info("✅ Ping bem-sucedido!")
            
            # Testar acesso à conta
            account_info = self.client.get_account()
            logger.info(f"✅ Acesso à conta confirmado - Tipo: {account_info.get('accountType', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            raise

    def get_current_price(self, symbol):
        """Retorna o preço atual do ativo"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Erro ao obter preço de {symbol}: {e}")
            raise

    def get_balance(self, asset='USDT'):
        """Retorna o saldo disponível do ativo"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            logger.error(f"Erro ao obter saldo de {asset}: {e}")
            raise

    def place_order(self, symbol, side, quantity):
        """Executa ordem de mercado (apenas para testnet)"""
        try:
            if not USE_TESTNET:
                logger.warning("⚠️ Ordem bloqueada - não está em modo testnet")
                return None
                
            # Formatar quantidade para 8 casas decimais
            quantity = round(float(quantity), 8)
            
            if side.upper() == 'BUY':
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
            elif side.upper() == 'SELL':
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
            
            logger.info(f"✅ Ordem executada: {side} {quantity} {symbol}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"❌ Erro na ordem: {e}")
            return None

    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Obter dados históricos"""
        try:
            klines = self.client.get_historical_klines(
                symbol, interval, f"{limit} hours ago UTC"
            )
            
            dados = []
            for kline in klines:
                dados.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            return dados
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos: {e}")
            return []

    def save_historical_data(self, symbol, interval='1h', limit=100):
        """Salva dados históricos em CSV"""
        dados = self.get_historical_data(symbol, interval, limit)
        
        if not dados:
            logger.warning("Nenhum dado histórico obtido")
            return
        
        os.makedirs('data', exist_ok=True)
        
        with open('data/historical_data.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            writer.writeheader()
            writer.writerows(dados)
        
        logger.info(f"✅ {len(dados)} registros salvos em data/historical_data.csv")