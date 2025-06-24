from .client import BinanceClient
from .transactions import registrar_transacao
from .config import SYMBOL, RISK_TOLERANCE, PROFIT_TARGET

class Trader:
    def __init__(self, saldo_usdt, saldo_btc):
        self.client = BinanceClient()
        self.capital_usdt = saldo_usdt
        self.capital_btc = saldo_btc

    def calcular_quantidade(self, preco, capital):
        """Calcula a quantidade de BTC a ser comprada/vendida com base no preço e capital disponível"""
        return capital / preco

    def comprar(self, preco_atual):
        """Decide se deve comprar baseado no preço e risco tolerado"""
        if self.capital_usdt > 0:
            quantidade = self.calcular_quantidade(preco_atual, self.capital_usdt)
            self.client.place_order(SYMBOL, 'buy', quantidade)
            registrar_transacao('compra', SYMBOL, quantidade, self.capital_btc, self.capital_usdt)

    def vender(self, preco_atual):
        """Decide se deve vender baseado no ganho desejado"""
        if self.capital_btc > 0:
            quantidade = self.capital_btc  # Vende todo o BTC
            self.client.place_order(SYMBOL, 'sell', quantidade)
            registrar_transacao('venda', SYMBOL, quantidade, self.capital_btc, self.capital_usdt)
