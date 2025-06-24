# src/main.py

import logging
from src.logger import setup_logger
from src.trading_logic.order_execution import place_order
from src.trading_logic.strategies import rsi_strategy, bollinger_bands_strategy
from src.trading_logic.risk_management import RiskManager
from src.data_fetcher import get_last_price, get_balance

# Configurando o logger
setup_logger()

# Inicializando o Gerenciador de Risco
risk_manager = RiskManager(risk_percentage=0.01, max_loss=0.05)

def run_bot():
    prices = []  # Preços históricos
    buy_price = None
    amount = 0.001  # Exemplo de quantidade de BTC

    while True:
        last_price = get_last_price()
        prices.append(last_price)
        usdt_balance, btc_balance = get_balance()

        logging.info(f"Preço atual do BTC: {last_price} | Saldo USDT: {usdt_balance} | Saldo BTC: {btc_balance}")

        # Estratégias Dinâmicas
        action_rsi = rsi_strategy(prices)
        action_bollinger = bollinger_bands_strategy(prices)

        logging.info(f"Ação do RSI: {action_rsi} | Ação das Bandas de Bollinger: {action_bollinger}")

        # Gerenciamento de Risco
        if not buy_price and action_rsi == "buy" and action_bollinger == "buy":
            trade_size = risk_manager.calculate_trade_size(usdt_balance, last_price, stop_loss_percent=2.0)
            logging.info(f"Compra determinada: {trade_size} BTC com base nas estratégias")
            place_order(order_type="buy", amount=trade_size)
            buy_price = last_price
            logging.info(f"Compra realizada por {buy_price}")

        if buy_price:
            # Verificar Take Profit / Stop Loss
            if last_price > buy_price * 1.02:  # Take Profit
                logging.info("Lucro atingido! Vendendo BTC")
                place_order(order_type="sell", amount=btc_balance)
                profit = (last_price - buy_price) * btc_balance
                logging.info(f"Lucro da transação: {profit}")
                buy_price = None
            elif last_price < buy_price * 0.98:  # Stop Loss
                logging.info("Stop Loss atingido! Vendendo BTC")
                place_order(order_type="sell", amount=btc_balance)
                buy_price = None

if __name__ == "__main__":
    logging.info("Iniciando o bot de trading")
    run_bot()
