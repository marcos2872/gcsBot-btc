from src.client import BinanceClient
from src.trading import Trader
from src.ml import train_model, predict_action
from src.logger import logger
from src.utils import salvar_saldo_atual, recuperar_saldo_atual
import numpy as np

if __name__ == "__main__":
    # Inicializa o cliente Binance
    client = BinanceClient()

    # Recuperar saldo da carteira, se o arquivo existir
    saldo_usdt, saldo_btc = recuperar_saldo_atual()

    # Verifique se o saldo inicial é zero e defina um valor inicial caso necessário
    if saldo_usdt == 0:
        saldo_usdt = 1000  # Defina um valor inicial de USDT, por exemplo, 1000 USDT

    # Inicializa o Trader com o saldo recuperado
    trader = Trader(saldo_usdt, saldo_btc)

    # Coletar e salvar os dados históricos de 1000 registros de 1h para BTCUSDT
    client.save_historical_data('BTCUSDT', '1h', 1000)

    # Treina o modelo de aprendizado
    train_model()

    # Carrega o modelo treinado
    from stable_baselines3 import PPO
    model = PPO.load("trading_model")

    # Simula o ambiente de trading
    current_price = client.get_current_price('BTCUSDT')
    state = np.array([saldo_usdt])  # Usa o saldo USDT como parte do estado inicial
    action = predict_action(model, state)

    # Realiza a ação (compra ou venda)
    if action == 1:  # Comprar
        trader.comprar(current_price)
    elif action == 2:  # Vender
        trader.vender(current_price)

    # Salvar o saldo atual após a transação
    saldo_usdt = client.get_balance('USDT')
    saldo_btc = client.get_balance('BTC')
    salvar_saldo_atual(saldo_usdt, saldo_btc)

    # Log do desempenho
    logger.info(f"Desempenho atual: Saldo USDT = {saldo_usdt}, Saldo BTC = {saldo_btc}")
