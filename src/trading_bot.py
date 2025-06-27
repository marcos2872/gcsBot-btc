# src/trading_bot.py

import pandas as pd
import numpy as np
import joblib
import os
import time
import csv
import json
import signal
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.logger import logger
from src.config import (
    API_KEY, API_SECRET, USE_TESTNET, SYMBOL,  
    MODEL_FILE, SCALER_FILE, TRADES_LOG_FILE, BOT_STATE_FILE,
    TRADE_AMOUNT_USDT, PROFIT_THRESHOLD, STOP_LOSS_THRESHOLD
)
from src.model_trainer import ModelTrainer # Import para usar o prepare_features

class TradingBot:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET, tld='com', testnet=USE_TESTNET)
        self.model = None
        self.scaler = None
        self.trainer = ModelTrainer() # Para acesso ao _prepare_features
        self.historical_data = []
        self.usdt_balance = 0.0
        # Estado da posição
        self.in_position = False
        self.buy_price = 0.0
        self.buy_quantity = 0.0
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    def graceful_shutdown(self, signum, frame):
        """Lida com a parada graciosa do bot."""
        logger.info("="*50)
        logger.info("PARADA SOLICITADA. ENCERRANDO O BOT DE FORMA SEGURA...")
        logger.info("O estado atual já está salvo. O bot poderá continuar de onde parou.")
        logger.info("="*50)
        sys.exit(0)

    def _initialize_trade_log(self):
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'type', 'price', 'quantity_btc', 'value_usdt', 'reason'])

    def _log_trade(self, trade_type, price, quantity_btc, reason):
        with open(TRADES_LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([pd.Timestamp.now(tz='UTC'), trade_type, price, quantity_btc, price * quantity_btc, reason])

    def _save_state(self):
        state = {'in_position': self.in_position, 'buy_price': self.buy_price, 'buy_quantity': self.buy_quantity}
        with open(BOT_STATE_FILE, 'w') as f:
            json.dump(state, f)
        logger.info(f"Estado do bot salvo: {state}")

    def _load_state(self):
        if os.path.exists(BOT_STATE_FILE):
            with open(BOT_STATE_FILE, 'r') as f:
                state = json.load(f)
                self.in_position = state.get('in_position', False)
                self.buy_price = state.get('buy_price', 0.0)
                self.buy_quantity = state.get('buy_quantity', 0.0)
                logger.info(f"Estado do bot carregado: {state}")
        else:
            logger.info("Nenhum arquivo de estado encontrado. Iniciando do zero.")

    def load_model(self):
        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            logger.info("✅ Modelo e normalizador carregados com sucesso.")
            return True
        except FileNotFoundError:
            logger.error("ERRO: Arquivos de modelo/normalizador não encontrados. Execute o modo 'optimize' primeiro.")
            return False

    def get_real_balance(self):
        try:
            balance_usdt = float(self.client.get_asset_balance(asset='USDT')['free'])
            return balance_usdt
        except BinanceAPIException as e:
            logger.error(f"Erro ao buscar saldo na Binance: {e}")
            return 0.0

    def _prepare_prediction_data(self):
        # Usamos pelo menos 50 velas para garantir que as features (SMA 25, etc.) possam ser calculadas.
        klines = self.client.get_klines(symbol=SYMBOL, interval='1m', limit=50)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'nt', 'tbbav', 'tbqav', 'ignore'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Usa o mesmo preparador de features do treino para consistência
        df_features = self.trainer._prepare_features(df.copy())
        if df_features.empty:
            return pd.DataFrame(), None

        last_feature_row = df_features[self.trainer.feature_names].tail(1)
        current_price = df['close'].iloc[-1]
        return last_feature_row, current_price

    def execute_trade(self, side, quantity):
        try:
            order = self.client.create_order(symbol=SYMBOL, side=side, type=Client.ORDER_TYPE_MARKET, quantity=quantity)
            logger.info(f"ORDEM ENVIADA: {order}")
            
            fills = order.get('fills', [])
            if fills:
                avg_price = sum(float(f['price']) * float(f['qty']) for f in fills) / sum(float(f['qty']) for f in fills)
                executed_qty = sum(float(f['qty']) for f in fills)
                logger.info(f"Ordem preenchida. Preço Médio: {avg_price:.2f}, Quantidade: {executed_qty}")
                return order, avg_price, executed_qty
            return order, None, None
        except BinanceAPIException as e:
            logger.error(f"ERRO DE API AO EXECUTAR ORDEM: {e}")
            return None, None, None
        except Exception as e:
            logger.error(f"ERRO INESPERADO AO EXECUTAR ORDEM: {e}")
            return None, None, None

    def run(self):
        if not self.load_model(): return
        
        self._initialize_trade_log()
        self._load_state()
        
        self.usdt_balance = self.get_real_balance()
        logger.info(f"Saldo disponível: {self.usdt_balance:.2f} USDT")

        while True:
            try:
                features_df, current_price = self._prepare_prediction_data()
                if features_df.empty:
                    logger.warning("Não foi possível gerar features para predição. Aguardando próximo ciclo.")
                    time.sleep(60)
                    continue

                # 1. Lógica de Venda (Stop-Loss ou Take-Profit)
                if self.in_position:
                    profit_pct = (current_price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0
                    
                    sell_reason = None
                    if profit_pct >= PROFIT_THRESHOLD: sell_reason = f"Take-Profit de {profit_pct:.2%}"
                    elif profit_pct <= -STOP_LOSS_THRESHOLD: sell_reason = f"Stop-Loss de {profit_pct:.2%}"
                    
                    if sell_reason:
                        logger.info(f"🎯 {sell_reason}. Vendendo...")
                        order, sell_price, _ = self.execute_trade(side=Client.SIDE_SELL, quantity=self.buy_quantity)
                        if order:
                            self.in_position = False
                            self._log_trade("SELL", sell_price or current_price, self.buy_quantity, sell_reason)
                            self._save_state()
                            self.usdt_balance = self.get_real_balance()
                
                # 2. Lógica de Compra
                if not self.in_position:
                    scaled_features = self.scaler.transform(features_df)
                    prediction = self.model.predict(scaled_features)[0]

                    if prediction == 1: # Sinal de Compra
                        if self.usdt_balance >= TRADE_AMOUNT_USDT:
                            logger.info(f"🧠 Modelo previu COMPRA. Executando...")
                            quantity_to_buy = round(TRADE_AMOUNT_USDT / current_price, 6)
                            order, buy_price, executed_qty = self.execute_trade(side=Client.SIDE_BUY, quantity=quantity_to_buy)
                            
                            if order:
                                self.in_position = True
                                self.buy_price = buy_price or current_price
                                self.buy_quantity = executed_qty or quantity_to_buy
                                self._log_trade("BUY", self.buy_price, self.buy_quantity, "Sinal do ML")
                                self._save_state()
                                self.usdt_balance = self.get_real_balance()
                        else:
                            logger.warning(f"Sinal de compra, mas saldo USDT ({self.usdt_balance:.2f}) insuficiente.")
                    else:
                        logger.info(f"Preço: ${current_price:,.2f} | Modelo previu {'HOLD' if prediction == 0 else 'SELL'}. Nenhuma ação de compra.")
                
                # Log de status
                if self.in_position:
                    pnl = (current_price - self.buy_price) * self.buy_quantity
                    logger.info(f"Em Posição | Preço Atual: ${current_price:,.2f} | Preço Compra: ${self.buy_price:,.2f} | P&L: ${pnl:,.2f}")

                time.sleep(60)

            except Exception as e:
                logger.error(f"Erro inesperado no loop principal: {e}", exc_info=True)
                time.sleep(60)