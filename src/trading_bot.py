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
# ### CORREÇÃO ###: O import do ModelTrainer precisa do caminho 'src'
from src.model_trainer import ModelTrainer
from src.logger import logger
from src.config import (
    API_KEY, API_SECRET, USE_TESTNET, SYMBOL,  
    MODEL_FILE, SCALER_FILE, TRADES_LOG_FILE, BOT_STATE_FILE,
    TRADE_AMOUNT_USDT, STRATEGY_PARAMS_FILE 
)
from src.data_manager import DataManager 

class TradingBot:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET, tld='com', testnet=USE_TESTNET)
        self.model = None
        self.scaler = None
        self.data_manager = DataManager()
        self.usdt_balance = 0.0
        self.in_position = False
        self.buy_price = 0.0
        self.buy_quantity = 0.0
        self.strategy_params = {
            'profit_threshold': 0.02,
            'stop_loss_threshold': 0.01,
            'prediction_confidence': 0.75
        }
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    def graceful_shutdown(self, signum, frame):
        logger.info("="*50)
        logger.info("PARADA SOLICITADA. ENCERRANDO O BOT DE FORMA SEGURA...")
        logger.info("O estado atual já está salvo. O bot poderá continuar de onde parou.")
        logger.info("="*50)
        sys.exit(0)

    def _initialize_trade_log(self):
        header = ['timestamp', 'type', 'price', 'quantity_btc', 'value_usdt', 'reason', 'pnl_usdt', 'pnl_percent']
        if not os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(header)

    # ### CORREÇÃO ###: Adicionamos pnl_usdt e pnl_percent à assinatura da função com valores padrão None.
    def _log_trade(self, trade_type, price, quantity_btc, reason, pnl_usdt=None, pnl_percent=None):
        with open(TRADES_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Agora as variáveis pnl_usdt e pnl_percent existem no escopo da função.
            pnl_usdt_str = f"{pnl_usdt:.2f}" if pnl_usdt is not None else ""
            pnl_percent_str = f"{pnl_percent:.4f}" if pnl_percent is not None else ""
            
            writer.writerow([
                pd.Timestamp.now(tz='UTC'), trade_type, price, quantity_btc, 
                price * quantity_btc, reason, pnl_usdt_str, pnl_percent_str
            ])

    def _save_state(self):
        state = {'in_position': self.in_position, 'buy_price': self.buy_price, 'buy_quantity': self.buy_quantity}
        with open(BOT_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Estado do bot salvo: {state}")

    def _load_state(self):
        if os.path.exists(BOT_STATE_FILE):
            try:
                with open(BOT_STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.in_position = state.get('in_position', False)
                    self.buy_price = state.get('buy_price', 0.0)
                    self.buy_quantity = state.get('buy_quantity', 0.0)
                    logger.info(f"Estado do bot carregado: {state}")
            except (json.JSONDecodeError, TypeError):
                logger.error("Erro ao ler o arquivo de estado ou arquivo corrompido. Começando do zero.")
                self.in_position, self.buy_price, self.buy_quantity = False, 0.0, 0.0
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

    def _load_strategy_params(self):
        try:
            with open(STRATEGY_PARAMS_FILE, 'r') as f:
                self.strategy_params = json.load(f)
            logger.info(f"✅ Parâmetros da estratégia carregados: {self.strategy_params}")
        except FileNotFoundError:
            logger.warning(f"Arquivo de parâmetros '{STRATEGY_PARAMS_FILE}' não encontrado. Usando valores padrão.")
        except Exception as e:
            logger.error(f"Erro ao carregar parâmetros da estratégia: {e}. Usando valores padrão.")

    def get_real_balance(self):
        try:
            balance = self.client.get_asset_balance(asset='USDT')
            return float(balance['free']) if balance else 0.0
        except BinanceAPIException as e:
            logger.error(f"Erro ao buscar saldo na Binance: {e}")
            return 0.0

    def _prepare_prediction_data(self):
        try:
            end_date = pd.Timestamp.now(tz='UTC')
            start_date = end_date - pd.Timedelta(days=2)
            
            df_combined = self.data_manager.get_historical_data_by_batch(SYMBOL, '1m', start_date, end_date)
            
            if df_combined.empty:
                logger.warning("Não foi possível obter dados históricos para predição.")
                return pd.DataFrame(), None

            trainer_for_features = ModelTrainer()
            df_features = trainer_for_features._prepare_features(df_combined.copy())

            if df_features.empty:
                return pd.DataFrame(), None

            # Garante que as colunas estarão na ordem correta para a predição
            last_feature_row = df_features.reindex(columns=trainer_for_features.feature_names, fill_value=0).tail(1)
            current_price = df_combined['close'].iloc[-1]
            return last_feature_row, current_price
        except Exception as e:
            logger.error(f"Erro ao preparar dados para predição: {e}", exc_info=True)
            return pd.DataFrame(), None

    def execute_trade(self, side, quantity):
        try:
            formatted_quantity = "{:.6f}".format(quantity)
            logger.info(f"Enviando ordem: Lado={side}, Quantidade={formatted_quantity}")
            
            # Para testnet ou produção, a chamada é a mesma
            order = self.client.create_order(symbol=SYMBOL, side=side, type=Client.ORDER_TYPE_MARKET, quantity=formatted_quantity)
            logger.info(f"ORDEM ENVIADA: {order}")
            
            fills = order.get('fills', [])
            if fills:
                avg_price = sum(float(f['price']) * float(f['qty']) for f in fills) / sum(float(f['qty']) for f in fills)
                executed_qty = sum(float(f['qty']) for f in fills)
                logger.info(f"Ordem preenchida. Preço Médio: {avg_price:.2f}, Quantidade: {executed_qty}")
                return order, avg_price, executed_qty
            
            # Fallback caso 'fills' esteja vazio
            return order, float(order.get('price', 0)), float(order.get('executedQty', 0))
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
        self._load_strategy_params()
        
        self.usdt_balance = self.get_real_balance()
        logger.info(f"Saldo inicial disponível: {self.usdt_balance:.2f} USDT")

        while True:
            try:
                features_df, current_price = self._prepare_prediction_data()
                if features_df.empty or current_price is None:
                    logger.warning("Não foi possível gerar features para predição. Aguardando próximo ciclo.")
                    time.sleep(60)
                    continue

                if self.in_position:
                    profit_pct = (current_price / self.buy_price) - 1 if self.buy_price > 0 else 0
                    
                    sell_reason = None
                    if profit_pct >= self.strategy_params['profit_threshold']:
                        sell_reason = f"Take-Profit de {profit_pct:.2%}"
                    elif profit_pct <= -self.strategy_params['stop_loss_threshold']:
                        sell_reason = f"Stop-Loss de {profit_pct:.2%}"
                    
                    if sell_reason:
                        logger.info(f"🎯 {sell_reason}. Vendendo {self.buy_quantity} BTC...")
                        # ### CORREÇÃO ###: Capturamos o `executed_qty` retornado pela função.
                        order, sell_price, executed_qty = self.execute_trade(side=Client.SIDE_SELL, quantity=self.buy_quantity)
                        
                        if order and executed_qty > 0:
                            pnl_usdt = (sell_price - self.buy_price) * executed_qty
                            pnl_percent = (sell_price / self.buy_price) - 1
                            
                            self.in_position = False
                            self._log_trade("SELL", sell_price, executed_qty, sell_reason, pnl_usdt, pnl_percent)
                            self._save_state()
                            self.usdt_balance = self.get_real_balance()
                            logger.info(f"Venda concluída. P&L: ${pnl_usdt:.2f} ({pnl_percent:.2%}). Novo saldo: {self.usdt_balance:.2f} USDT")
                else: # Se não está em posição, avalia uma compra
                    scaled_features = self.scaler.transform(features_df)
                    prediction_proba = self.model.predict_proba(scaled_features)[0]
                    buy_confidence = prediction_proba[1]

                    logger.info(f"Preço: ${current_price:,.2f} | Confiança de Compra do Modelo: {buy_confidence:.2%}")
                    
                    features_for_log = features_df.iloc[0].to_dict()
                    logger.debug(f"Valores das features para esta predição: {json.dumps(features_for_log)}")

                    if buy_confidence > self.strategy_params['prediction_confidence']:
                        if self.usdt_balance >= TRADE_AMOUNT_USDT:
                            logger.info(f"🧠 Confiança ({buy_confidence:.2%}) > limite ({self.strategy_params['prediction_confidence']:.2%}). EXECUTANDO COMPRA...")
                            quantity_to_buy = TRADE_AMOUNT_USDT / current_price
                            
                            order, buy_price_filled, executed_qty = self.execute_trade(side=Client.SIDE_BUY, quantity=quantity_to_buy)
                            
                            if order and executed_qty > 0:
                                self.in_position = True
                                self.buy_price = buy_price_filled
                                self.buy_quantity = executed_qty
                                # Para compra, não há P&L, então os valores padrão (None) são usados
                                self._log_trade("BUY", self.buy_price, self.buy_quantity, f"Sinal do ML ({buy_confidence:.2%})")
                                self._save_state()
                                self.usdt_balance = self.get_real_balance()
                                logger.info(f"Compra concluída. Novo saldo: {self.usdt_balance:.2f} USDT")
                        else:
                            logger.warning(f"Sinal de compra, mas saldo USDT ({self.usdt_balance:.2f}) é insuficiente para uma ordem de {TRADE_AMOUNT_USDT} USDT.")
                
                if self.in_position:
                    pnl = (current_price - self.buy_price) * self.buy_quantity
                    logger.info(f"EM POSIÇÃO | Preço Atual: ${current_price:,.2f} | Preço Compra: ${self.buy_price:,.2f} | P&L Aberto: ${pnl:,.2f}")

                logger.info("--- Ciclo concluído, aguardando 60 segundos ---")
                time.sleep(60)

            except Exception as e:
                logger.error(f"Erro inesperado no loop principal: {e}", exc_info=True)
                time.sleep(60)