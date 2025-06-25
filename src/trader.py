# src/trader.py
import time
import pandas as pd
import csv
import os
import json
from src.client import BinanceClient
from src.ml_trading import MLTrader
from src.logger import logger
from src.config import (
    SYMBOL,
    SIMULATION_MODE,
    SIMULATION_INITIAL_USDT,
    SIMULATION_INITIAL_BTC,
    SIMULATION_TRADE_RATIO
)

class TradingBot:
    # A função __init__ e outras funções de estado (load, save, log)
    # permanecem as mesmas da versão anterior.
    def __init__(self):
        self.symbol = SYMBOL
        self.client = BinanceClient()
        self.ml_model = MLTrader()
        
        self.state_filepath = 'data/simulation_state.json'
        
        self.simulation_mode = SIMULATION_MODE
        self.usdt_balance = SIMULATION_INITIAL_USDT
        self.btc_balance = SIMULATION_INITIAL_BTC
        self.trade_ratio = SIMULATION_TRADE_RATIO
        self.last_buy_price = 0.0

        self.dynamic_profit_target = 0.0
        self.dynamic_stop_loss = 0.0
        
        self.load_state()

        self.trades_log_file = 'data/trades.csv'
        self._initialize_trade_log()

    def load_state(self):
        if not self.simulation_mode: return
        try:
            if os.path.exists(self.state_filepath):
                with open(self.state_filepath, 'r') as f:
                    state = json.load(f)
                    self.usdt_balance = state.get('usdt_balance', self.usdt_balance)
                    self.btc_balance = state.get('btc_balance', self.btc_balance)
                    self.last_buy_price = state.get('last_buy_price', self.last_buy_price)
                    self.dynamic_profit_target = state.get('dynamic_profit_target', 0.0)
                    self.dynamic_stop_loss = state.get('dynamic_stop_loss', 0.0)
                    logger.info(f"✅ Estado da simulação carregado: USDT: {self.usdt_balance:.4f}, BTC: {self.btc_balance:.8f}")
            else:
                logger.info("Nenhum estado de simulação guardado encontrado. A começar com valores iniciais.")
        except Exception as e:
            logger.error(f"Erro ao carregar o estado da simulação: {e}. A usar valores padrão.")

    def save_state(self):
        if not self.simulation_mode: return
        state = {
            'usdt_balance': self.usdt_balance,
            'btc_balance': self.btc_balance,
            'last_buy_price': self.last_buy_price,
            'dynamic_profit_target': self.dynamic_profit_target,
            'dynamic_stop_loss': self.dynamic_stop_loss
        }
        try:
            with open(self.state_filepath, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Erro ao guardar o estado da simulação: {e}")

    def _initialize_trade_log(self):
        if not os.path.exists(self.trades_log_file):
            with open(self.trades_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'type', 'reason', 'price', 'quantity_btc', 'value_usdt', 'portfolio_value_usdt', 'profit_target', 'stop_loss_target'])

    def _log_trade(self, trade_type, reason, price, btc_qty, usdt_value):
        portfolio_value = self.usdt_balance + (self.btc_balance * price)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.trades_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, trade_type, reason, f"{price:,.4f}", f"{btc_qty:.8f}", f"{usdt_value:,.4f}", f"{portfolio_value:,.4f}", f"{self.dynamic_profit_target:.4f}", f"{self.dynamic_stop_loss:.4f}"])


    def initialize(self):
        logger.info("🚀 A inicializar o Trading Bot...")
        if self.simulation_mode:
            logger.info("--- MODO DE SIMULAÇÃO ATIVADO ---")
            logger.info(f"💰 Saldo Atual da Carteira - USDT: {self.usdt_balance:.4f}, BTC: {self.btc_balance:.8f}")
        else:
            self.usdt_balance = self.client.get_balance('USDT')
            self.btc_balance = self.client.get_balance('BTC')

        logger.info("📊 A atualizar a base de dados históricos...")
        self.client.update_historical_data(self.symbol, '1h', initial_limit=300)
        
        if self.last_buy_price == 0.0 and self.btc_balance > 0:
            self.last_buy_price = self.client.get_current_price(self.symbol)
            logger.info(f"Preço de referência (última compra) não encontrado. Definido para o preço atual: ${self.last_buy_price:,.2f}")

        logger.info("🧠 A carregar/treinar o modelo ML...")
        try:
            historical_data = pd.read_csv('data/historical_data.csv')
            if not self.ml_model.load_model():
                logger.info("Modelo não encontrado. A treinar um novo modelo...")
                self.ml_model.train(historical_data)
        except FileNotFoundError:
            logger.error("Falha crítica: ficheiro de dados históricos não encontrado.")
            raise
        
        logger.info("✅ Bot inicializado com sucesso!")

    # --- LÓGICA DE DECISÃO COM TRAVA DE SEGURANÇA ---
    def execute_action(self, current_price, features):
        final_decision = 'HOLD'
        decision_reason = "Aguardando condições ideais."

        if self.btc_balance > 0.00001:
            if current_price >= self.dynamic_profit_target:
                final_decision = 'SELL'
                decision_reason = "Take-Profit Dinâmico Atingido"
            
            elif current_price <= self.dynamic_stop_loss:
                final_decision = 'SELL'
                decision_reason = "Stop-Loss Dinâmico Acionado"
            
            else:
                 profit_pct = (current_price - self.last_buy_price) / self.last_buy_price * 100
                 decision_reason = f"Posição Aberta. Lucro: {profit_pct:.3f}%. Alvo: ${self.dynamic_profit_target:,.2f}"
        else:
            final_decision = 'BUY'
            decision_reason = "Iniciando nova operação."

        logger.info(f"🎯 Decisão Final: {final_decision} | Razão: {decision_reason}")

        if final_decision == 'BUY' and self.usdt_balance > 10:
            usdt_to_spend = self.usdt_balance * self.trade_ratio
            if usdt_to_spend < 10: usdt_to_spend = 10

            btc_bought = usdt_to_spend / current_price
            self.usdt_balance -= usdt_to_spend
            self.btc_balance += btc_bought
            self.last_buy_price = current_price

            try:
                targets = self.ml_model.predict(features)
                if targets:
                    predicted_profit_target = targets['profit_target_price']
                    predicted_stop_loss = targets['stop_loss_price']

                    # --- SANITY CHECK (A TRAVA DE SEGURANÇA) ---
                    # 1. O alvo de lucro DEVE ser maior que o preço de compra.
                    if predicted_profit_target > self.last_buy_price:
                        self.dynamic_profit_target = predicted_profit_target
                    else:
                        # Fallback: Se o ML errar, define uma meta de lucro mínima de 0.1%
                        self.dynamic_profit_target = self.last_buy_price * 1.001
                        logger.warning(f"ML previu lucro inválido (${predicted_profit_target:,.2f}). Usando meta de fallback: ${self.dynamic_profit_target:,.2f}")

                    # 2. O alvo de stop DEVE ser menor que o preço de compra.
                    if predicted_stop_loss < self.last_buy_price:
                        self.dynamic_stop_loss = predicted_stop_loss
                    else:
                        # Fallback: Se o ML errar, define um stop mínimo de 0.2%
                        self.dynamic_stop_loss = self.last_buy_price * 0.998
                        logger.warning(f"ML previu stop inválido (${predicted_stop_loss:,.2f}). Usando meta de fallback: ${self.dynamic_stop_loss:,.2f}")
                else:
                    raise ValueError("Previsão do ML retornou None")
            except Exception as e:
                logger.error(f"Erro ao prever metas dinâmicas: {e}. Usando metas de fallback.")
                self.dynamic_profit_target = current_price * 1.001
                self.dynamic_stop_loss = current_price * 0.998
            
            logger.info(f"🧠 Metas definidas -> Lucro: ${self.dynamic_profit_target:,.2f} | Stop: ${self.dynamic_stop_loss:,.2f}")
            self._log_trade('BUY', decision_reason, current_price, btc_bought, usdt_to_spend)
            self.save_state()

        elif final_decision == 'SELL' and self.btc_balance > 0.00001:
            btc_to_sell = self.btc_balance 
            usdt_gained = btc_to_sell * current_price
            profit = usdt_gained - (btc_to_sell * self.last_buy_price)
            
            self.btc_balance = 0.0
            self.usdt_balance += usdt_gained
            self.last_buy_price = 0.0
            self.dynamic_profit_target = 0.0
            self.dynamic_stop_loss = 0.0
            
            log_profit = f"Lucro: ${profit:,.4f}" if profit >= 0 else f"Prejuízo: ${-profit:,.4f}"
            logger.info(f"📉 VENDA SIMULADA: {btc_to_sell:.8f} BTC por ${usdt_gained:,.4f}. {log_profit}")
            self._log_trade('SELL', decision_reason, current_price, btc_to_sell, usdt_gained)
            self.save_state()

    def run_trading_cycle(self):
        try:
            current_price = self.client.get_current_price(self.symbol)
            logger.info(f"💹 Preço atual BTC: ${current_price:,.2f}")
            
            historical_data = pd.read_csv('data/historical_data.csv')
            features = self.ml_model.prepare_data_for_prediction(historical_data, current_price)
            
            self.execute_action(current_price, features)
            
            portfolio_value_usdt = self.usdt_balance + (self.btc_balance * current_price)
            logger.info(f"Portfólio: ${portfolio_value_usdt:,.4f} | USDT: {self.usdt_balance:,.4f} | BTC: {self.btc_balance:.8f}")

        except Exception as e:
            logger.error(f"Ocorreu um erro no ciclo de trading: {e}")

    def run(self, cycles=100, retrain_every=20):
        self.initialize()
        for i in range(cycles):
            logger.info(f"--- 🔄 Ciclo {i + 1}/{cycles} ---")
            self.run_trading_cycle()
            
            if (i + 1) % retrain_every == 0 and i > 0:
                logger.info("🔔 HORA DE APRENDER! A retreinar os modelos dinâmicos...")
                self.client.update_historical_data(self.symbol, '1h')
                
                try:
                    logger.info("🧠 A retreinar modelos ML com a base de dados expandida...")
                    historical_data = pd.read_csv('data/historical_data.csv')
                    self.ml_model.train(historical_data)
                except Exception as e:
                    logger.error(f"Falha ao retreinar os modelos: {e}")

            if i < cycles - 1:
                time.sleep(5) 
        logger.info("🏁 Execução finalizada!")