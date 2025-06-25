# src/trader.py
import time
import pandas as pd
import csv
import os
import json
from src.client import BinanceClient
from src.ml_trading import MLTrader
from src.logger import logger
from src.config import SYMBOL, SIMULATION_MODE, SIMULATION_INITIAL_USDT, SIMULATION_INITIAL_BTC

class TradingBot:
    def __init__(self):
        self.client = BinanceClient()
        self.ml_model = MLTrader()
        
        self.usdt_balance = SIMULATION_INITIAL_USDT
        self.btc_balance = SIMULATION_INITIAL_BTC
        self.average_cost_price = 0.0
        self.last_sell_price = 0.0
        
        # --- PARÂMETROS HIPER-AGRESSIVOS ---
        self.trade_ratio = 0.15 # Aumentado para 15% do saldo por operação
        self.base_profit_target = 0.0012 # Alvo de lucro base de 0.12% (muito sensível)
        self.base_dip_target = 0.0018    # Alvo de queda base para compra de 0.18%
        self.rebuy_dip_target = 0.001    # Recompra com apenas 0.1% de queda após a venda

        self.state_filepath = 'data/simulation_state.json'
        self.trades_log_file = 'data/trades.csv'
        self.load_state()
        self._initialize_trade_log()

    def load_state(self):
        if not os.path.exists(self.state_filepath): return
        with open(self.state_filepath, 'r') as f:
            state = json.load(f)
            self.usdt_balance = state.get('usdt_balance', self.usdt_balance)
            self.btc_balance = state.get('btc_balance', self.btc_balance)
            self.average_cost_price = state.get('average_cost_price', self.average_cost_price)
            self.last_sell_price = state.get('last_sell_price', self.last_sell_price)
            logger.info(f"✅ Estado carregado. Custo Médio: ${self.average_cost_price:,.2f}")

    def save_state(self):
        state = {
            'usdt_balance': self.usdt_balance,
            'btc_balance': self.btc_balance,
            'average_cost_price': self.average_cost_price,
            'last_sell_price': self.last_sell_price
        }
        with open(self.state_filepath, 'w') as f: json.dump(state, f, indent=4)

    def _initialize_trade_log(self):
        if not os.path.exists(self.trades_log_file):
            with open(self.trades_log_file, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'type', 'reason', 'price', 'btc_qty', 'usdt_value', 'avg_cost', 'portfolio_value'])

    def _log_trade(self, t_type, reason, price, btc, usdt):
        portfolio = self.usdt_balance + (self.btc_balance * price)
        with open(self.trades_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), t_type, reason, f"{price:,.2f}", f"{btc:.8f}", f"{usdt:,.2f}", f"{self.average_cost_price:,.2f}", f"{portfolio:,.2f}"])

    def initialize(self):
        logger.info("🚀 A inicializar o Trader Adaptativo de Alta Frequência...")
        self.client.update_historical_data(SYMBOL, '1h', 300)
        try:
            historical_data = pd.read_csv('data/historical_data.csv')
            if not self.ml_model.load_model():
                self.ml_model.train(historical_data)
        except Exception as e:
            logger.error(f"Falha na inicialização do ML: {e}")
            raise
        logger.info("✅ Bot inicializado com sucesso!")
        
    def execute_action(self, current_price, signal_strength):
        # --- LÓGICA DE INTUIÇÃO AMPLIFICADA ---
        aggression_factor = 3.0 # Aumentamos a influência do ML
        sell_pressure = 1 + (signal_strength.get('sell', 0) * aggression_factor)
        buy_pressure = 1 + (signal_strength.get('buy', 0) * aggression_factor)
        
        adjusted_profit_target = self.base_profit_target / sell_pressure
        adjusted_dip_target = self.base_dip_target / buy_pressure
        logger.info(f"Intuição ML -> Compra: {buy_pressure:.2f}x, Venda: {sell_pressure:.2f}x | Metas -> Lucro: {adjusted_profit_target:.3%}, Queda: {adjusted_dip_target:.3%}")

        # --- LÓGICA DE VENDA (Take Profit) ---
        if self.btc_balance > 0.00001 and self.average_cost_price > 0:
            if current_price > self.average_cost_price * (1 + adjusted_profit_target):
                # Vende uma fração da posição
                btc_to_sell = self.btc_balance * 0.5 # Vende 50% para realizar lucro mas manter exposição
                if btc_to_sell * current_price < 10: btc_to_sell = self.btc_balance # Se for pouco, vende tudo
                
                usdt_gained = btc_to_sell * current_price
                
                self.btc_balance -= btc_to_sell
                self.usdt_balance += usdt_gained
                self.last_sell_price = current_price
                
                reason = f"Lucro de {(current_price / self.average_cost_price - 1):.2%}"
                logger.info(f"🎯 VENDA LUCRATIVA: {btc_to_sell:.8f} BTC. Razão: {reason}")
                self._log_trade('SELL', reason, current_price, btc_to_sell, usdt_gained)
                self.save_state()
                # Se ainda sobrou BTC, zera o preço médio para forçar uma reavaliação na próxima compra
                if self.btc_balance < 0.00001: self.average_cost_price = 0
                return

        # --- LÓGICA DE COMPRA AGRESSIVA ---
        buy_opportunity = False
        reason = ""
        # 1. Compra para reduzir o preço médio se o preço cair
        if self.btc_balance > 0 and current_price < self.average_cost_price * (1 - adjusted_dip_target):
             buy_opportunity = True
             reason = "Redução de Custo Médio"
        # 2. Recompra rápido após uma venda se o preço cair um pouco
        elif self.last_sell_price > 0 and current_price < self.last_sell_price * (1 - self.rebuy_dip_target):
             buy_opportunity = True
             reason = "Recompra Estratégica"
        # 3. Entrada inicial se não tivermos BTC
        elif self.btc_balance < 0.00001:
             buy_opportunity = True
             reason = "Entrada Inicial no Mercado"

        if buy_opportunity and self.usdt_balance > 10:
            usdt_to_spend = self.usdt_balance * self.trade_ratio
            btc_bought = usdt_to_spend / current_price
            
            total_cost_before = self.btc_balance * self.average_cost_price if self.btc_balance > 0 else 0
            self.btc_balance += btc_bought
            self.average_cost_price = (total_cost_before + usdt_to_spend) / self.btc_balance
            self.usdt_balance -= usdt_to_spend
            
            logger.info(f"🎯 COMPRA ESTRATÉGICA: {btc_bought:.8f} BTC. Nova Média de Custo: ${self.average_cost_price:,.2f}")
            self._log_trade('BUY', reason, current_price, btc_bought, usdt_to_spend)
            self.save_state()
        else:
            if self.btc_balance > 0:
                profit_loss_pct = (current_price / self.average_cost_price - 1) * 100
                logger.info(f"Posição aberta. P/L atual: {profit_loss_pct:+.2f}%")

    def run(self, cycles=500, retrain_every=50): # Retreinamento mais frequente
        self.initialize()
        for i in range(cycles):
            logger.info(f"--- 🔄 Ciclo {i + 1}/{cycles} ---")
            try:
                current_price = self.client.get_current_price(SYMBOL)
                logger.info(f"💹 Preço BTC: ${current_price:,.2f}")

                historical_data = pd.read_csv('data/historical_data.csv')
                features = self.ml_model.prepare_data_for_prediction(historical_data, current_price)
                _, signal_strength = self.ml_model.predict_signal(features)
                
                self.execute_action(current_price, signal_strength)
                
                portfolio_value = self.usdt_balance + (self.btc_balance * current_price)
                logger.info(f"Portfólio: ${portfolio_value:,.2f} | USDT: {self.usdt_balance:,.2f} | BTC: {self.btc_balance:.8f} | Custo Médio: ${self.average_cost_price:,.2f}")

                if (i + 1) % retrain_every == 0:
                    self.client.update_historical_data(SYMBOL, '1h')
                    self.ml_model.train(pd.read_csv('data/historical_data.csv'))

            except Exception as e:
                logger.error(f"Erro no ciclo de trading: {e}", exc_info=True)

            if i < cycles - 1: time.sleep(5)
        logger.info("🏁 Execução finalizada!")