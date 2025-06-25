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
    def __init__(self):
        self.symbol = SYMBOL
        self.client = BinanceClient()
        self.ml_model = MLTrader()
        
        self.state_filepath = 'data/simulation_state.json'
        
        self.simulation_mode = SIMULATION_MODE
        self.usdt_balance = SIMULATION_INITIAL_USDT
        self.btc_balance = SIMULATION_INITIAL_BTC
        self.trade_ratio = SIMULATION_TRADE_RATIO
        self.last_trade_price = 0.0
        self.scalping_profit_target = 0.001 
        self.scalping_dip_target = 0.001    
        
        self.load_state() 

        self.trades_log_file = 'data/trades.csv'
        self._initialize_trade_log()

    def load_state(self):
        if not self.simulation_mode:
            return
        try:
            if os.path.exists(self.state_filepath):
                with open(self.state_filepath, 'r') as f:
                    state = json.load(f)
                    self.usdt_balance = state.get('usdt_balance', self.usdt_balance)
                    self.btc_balance = state.get('btc_balance', self.btc_balance)
                    self.last_trade_price = state.get('last_trade_price', self.last_trade_price)
                    logger.info(f"✅ Estado da simulação carregado: USDT: {self.usdt_balance:.2f}, BTC: {self.btc_balance:.8f}")
            else:
                 logger.info("Nenhum estado de simulação salvo encontrado. A começar com valores iniciais.")
        except Exception as e:
            logger.error(f"Erro ao carregar o estado da simulação: {e}. A usar valores padrão.")

    def save_state(self):
        if not self.simulation_mode:
            return
        state = {
            'usdt_balance': self.usdt_balance,
            'btc_balance': self.btc_balance,
            'last_trade_price': self.last_trade_price
        }
        try:
            with open(self.state_filepath, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Erro ao salvar o estado da simulação: {e}")

    def _initialize_trade_log(self):
        if not os.path.exists(self.trades_log_file):
            with open(self.trades_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'type', 'reason', 'price', 'quantity_btc', 'value_usdt', 'portfolio_value_usdt'])

    def _log_trade(self, trade_type, reason, price, btc_qty, usdt_value):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        portfolio_value = self.usdt_balance + (self.btc_balance * price)
        with open(self.trades_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, trade_type, reason, price, btc_qty, usdt_value, portfolio_value])
            
    def initialize(self):
        logger.info("🚀 A inicializar o Trading Bot...")
        if self.simulation_mode:
            logger.info("--- MODO DE SIMULAÇÃO ATIVADO ---")
            logger.info(f"💰 Saldo Atual - USDT: {self.usdt_balance:.2f}, BTC: {self.btc_balance:.8f}")
        else:
            self.usdt_balance = self.client.get_balance('USDT')
            self.btc_balance = self.client.get_balance('BTC')
            logger.info(f"💰 Saldos Reais - USDT: {self.usdt_balance:.2f}, BTC: {self.btc_balance:.8f}")

        logger.info("📊 A atualizar a base de dados históricos...")
        self.client.update_historical_data(self.symbol, '1h', initial_limit=300)
        
        # Define o last_trade_price apenas se não foi carregado de um estado
        if self.last_trade_price == 0.0:
            self.last_trade_price = self.client.get_current_price(self.symbol)
            logger.info(f"Preço inicial definido para o scalping: ${self.last_trade_price:,.2f}")

        logger.info("🧠 A treinar o modelo ML...")
        try:
            historical_data = pd.read_csv('data/historical_data.csv')
            if not self.ml_model.load_model():
                self.ml_model.train(historical_data)
        except FileNotFoundError:
            logger.error("Falha crítica: ficheiro de dados históricos não encontrado para o treino inicial.")
            raise
        
        logger.info("✅ Bot inicializado com sucesso!")


    def execute_action(self, action, current_price):
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        ml_recommendation = action_map.get(action, 'UNKNOWN')
        final_decision = 'HOLD'
        decision_reason = "N/A"

        # --- INÍCIO DA LÓGICA DE DECISÃO REFINADA ---

        # ESTADO 1: SEM BTC. A ÚNICA MISSÃO É COMPRAR.
        if self.btc_balance == 0:
            decision_reason = f"Estado Inicial (ML previu: {ml_recommendation})"
            if ml_recommendation == 'BUY':
                logger.info("💡 DIRETIVA DE ESTADO INICIAL: ML recomendou COMPRA. A executar.")
                final_decision = 'BUY'
            else:
                logger.info(f"💡 DIRETIVA DE ESTADO INICIAL: A aguardar sinal de COMPRA. Ignorando '{ml_recommendation}'.")
                final_decision = 'HOLD' # Força a espera, ignora qualquer outra recomendação.

        # ESTADO 2: COM BTC. USAR A ESTRATÉGIA HÍBRIDA (SCALPING + ML).
        else:
            # 2.1. Estratégia de Scalping (Prioridade Máxima)
            if self.last_trade_price > 0:
                price_increase = (current_price - self.last_trade_price) / self.last_trade_price
                if self.btc_balance * current_price > 10 and price_increase >= self.scalping_profit_target:
                    final_decision = 'SELL'
                    decision_reason = f"Scalp (Lucro de {price_increase:.2%})"
                
                price_decrease = (self.last_trade_price - current_price) / self.last_trade_price
                if self.usdt_balance > 10 and price_decrease >= self.scalping_dip_target:
                    final_decision = 'BUY'
                    decision_reason = f"Scalp (Queda de {price_decrease:.2%})"

            # 2.2. Lógica do ML (se o scalping não foi acionado)
            if final_decision == 'HOLD':
                decision_reason = f"ML ({ml_recommendation})"
                final_decision = ml_recommendation
                
                # Regra de segurança para evitar ficar sem dinheiro para taxas
                if self.usdt_balance < 10 and final_decision == 'BUY':
                    logger.warning("⚠️ Decisão de COMPRA do ML bloqueada por baixo saldo de USDT.")
                    final_decision = 'HOLD'
        
        logger.info(f"🎯 Decisão Final: {final_decision} | Razão: {decision_reason}")
        
        # --- LÓGICA DE EXECUÇÃO ---
        
        if final_decision == 'BUY' and self.usdt_balance > 10:
            usdt_to_spend = self.usdt_balance * self.trade_ratio
            btc_bought = usdt_to_spend / current_price
            
            self.usdt_balance -= usdt_to_spend
            self.btc_balance += btc_bought
            self.last_trade_price = current_price 
            
            logger.info(f"📈 COMPRA SIMULADA: {btc_bought:.8f} BTC por ${usdt_to_spend:,.2f}")
            self._log_trade('BUY', decision_reason, current_price, btc_bought, usdt_to_spend)
            self.save_state()

        elif final_decision == 'SELL' and self.btc_balance * current_price > 10:
            btc_to_sell = self.btc_balance * self.trade_ratio
            usdt_gained = btc_to_sell * current_price
            
            self.btc_balance -= btc_to_sell
            self.usdt_balance += usdt_gained
            self.last_trade_price = current_price
            
            logger.info(f"📉 VENDA SIMULADA: {btc_to_sell:.8f} BTC por ${usdt_gained:,.2f}")
            self._log_trade('SELL', decision_reason, current_price, btc_to_sell, usdt_gained)
            self.save_state()

    def run_trading_cycle(self):
        try:
            current_price = self.client.get_current_price(self.symbol)
            logger.info(f"💹 Preço atual BTC: ${current_price:,.2f}")

            historical_data = pd.read_csv('data/historical_data.csv')
            features = self.ml_model.prepare_data_for_prediction(historical_data, current_price)
            action = self.ml_model.predict(features)
            
            self.execute_action(action, current_price)

            portfolio_value_usdt = self.usdt_balance + (self.btc_balance * current_price)
            logger.info(f"Portfólio: ${portfolio_value_usdt:,.2f} | USDT: {self.usdt_balance:,.2f} | BTC: {self.btc_balance:.8f}")

        except Exception as e:
            logger.error(f"Ocorreu um erro no ciclo de trading: {e}")

    def run(self, cycles=1000, retrain_every=100):
        self.initialize()
        for i in range(cycles):
            logger.info(f"--- 🔄 Ciclo {i + 1}/{cycles} ---")
            self.run_trading_cycle()
            if (i + 1) % retrain_every == 0 and i > 0:
                logger.info("🔔 HORA DE APRENDER! A retreinar o modelo...")
                logger.info("📊 A atualizar a base de dados históricos...")
                self.client.update_historical_data(self.symbol, '1h')
                logger.info("🧠 A retreinar o modelo ML com a base de dados expandida...")
                historical_data = pd.read_csv('data/historical_data.csv')
                self.ml_model.train(historical_data)
                logger.info("✅ Modelo retreinado e atualizado!")
            if i < cycles - 1:
                time.sleep(5) 
        logger.info("🏁 Execução finalizada!")