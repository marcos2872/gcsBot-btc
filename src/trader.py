# src/trader.py
from .client import BinanceClient
from .ml_trading import MLTradingModel
from .logger import logger
import time

class TradingBot:
    def __init__(self):
        self.client = BinanceClient()
        self.ml_model = MLTradingModel()
        self.symbol = "BTCUSDT"
        self.min_order_size = 0.001  # BTC mínimo para ordem
        
    def initialize(self):
        """Inicializa o bot"""
        logger.info("🚀 Inicializando Trading Bot...")
        
        # Verificar saldos
        usdt_balance = self.client.get_balance('USDT')
        btc_balance = self.client.get_balance('BTC')
        
        logger.info(f"💰 Saldos - USDT: {usdt_balance:.2f}, BTC: {btc_balance:.8f}")
        
        # Coletar dados históricos
        logger.info("📊 Coletando dados históricos...")
        self.client.save_historical_data(self.symbol, '1h', 200)
        
        # Treinar modelo
        logger.info("🧠 Treinando modelo ML...")
        historical_data = self.client.get_historical_data(self.symbol, '1h', 200)
        
        if not self.ml_model.load_model():
            self.ml_model.train(historical_data)
        
        logger.info("✅ Bot inicializado com sucesso!")
    
    def execute_trade(self, action, current_price):
        """Executa operação baseada na ação"""
        usdt_balance = self.client.get_balance('USDT')
        btc_balance = self.client.get_balance('BTC')
        
        if action == 1:  # Buy
            if usdt_balance >= 11:  # Mínimo ~$11 para comprar BTC
                quantity = (usdt_balance * 0.95) / current_price  # Usar 95% do saldo
                quantity = round(quantity, 6)
                
                if quantity >= self.min_order_size:
                    order = self.client.place_order(self.symbol, 'BUY', quantity)
                    if order:
                        logger.info(f"🟢 COMPRA executada: {quantity:.6f} BTC por ${current_price:.2f}")
                        return True
                else:
                    logger.warning("Quantidade insuficiente para compra")
            else:
                logger.warning("Saldo USDT insuficiente para compra")
                
        elif action == 2:  # Sell
            if btc_balance >= self.min_order_size:
                quantity = round(btc_balance * 0.99, 6)  # Vender 99% do BTC
                
                order = self.client.place_order(self.symbol, 'SELL', quantity)
                if order:
                    logger.info(f"🔴 VENDA executada: {quantity:.6f} BTC por ${current_price:.2f}")
                    return True
            else:
                logger.warning("Saldo BTC insuficiente para venda")
        
        return False
    
    def run_trading_cycle(self):
        """Executa um ciclo de trading"""
        try:
            # Obter preço atual
            current_price = self.client.get_current_price(self.symbol)
            logger.info(f"💹 Preço atual BTC: ${current_price:.2f}")
            
            # Obter dados recentes para predição
            recent_data = self.client.get_historical_data(self.symbol, '1h', 50)
            
            if not recent_data:
                logger.warning("Dados insuficientes para análise")
                return
            
            # Fazer predição
            action = self.ml_model.predict(recent_data)
            
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            logger.info(f"🎯 Ação recomendada: {action_map[action]}")
            
            # Executar ação se não for HOLD
            if action != 0:
                self.execute_trade(action, current_price)
                
        except Exception as e:
            logger.error(f"Erro no ciclo de trading: {e}")
    
    def run(self, cycles=1, interval_minutes=60):
        """Executa o bot"""
        self.initialize()
        
        for cycle in range(cycles):
            logger.info(f"🔄 Ciclo {cycle + 1}/{cycles}")
            self.run_trading_cycle()
            
            if cycle < cycles - 1:
                logger.info(f"⏰ Aguardando {interval_minutes} minutos...")
                time.sleep(interval_minutes * 60)
        
        logger.info("🏁 Execução finalizada!")