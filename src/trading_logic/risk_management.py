# src/trading_logic/risk_management.py

class RiskManager:
    def __init__(self, risk_percentage=0.01, max_loss=0.05):
        self.risk_percentage = risk_percentage
        self.max_loss = max_loss

    def calculate_trade_size(self, balance, price, stop_loss_percent):
        """Calcula o tamanho da posição com base no risco e no stop loss"""
        capital_risked = balance * self.risk_percentage
        trade_size = capital_risked / (price * (stop_loss_percent / 100))
        return trade_size

    def check_max_loss(self, total_loss):
        """Verifica se a perda máxima foi atingida"""
        if total_loss >= self.max_loss:
            return True
        return False
