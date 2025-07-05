# src/confidence_manager.py

import numpy as np
from src.logger import logger

class AdaptiveConfidenceManager:
    """
    Gerencia dinamicamente o limiar de confiança para entrada em trades
    com base na performance recente.
    """
    def __init__(self, initial_confidence: float, learning_rate: float = 0.05, min_confidence: float = 0.505, max_confidence: float = 0.85):
        """
        Args:
            initial_confidence (float): O limiar de confiança inicial, otimizado pelo Optuna.
            learning_rate (float): Quão agressivamente a confiança se ajusta. Valores maiores = ajustes mais rápidos.
            min_confidence (float): O valor mínimo que a confiança pode atingir.
            max_confidence (float): O valor máximo que a confiança pode atingir.
        """
        self.initial_confidence = initial_confidence
        self.current_confidence = initial_confidence
        self.learning_rate = learning_rate
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.trade_count = 0
        
        logger.debug(f"AdaptiveConfidenceManager inicializado com confiança inicial de {initial_confidence:.3f}")

    def update(self, pnl_percent: float):
        """
        Atualiza o limiar de confiança com base no resultado do último trade.
        - Se o trade foi lucrativo (pnl_percent > 0), a confiança diminui (fica mais "ousado").
        - Se o trade deu prejuízo (pnl_percent <= 0), a confiança aumenta (fica mais "cauteloso").
        """
        self.trade_count += 1
        
        # O ajuste é proporcional ao PnL, mas com um limite para evitar mudanças bruscas
        # Um lucro de 2% (0.02) ou uma perda de 2% (-0.02) são considerados o "máximo" para o ajuste.
        clamped_pnl = np.clip(pnl_percent, -0.02, 0.02)
        
        # A fórmula central: subtrai o PnL ponderado. Se PnL é positivo, confiança cai. Se PnL é negativo, confiança sobe.
        adjustment = self.learning_rate * clamped_pnl
        new_confidence = self.current_confidence - adjustment
        
        # Garante que a nova confiança permaneça dentro dos limites definidos
        self.current_confidence = np.clip(new_confidence, self.min_confidence, self.max_confidence)
        
        logger.debug(f"Trade #{self.trade_count}: PnL={pnl_percent:+.2%}. Confiança ajustada de {self.current_confidence + adjustment:.3f} para {self.current_confidence:.3f}")

    def get_confidence(self) -> float:
        """Retorna o limiar de confiança atual."""
        return self.current_confidence