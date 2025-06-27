# src/model_trainer.py (VERSÃO OTIMIZADA COM NUMBA)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from src.logger import logger
from src.config import MODEL_FILE, SCALER_FILE
from ta.volatility import BollingerBands, AverageTrueRange
from numba import jit # Importa o compilador JIT do Numba

# ======================================================================
# NOVO: Função de labels otimizada com Numba
# O decorador @jit compila esta função para código de máquina, tornando-a muito mais rápida.
# nopython=True garante a máxima performance.
# ======================================================================
@jit(nopython=True)
def create_labels_fast(closes, volatility, future_periods, profit_mult, stop_mult):
    """
    Versão da criação de labels otimizada com Numba.
    Opera sobre arrays NumPy para máxima velocidade.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int64)
    profit_barrier = closes + volatility * profit_mult
    stop_barrier = closes - volatility * stop_mult

    for i in range(n - future_periods):
        # Itera sobre a janela futura para cada ponto
        for j in range(1, future_periods + 1):
            future_price = closes[i + j]
            
            # Checa se tocou a barreira de lucro
            if future_price >= profit_barrier[i]:
                labels[i] = 1  # Compra
                break  # Sai do loop interno, pois um resultado foi encontrado
            
            # Checa se tocou a barreira de stop
            if future_price <= stop_barrier[i]:
                labels[i] = 2  # Venda
                break # Sai do loop interno
    
    return labels

class ModelTrainer:
    def __init__(self):
        self.feature_names = [
            'sma_7', 'sma_25', 'rsi', 'price_change_1m', 'price_change_5m', 'volume',
            'bb_width', 'atr'
        ]

    def _prepare_features(self, df):
        # Este método continua o mesmo
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_5m'] = df['close'].pct_change(5)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df.dropna(inplace=True)
        return df

    def train(self, data, model_params):
        if len(data) < 2000:
            logger.warning(f"Dados insuficientes ({len(data)} registros).")
            return None, None

        df_full = self._prepare_features(data.copy())
        
        # Parâmetros para a criação de labels
        future_periods = model_params.pop('future_periods', 15)
        profit_mult = model_params.pop('profit_mult', 1.5)
        stop_mult = model_params.pop('stop_mult', 1.5)
        
        # Prepara os arrays NumPy para a função otimizada
        closes_np = df_full['close'].to_numpy()
        volatility_np = df_full['atr'].rolling(window=future_periods).mean().bfill().to_numpy()
        
        # Chama a nova função rápida
        labels_np = create_labels_fast(closes_np, volatility_np, future_periods, profit_mult, stop_mult)
        
        # Converte de volta para Série Pandas para alinhamento
        y = pd.Series(labels_np, index=df_full.index)
        
        X = df_full[self.feature_names]
        
        if y.value_counts().get(1, 0) < 10 or y.value_counts().get(2, 0) < 10:
            logger.warning("Não há exemplos suficientes de compra/venda. Pulando.")
            return None, None
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(**model_params, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_scaled, y)
        
        return model, scaler

    def save_model(self, model, scaler):
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        logger.info(f"✅ Modelo final e robusto salvo em '{MODEL_FILE}'")