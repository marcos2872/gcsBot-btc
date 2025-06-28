# src/model_trainer.py

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from numba import jit

from src.logger import logger
from src.config import MODEL_FILE, SCALER_FILE
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
from ta.momentum import StochasticOscillator

@jit(nopython=True)
def create_labels_triple_barrier(
    closes: np.ndarray, 
    highs: np.ndarray, 
    lows: np.ndarray, 
    atr: np.ndarray, 
    future_periods: int, 
    profit_multiplier: float, 
    stop_multiplier: float
) -> np.ndarray:
    n = len(closes)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(n - future_periods):
        if atr[i] == 0: continue

        profit_barrier = closes[i] + (atr[i] * profit_multiplier)
        stop_barrier = closes[i] - (atr[i] * stop_multiplier)
        
        target_hit = False
        for j in range(1, future_periods + 1):
            future_high, future_low = highs[i + j], lows[i + j]
            if future_high >= profit_barrier:
                labels[i] = 1
                target_hit = True
                break
            if future_low <= stop_barrier:
                labels[i] = 2
                target_hit = True
                break
        
        if not target_hit:
            labels[i] = 1 if closes[i + future_periods] > closes[i] else 2

    return labels

class ModelTrainer:
    def __init__(self):
        self.feature_names = [
            'sma_7', 'sma_25', 'rsi', 'price_change_1m', 'price_change_5m', 'volume',
            'bb_width', 'atr', 'dxy_change_1m', 'dxy_change_5m',
            'macd_diff', 'stoch_osc'
        ]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos (features) para o modelo,
        com lógica robusta para evitar NaNs.
        """
        # Adiciona um valor muito pequeno (epsilon) para evitar divisão por zero
        epsilon = 1e-10
        
        # --- Volatilidade ---
        atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        # Protegido contra divisão por zero
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + epsilon)
        
        # --- Tendência ---
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_diff'] = macd.macd_diff()

        # --- Momento ---
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_5m'] = df['close'].pct_change(5)
        
        # ### CORREÇÃO DEFINITIVA PARA O RSI ###
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Calcula o RS e substitui valores infinitos (quando loss=0) por um número grande
        rs = gain / (loss + epsilon)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_osc'] = stoch.stoch()

        # --- Features Externas (DXY) ---
        if 'dxy_close' in df.columns:
            df['dxy_change_1m'] = df['dxy_close'].pct_change(1)
            df['dxy_change_5m'] = df['dxy_close'].pct_change(5)
        else:
            df['dxy_change_1m'], df['dxy_change_5m'] = 0, 0
        
        # Remove NaNs iniciais e substitui qualquer Infinito que possa ter restado
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return df

    def train(self, data: pd.DataFrame, all_params: dict):
        if len(data) < 500:
            logger.warning(f"Dados insuficientes para treino ({len(data)} registros). Pulando.")
            return None, None

        logger.debug("Iniciando preparação de features para o treinamento...")
        df_full = self._prepare_features(data.copy())
        
        if df_full.empty:
            # Esta mensagem agora só aparecerá se houver um problema muito sério.
            logger.warning("DataFrame ficou vazio após a preparação de features. Pulando trial.")
            return None, None

        future_periods = all_params.get('future_periods', 30)
        profit_mult = all_params.get('profit_mult', 2.0)
        stop_mult = all_params.get('stop_mult', 2.0)
        
        logger.debug(f"Gerando labels com: future_periods={future_periods}, profit_mult={profit_mult}, stop_mult={stop_mult}")
        
        labels_np = create_labels_triple_barrier(
            closes=df_full['close'].to_numpy(), highs=df_full['high'].to_numpy(),
            lows=df_full['low'].to_numpy(), atr=df_full['atr'].to_numpy(),
            future_periods=future_periods, profit_multiplier=profit_mult, stop_multiplier=stop_mult
        )
        
        y = pd.Series(labels_np, index=df_full.index, name="label")
        X = df_full[self.feature_names]
        
        if y.value_counts().get(1, 0) < 20 or y.value_counts().get(2, 0) < 20:
            logger.warning(f"Não há exemplos suficientes de compra/venda. Counts: {y.value_counts().to_dict()}")
            return None, None
            
        logger.debug("Normalizando features e treinando o modelo LightGBM...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model_params = {
            'n_estimators': all_params.get('n_estimators', 200),
            'learning_rate': all_params.get('learning_rate', 0.05),
            'num_leaves': all_params.get('num_leaves', 31),
            'max_depth': all_params.get('max_depth', 10),
            'min_child_samples': all_params.get('min_child_samples', 20),
        }

        model = LGBMClassifier(**model_params, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_scaled, y)
        
        logger.debug("Treinamento do modelo concluído com sucesso.")
        return model, scaler

    def save_model(self, model, scaler):
        """Salva o modelo e o normalizador de features em arquivos .pkl."""
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        logger.info(f"✅ Modelo final e robusto salvo em '{MODEL_FILE}' e '{SCALER_FILE}'")