# src/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from src.logger import logger
from src.config import MODEL_FILE, SCALER_FILE
from ta.volatility import BollingerBands, AverageTrueRange

class ModelTrainer:
    def __init__(self):
        self.feature_names = [
            'sma_7', 'sma_25', 'rsi', 'price_change_1m', 'price_change_5m', 'volume',
            'bb_width', 'atr'
        ]

    def _prepare_features(self, df):
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

    def _create_labels(self, df, future_periods=15, profit_mult=1.5, stop_mult=1.5):
        volatility = df['atr'].rolling(window=future_periods).mean().bfill()
        profit_barrier = df['close'] + volatility * profit_mult
        stop_barrier = df['close'] - volatility * stop_mult
        
        labels = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        
        for i in range(len(df) - future_periods):
            path = df['close'].iloc[i+1 : i+1+future_periods]
            
            profit_touch = path[path >= profit_barrier.iloc[i]].first_valid_index()
            stop_touch = path[path <= stop_barrier.iloc[i]].first_valid_index()
            
            if profit_touch is not None and (stop_touch is None or profit_touch < stop_touch):
                labels.iloc[i] = 1 # Compra
            elif stop_touch is not None and (profit_touch is None or stop_touch < profit_touch):
                labels.iloc[i] = 2 # Venda
        
        return labels

    def train(self, data, model_params):
        if len(data) < 2000:
            logger.warning(f"Dados insuficientes para um treino robusto ({len(data)} registros).")
            return None, None

        df_full = self._prepare_features(data.copy())
        
        labels = self._create_labels(
            df_full,
            future_periods=model_params.pop('future_periods', 15),
            profit_mult=model_params.pop('profit_mult', 1.5),
            stop_mult=model_params.pop('stop_mult', 1.5)
        )
        
        X = df_full[self.feature_names]
        min_len = min(len(X), len(labels))
        X = X.iloc[:min_len]
        y = labels.iloc[:min_len]
        
        if y.value_counts().get(1, 0) < 10 or y.value_counts().get(2, 0) < 10:
            logger.warning("Não há exemplos suficientes de compra/venda neste período de treino. Pulando.")
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