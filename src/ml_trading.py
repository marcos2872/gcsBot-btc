# src/ml_trading.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.logger import logger
import numpy as np

class MLTrader:
    def __init__(self, profit_model_path='data/profit_model.pkl', stop_loss_model_path='data/stop_loss_model.pkl'):
        self.profit_model_path = profit_model_path
        self.stop_loss_model_path = stop_loss_model_path
        self.profit_model = None
        self.stop_loss_model = None
        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        os.makedirs(os.path.dirname(self.profit_model_path), exist_ok=True)

    # --- LÓGICA DE ALVOS CORRIGIDA ---
    def prepare_features(self, data):
        """
        Prepara features e calcula os alvos de forma mais robusta,
        garantindo que estamos olhando para o futuro.
        """
        # As features continuam as mesmas
        data['sma_7'] = data['close'].rolling(window=7).mean()
        data['sma_25'] = data['close'].rolling(window=25).mean()
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['std_20'] = data['close'].rolling(window=20).std()
        data['bollinger_upper'] = data['sma_20'] + (data['std_20'] * 2)
        data['bollinger_lower'] = data['sma_20'] - (data['std_20'] * 2)
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # --- CÁLCULO DE ALVO CORRIGIDO ---
        window = 5
        # Alvo de Lucro: O preço MÁXIMO nos próximos 'window' períodos.
        # .rolling().max() calcula o máximo da janela atual.
        # .shift(-window) move esse valor para o presente, efetivamente olhando para o futuro.
        data['future_max_price'] = data['close'].rolling(window).max().shift(-window)
        # Alvo de Stop: O preço MÍNIMO nos próximos 'window' períodos.
        data['future_min_price'] = data['close'].rolling(window).min().shift(-window)
        
        data.dropna(inplace=True)

        features = ['sma_7', 'sma_25', 'rsi', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']
        targets = data[['future_max_price', 'future_min_price']]
        return data[features], targets

    # O resto do arquivo (train, predict, save, load) permanece o mesmo,
    # pois a estrutura de dois modelos de regressão já está correta.
    def train(self, data):
        features, targets = self.prepare_features(data)
        if len(features) < 50:
            logger.error("Não há dados suficientes para treinar os modelos de regressão.")
            return

        y_profit = targets['future_max_price']
        y_stop_loss = targets['future_min_price']

        X_train, X_test, y_train_profit, y_test_profit = train_test_split(features, y_profit, test_size=0.2, random_state=42)
        self.profit_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.profit_model.fit(X_train, y_train_profit)
        profit_preds = self.profit_model.predict(X_test)
        profit_rmse = np.sqrt(mean_squared_error(y_test_profit, profit_preds))
        logger.info(f"Modelo de Lucro treinado. RMSE: {profit_rmse:.2f}")

        X_train, X_test, y_train_stop, y_test_stop = train_test_split(features, y_stop_loss, test_size=0.2, random_state=42)
        self.stop_loss_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.stop_loss_model.fit(X_train, y_train_stop)
        stop_preds = self.stop_loss_model.predict(X_test)
        stop_rmse = np.sqrt(mean_squared_error(y_test_stop, stop_preds))
        logger.info(f"Modelo de Stop-Loss treinado. RMSE: {stop_rmse:.2f}")

        self.save_model()

    def prepare_data_for_prediction(self, historical_data, current_price):
        latest_data = historical_data.copy()
        new_row = pd.DataFrame([{'close': current_price, 'open': 0, 'high': 0, 'low': 0, 'volume': 0}])
        latest_data = pd.concat([latest_data, new_row], ignore_index=True)
        
        latest_data['sma_7'] = latest_data['close'].rolling(window=7).mean()
        latest_data['sma_25'] = latest_data['close'].rolling(window=25).mean()
        delta = latest_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        latest_data['rsi'] = 100 - (100 / (1 + rs))
        latest_data['sma_20'] = latest_data['close'].rolling(window=20).mean()
        latest_data['std_20'] = latest_data['close'].rolling(window=20).std()
        latest_data['bollinger_upper'] = latest_data['sma_20'] + (latest_data['std_20'] * 2)
        latest_data['bollinger_lower'] = latest_data['sma_20'] - (latest_data['std_20'] * 2)
        exp1 = latest_data['close'].ewm(span=12, adjust=False).mean()
        exp2 = latest_data['close'].ewm(span=26, adjust=False).mean()
        latest_data['macd'] = exp1 - exp2
        latest_data['macd_signal'] = latest_data['macd'].ewm(span=9, adjust=False).mean()

        return latest_data[['sma_7', 'sma_25', 'rsi', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']].tail(1)
        
    def save_model(self):
        joblib.dump(self.profit_model, self.profit_model_path)
        joblib.dump(self.stop_loss_model, self.stop_loss_model_path)
        logger.info("✅ Modelos dinâmicos (Lucro e Stop) salvos com sucesso")

    def load_model(self):
        if os.path.exists(self.profit_model_path) and os.path.exists(self.stop_loss_model_path):
            self.profit_model = joblib.load(self.profit_model_path)
            self.stop_loss_model = joblib.load(self.stop_loss_model_path)
            logger.info("✅ Modelos dinâmicos carregados com sucesso")
            return True
        logger.info("Nenhum modelo dinâmico salvo encontrado")
        return False

    def predict(self, features):
        if self.profit_model is None or self.stop_loss_model is None:
            raise ValueError("Os modelos de regressão não foram treinados ou carregados.")
        
        if features.isnull().values.any():
             logger.warning("Features com valores nulos para predição. Ignorando previsão.")
             return None

        profit_prediction = self.profit_model.predict(features)[0]
        stop_loss_prediction = self.stop_loss_model.predict(features)[0]
        
        return {
            'profit_target_price': profit_prediction,
            'stop_loss_price': stop_loss_prediction
        }