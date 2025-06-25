# src/ml_trading.py
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.logger import logger

class MLTrader:
    def __init__(self, model_path='data/adaptive_trader_model.pkl'):
        self.model_path = model_path
        self.model = None
        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def prepare_features(self, data):
        # Usando um conjunto rico de indicadores para dar o máximo de contexto ao modelo.
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
        
        # Remove quaisquer linhas com NaN gerados pelos indicadores
        data.dropna(inplace=True)
        
        # --- LÓGICA DE ALVO (TARGET) MAIS INTELIGENTE ---
        # O alvo é definido pela mudança percentual do preço nos próximos 5 períodos.
        future_price_change_pct = (data['close'].shift(-5) - data['close']) / data['close']
        
        # Limiares para compra e venda
        buy_threshold = 0.001  # Compra se o preço subir 0.1%
        sell_threshold = -0.001 # Vende se o preço cair 0.1%
        
        # Cria as classes: 0 para HOLD, 1 para BUY, 2 para SELL
        conditions = [
            future_price_change_pct > buy_threshold,
            future_price_change_pct < sell_threshold
        ]
        choices = [1, 2] # BUY, SELL
        data['target'] = np.select(conditions, choices, default=0) # HOLD é o padrão
        
        # Remove as linhas onde o futuro é desconhecido
        data.dropna(subset=['target'], inplace=True)

        features = ['sma_7', 'sma_25', 'rsi', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']
        return data[features], data['target']

    def train(self, data):
        features, target = self.prepare_features(data)
        if len(features) < 50:
            logger.error("Dados insuficientes para treinar o modelo adaptativo.")
            return

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1, max_depth=10)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Modelo de Intuição Adaptativa treinado. Acurácia: {accuracy:.2f}")
        self.save_model()

    def prepare_data_for_prediction(self, historical_data, current_price):
        latest_data = historical_data.copy()
        new_row = pd.DataFrame([{'close': current_price}])
        latest_data = pd.concat([latest_data, new_row], ignore_index=True)
        
        # Calcula todas as features para a última linha (preço atual)
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

    def predict_signal(self, features):
        if self.model is None: raise ValueError("Modelo não treinado.")
        if features.isnull().values.any(): return 0, {'hold': 1, 'buy': 0, 'sell': 0}

        probabilities = self.model.predict_proba(features)[0] 
        classes = self.model.classes_

        strength = {
            'hold': probabilities[np.where(classes == 0)[0][0]] if 0 in classes else 0,
            'buy': probabilities[np.where(classes == 1)[0][0]] if 1 in classes else 0,
            'sell': probabilities[np.where(classes == 2)[0][0]] if 2 in classes else 0
        }
        
        return classes[np.argmax(probabilities)], strength

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        logger.info("✅ Modelo de Intuição Adaptativa salvo com sucesso.")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info("✅ Modelo de Intuição Adaptativa carregado com sucesso.")
            return True
        return False