import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .logger import logger

class MLTradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data):
        """Prepara features para o modelo"""
        df = pd.DataFrame(data)
        
        # Indicadores técnicos simples
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Features finais
        features = ['sma_5', 'sma_20', 'rsi', 'price_change', 'volume_change']
        
        # Remover NaN
        df = df.dropna()
        
        return df[features].values, df
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_labels(self, df, threshold=0.02):
        """Cria labels para treinamento"""
        # 0 = Hold, 1 = Buy, 2 = Sell
        future_return = df['close'].shift(-1) / df['close'] - 1
        
        labels = np.where(future_return > threshold, 1,  # Buy
                         np.where(future_return < -threshold, 2, 0))  # Sell, Hold
        
        return labels[:-1]  # Remove último elemento (sem futuro)
    
    def train(self, historical_data):
        """Treina o modelo"""
        try:
            if len(historical_data) < 50:
                logger.warning("Dados insuficientes para treinamento")
                return False
            
            X, df = self.prepare_features(historical_data)
            y = self.create_labels(df)
            
            # Ajustar tamanhos
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Normalização
            X_scaled = self.scaler.fit_transform(X)
            
            # Treinamento
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Salvar modelo
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/trading_model.pkl')
            joblib.dump(self.scaler, 'models/scaler.pkl')
            
            logger.info("✅ Modelo treinado e salvo com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return False
    
    def load_model(self):
        """Carrega modelo salvo"""
        try:
            self.model = joblib.load('models/trading_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.is_trained = True
            logger.info("✅ Modelo carregado com sucesso")
            return True
        except:
            logger.info("Nenhum modelo salvo encontrado")
            return False
    
    def predict(self, current_data):
        """Faz predição"""
        if not self.is_trained:
            return 0  # Hold por padrão
        
        try:
            X, _ = self.prepare_features(current_data)
            if len(X) == 0:
                return 0
            
            X_scaled = self.scaler.transform(X[-1:])  # Último ponto
            prediction = self.model.predict(X_scaled)[0]
            
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return 0