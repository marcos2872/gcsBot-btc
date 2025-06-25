import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.logger import logger

class MLTrader:
    def __init__(self, model_path='data/ml_model.pkl'):
        self.model_path = model_path
        self.model = None
        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def prepare_features(self, data):
        """
        Prepara recursos (features) e alvos (labels) usando indicadores técnicos.
        """
        # --- NOVOS INDICADORES ---
        # 1. Bandas de Bollinger: Medem a volatilidade.
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['std_20'] = data['close'].rolling(window=20).std()
        data['bollinger_upper'] = data['sma_20'] + (data['std_20'] * 2)
        data['bollinger_lower'] = data['sma_20'] - (data['std_20'] * 2)
        
        # 2. MACD (Moving Average Convergence Divergence): Mede a tendência e o momentum.
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

        data.dropna(inplace=True)
        
        # Define o alvo (target) com base na próxima variação de preço
        data['future_price_change'] = data['close'].shift(-5).rolling(5).mean().diff()
        data['target'] = 0 # HOLD
        data.loc[data['future_price_change'] > data['future_price_change'].quantile(0.65), 'target'] = 1 # BUY
        data.loc[data['future_price_change'] < data['future_price_change'].quantile(0.35), 'target'] = 2 # SELL
        
        data.dropna(inplace=True)

        features = ['sma_20', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']
        return data[features], data['target']

    def train(self, data):
        """
        Treina o modelo de RandomForestClassifier.
        """
        features, target = self.prepare_features(data)
        
        if len(features) < 10:
            logger.error("Não há dados suficientes para treinar o modelo.")
            return

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Acurácia do modelo melhorado: {accuracy:.2f}")
        
        self.save_model()

    def prepare_data_for_prediction(self, historical_data, current_price):
        """Prepara os dados mais recentes para fazer uma nova previsão."""
        latest_data = historical_data.copy()
        
        new_row = pd.DataFrame([{'close': current_price}])
        latest_data = pd.concat([latest_data, new_row], ignore_index=True)
        
        # Calcula as features com os dados atualizados
        latest_data['sma_20'] = latest_data['close'].rolling(window=20).mean()
        latest_data['std_20'] = latest_data['close'].rolling(window=20).std()
        latest_data['bollinger_upper'] = latest_data['sma_20'] + (latest_data['std_20'] * 2)
        latest_data['bollinger_lower'] = latest_data['sma_20'] - (latest_data['std_20'] * 2)
        exp1 = latest_data['close'].ewm(span=12, adjust=False).mean()
        exp2 = latest_data['close'].ewm(span=26, adjust=False).mean()
        latest_data['macd'] = exp1 - exp2
        latest_data['macd_signal'] = latest_data['macd'].ewm(span=9, adjust=False).mean()

        return latest_data[['sma_20', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']].tail(1)
        
    # --- Os outros métodos (save, load, predict) continuam os mesmos ---
    def save_model(self):
        """Salva o modelo treinado em um arquivo."""
        joblib.dump(self.model, self.model_path)
        logger.info("✅ Modelo treinado e salvo com sucesso")

    def load_model(self):
        """
        Carrega o modelo de um arquivo, se existir.
        Retorna True se o modelo foi carregado, False caso contrário.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info("✅ Modelo carregado com sucesso")
            return True
        logger.info("Nenhum modelo salvo encontrado")
        return False

    def predict(self, features):
        """
        Faz uma previsão de ação (comprar, vender, manter).
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado ou carregado.")
        return self.model.predict(features)[0]