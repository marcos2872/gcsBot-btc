import gym
import numpy as np
from stable_baselines3 import PPO

class TradingEnv(gym.Env):
    """Ambiente de trading customizado"""
    def __init__(self, initial_balance=1000, symbol="BTCUSDT"):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.btc_balance = 0
        self.current_price = 0
        self.total_balance = initial_balance
        self.action_space = gym.spaces.Discrete(3)  # 0 = manter, 1 = comprar, 2 = vender
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        """Reseta o ambiente para o próximo episódio"""
        self.balance = self.initial_balance
        self.btc_balance = 0
        self.current_price = 30000  # Preço fictício inicial
        self.total_balance = self.balance
        return np.array([self.total_balance])

    def step(self, action):
        """Realiza uma ação e retorna o próximo estado"""
        if action == 1:  # Comprar
            self.btc_balance += self.balance / self.current_price
            self.balance = 0
        elif action == 2:  # Vender
            self.balance += self.btc_balance * self.current_price
            self.btc_balance = 0

        self.total_balance = self.balance + self.btc_balance * self.current_price
        reward = self.total_balance - self.initial_balance
        done = False
        if self.total_balance <= 0:
            done = True
        return np.array([self.total_balance]), reward, done, {}

    def render(self):
        """Renderiza o estado do ambiente"""
        print(f"Balance: {self.balance}, BTC: {self.btc_balance}, Total Balance: {self.total_balance}")

def train_model():
    """Treinamento do modelo usando Reinforcement Learning (PPO)"""
    env = TradingEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("trading_model")

def predict_action(model, state):
    """Usa o modelo treinado para prever a próxima ação"""
    action, _states = model.predict(state)
    return action
