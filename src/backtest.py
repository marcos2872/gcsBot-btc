# src/backtest.py (VERSÃO FINAL COM TODAS AS INTELIGÊNCIAS INTEGRADAS)

import numpy as np
import pandas as pd
from src.logger import logger
from src.config import RISK_PER_TRADE_PCT
from src.confidence_manager import AdaptiveConfidenceManager

# --- Constantes de Custo Operacional ---
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

def run_backtest(model, scaler, test_data_with_features: pd.DataFrame, strategy_params: dict, feature_names: list):
    """
    Executa um backtest realista com gestão de confiança adaptativa e risco dinâmico (bet sizing).
    """
    initial_capital = 100.0
    capital = initial_capital
    btc_amount = 0.0
    in_position = False
    buy_price = 0.0
    trade_count = 0
    portfolio_values = []

    base_risk_per_trade = strategy_params.get('risk_per_trade_pct', RISK_PER_TRADE_PCT)

    for col in feature_names:
        if col not in test_data_with_features.columns:
            test_data_with_features[col] = 0

    X_test_features = test_data_with_features[feature_names].fillna(0)
    X_test_scaled_np = scaler.transform(X_test_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, index=X_test_features.index, columns=feature_names)
    predictions_proba = model.predict_proba(X_test_scaled_df)
    predictions_buy_proba = pd.Series(predictions_proba[:, 1], index=X_test_features.index)

    logger.debug(f"Iniciando backtest com {len(test_data_with_features)} velas e risco base de {base_risk_per_trade:.2%}.")

    # Usa a 'initial_confidence' otimizada pelo Optuna para iniciar o cérebro adaptativo
    initial_conf = strategy_params.get('initial_confidence', 0.6)
    confidence_manager = AdaptiveConfidenceManager(initial_confidence=initial_conf)

    for date, row in test_data_with_features.iterrows():
        price = row['close']
        current_portfolio_value = capital + (btc_amount * price)
        portfolio_values.append({'timestamp': date, 'value': current_portfolio_value})

        # 1. LÓGICA DE SAÍDA
        if in_position:
            profit_loss_pct = (price / buy_price) - 1 if buy_price > 0 else 0

            if (profit_loss_pct >= strategy_params['profit_threshold'] or
                profit_loss_pct <= -strategy_params['stop_loss_threshold']):

                sell_price_with_slippage = price * (1 - SLIPPAGE_RATE)
                capital_after_sell = btc_amount * sell_price_with_slippage
                capital += capital_after_sell * (1 - FEE_RATE)

                pnl_do_trade = (sell_price_with_slippage / buy_price) - 1
                # O cérebro aprende com o resultado do trade
                confidence_manager.update(pnl_do_trade)

                btc_amount, in_position, trade_count = 0.0, False, trade_count + 1

        # 2. LÓGICA DE ENTRADA
        else: # <<< CORREÇÃO: Usei 'else' em vez de 'elif' para maior clareza
            # <<< CORREÇÃO: Pega o limiar de confiança ATUAL e DINÂMICO do gerenciador
            current_confidence_threshold = confidence_manager.get_confidence()
            conviction = predictions_buy_proba.get(date, 0)

            if conviction > current_confidence_threshold:
                # --- LÓGICA DE RISCO DINÂMICO (BET SIZING) ---
                # <<< CORREÇÃO: A força do sinal agora é baseada no limiar dinâmico
                signal_strength = (conviction - current_confidence_threshold) / (1.0 - current_confidence_threshold)

                dynamic_risk_pct = base_risk_per_trade * (0.5 + signal_strength)
                trade_size_usdt = capital * dynamic_risk_pct
                
                if capital > 10 and trade_size_usdt > 10:
                    buy_price_with_slippage = price * (1 + SLIPPAGE_RATE)
                    amount_to_buy_btc = trade_size_usdt / buy_price_with_slippage
                    fee = trade_size_usdt * FEE_RATE

                    btc_amount = amount_to_buy_btc
                    capital -= (trade_size_usdt + fee)
                    buy_price, in_position, trade_count = buy_price_with_slippage, True, trade_count + 1

    # Liquidação final
    if in_position:
        last_price = test_data_with_features['close'].iloc[-1]
        sell_price_with_slippage = last_price * (1 - SLIPPAGE_RATE)
        capital += (btc_amount * sell_price_with_slippage) * (1 - FEE_RATE)

    # 3. CÁLCULO DE MÉTRICAS DE PERFORMANCE
    if not portfolio_values:
        return capital, -1.0

    portfolio_df = pd.DataFrame(portfolio_values).set_index('timestamp')
    portfolio_returns = portfolio_df['value'].pct_change().dropna()

    if portfolio_returns.std() == 0 or len(portfolio_returns) < 2:
        sharpe_ratio = 0.0
    else:
        annualization_factor = np.sqrt(365 * 24 * 60)
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * annualization_factor
        logger.debug(f"sharpe_ratio {sharpe_ratio} = (portfolio_returns.mean() {portfolio_returns.mean()} / portfolio_returns.std(){portfolio_returns.std()})  * annualization_factor {annualization_factor}")
        

    logger.debug(f"Backtest concluído. Capital Final: {capital:.2f}, Sharpe (Anualizado): {sharpe_ratio:.2f}, Trades: {trade_count}")
    return capital, sharpe_ratio