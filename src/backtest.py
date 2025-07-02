# src/backtest.py (VERSÃO CORRIGIDA E ATUALIZADA)

import numpy as np
import pandas as pd
from src.logger import logger
from src.config import RISK_PER_TRADE_PCT

# --- Constantes de Custo Operacional (Mantidas) ---
FEE_RATE = 0.001       # Taxa de corretagem por operação
SLIPPAGE_RATE = 0.0005 # Derrapagem simulada no preço de execução

def run_backtest(model, scaler, test_data_with_features: pd.DataFrame, strategy_params: dict, feature_names: list):
    """
    Executa um backtest realista com gestão de risco dinâmica.
    A performance (Sharpe Ratio) é calculada com base em retornos por minuto
    para fornecer uma métrica sensível para o otimizador.
    """
    initial_capital = 100.0
    capital = initial_capital
    btc_amount = 0.0
    in_position = False
    buy_price = 0.0
    trade_count = 0
    portfolio_values = []

    # Risco base, que será modulado pela confiança do modelo
    base_risk_per_trade = strategy_params.get('risk_per_trade_pct', RISK_PER_TRADE_PCT)

    # Garante que todas as colunas de features existam no dataframe de teste
    for col in feature_names:
        if col not in test_data_with_features.columns:
            test_data_with_features[col] = 0

    X_test_features = test_data_with_features[feature_names].fillna(0)

    # Prepara os dados e faz a predição para todo o período de teste de uma vez
    X_test_scaled_np = scaler.transform(X_test_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, index=X_test_features.index, columns=feature_names)

    predictions_proba = model.predict_proba(X_test_scaled_df)
    predictions_buy_proba = pd.Series(predictions_proba[:, 1], index=X_test_features.index)

    logger.debug(f"Iniciando backtest com {len(test_data_with_features)} velas e risco base de {base_risk_per_trade:.2%}.")

    # Loop principal que simula a passagem do tempo, vela a vela
    for date, row in test_data_with_features.iterrows():
        price = row['close']
        current_portfolio_value = capital + (btc_amount * price)
        portfolio_values.append({'timestamp': date, 'value': current_portfolio_value})

        # 1. LÓGICA DE SAÍDA: Verifica se a posição deve ser encerrada
        if in_position:
            profit_loss_pct = (price / buy_price) - 1 if buy_price > 0 else 0

            # Condição de saída: take-profit ou stop-loss atingidos
            if (profit_loss_pct >= strategy_params['profit_threshold'] or
                profit_loss_pct <= -strategy_params['stop_loss_threshold']):

                sell_price_with_slippage = price * (1 - SLIPPAGE_RATE)
                capital_after_sell = btc_amount * sell_price_with_slippage
                capital += capital_after_sell * (1 - FEE_RATE)

                btc_amount, in_position, trade_count = 0.0, False, trade_count + 1

        # 2. LÓGICA DE ENTRADA: Verifica se uma nova posição deve ser aberta
        elif predictions_buy_proba.get(date, 0) > strategy_params['prediction_confidence']:

            # --- LÓGICA DE RISCO DINÂMICO (BET SIZING) ---
            conviction = predictions_buy_proba.get(date, 0)
            min_conf = strategy_params['prediction_confidence']

            # Normaliza a força do sinal (0.0 a 1.0) acima do limiar mínimo
            signal_strength = (conviction - min_conf) / (1.0 - min_conf)

            # O risco para este trade varia de 50% a 150% do risco base, conforme a confiança do modelo
            dynamic_risk_pct = base_risk_per_trade * (0.5 + signal_strength)
            trade_size_usdt = capital * dynamic_risk_pct
            # --- FIM DA LÓGICA DE RISCO DINÂMICO ---

            if capital > 10 and trade_size_usdt > 10: # Trade mínimo de $10
                buy_price_with_slippage = price * (1 + SLIPPAGE_RATE)
                amount_to_buy_btc = trade_size_usdt / buy_price_with_slippage
                fee = trade_size_usdt * FEE_RATE # A taxa é sobre o valor em USDT

                btc_amount = amount_to_buy_btc
                capital -= (trade_size_usdt + fee)

                buy_price, in_position, trade_count = buy_price_with_slippage, True, trade_count + 1

    # Se a simulação terminar com uma posição aberta, ela é liquidada ao último preço
    if in_position:
        last_price = test_data_with_features['close'].iloc[-1]
        sell_price_with_slippage = last_price * (1 - SLIPPAGE_RATE)
        capital += (btc_amount * sell_price_with_slippage) * (1 - FEE_RATE)

    # 3. CÁLCULO DE MÉTRICAS DE PERFORMANCE ATUALIZADO
    if not portfolio_values:
        # Retorna -1 para penalizar fortemente no Optuna se nenhuma operação for registrada
        return capital, -1.0

    portfolio_df = pd.DataFrame(portfolio_values).set_index('timestamp')

    # --- MUDANÇA CRÍTICA: REVERTENDO PARA RETORNOS POR MINUTO ---
    # A amostragem horária (resample 'h') que você usava, combinada com a nova gestão
    # de risco (operações pequenas), tornava a variação do portfólio quase nula,
    # resultando em Sharpe Ratio = 0 e impedindo a otimização.
    # Voltamos aos retornos por minuto para dar ao Optuna uma métrica sensível para otimizar.
    portfolio_returns = portfolio_df['value'].pct_change().dropna()

    if portfolio_returns.std() == 0 or len(portfolio_returns) < 2:
        sharpe_ratio = 0.0
    else:
        # Fator de anualização CORRETO para dados de 1 minuto (365 dias * 24 horas * 60 minutos)
        annualization_factor = np.sqrt(365 * 24 * 60)
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * annualization_factor

    logger.debug(f"Backtest concluído. Capital Final: {capital:.2f}, Sharpe (Anualizado): {sharpe_ratio:.2f}, Trades: {trade_count}")
    return capital, sharpe_ratio