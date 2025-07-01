import numpy as np
import pandas as pd
from src.logger import logger

# --- Constantes de Custo Operacional ---
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005

def run_backtest(model, scaler, test_data_with_features: pd.DataFrame, strategy_params: dict, feature_names: list):
    """
    Executa um backtest realista com gestão de risco percentual e Sharpe Ratio estabilizado.
    """
    initial_capital = 100.0
    capital = initial_capital
    btc_amount = 0.0
    in_position = False
    buy_price = 0.0
    trade_count = 0
    portfolio_values = [] 

    risk_per_trade_pct = strategy_params.get('risk_per_trade_pct', 0.02)

    for col in feature_names:
        if col not in test_data_with_features.columns:
            test_data_with_features[col] = 0
            
    X_test_features = test_data_with_features[feature_names].fillna(0)
    
    # --- ATUALIZADO: Lógica para remover o UserWarning ---
    # 1. Normaliza as features, o que resulta em uma matriz NumPy
    X_test_scaled_np = scaler.transform(X_test_features)
    
    # 2. Converte a matriz de volta para um DataFrame, preservando os nomes das colunas
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, index=X_test_features.index, columns=feature_names)
    
    # 3. Gera as predições usando o DataFrame, que agora tem nomes de colunas válidos
    predictions_proba = model.predict_proba(X_test_scaled_df)
    # --- Fim da atualização ---
    
    predictions_buy_proba = pd.Series(predictions_proba[:, 1], index=X_test_features.index)
    
    logger.debug(f"Iniciando backtest realista com {len(test_data_with_features)} velas e risco de {risk_per_trade_pct:.2%}.")

    for date, row in test_data_with_features.iterrows():
        price = row['close']
        current_portfolio_value = capital + (btc_amount * price)
        portfolio_values.append({'timestamp': date, 'value': current_portfolio_value})

        if in_position:
            profit_loss_pct = (price / buy_price) - 1 if buy_price > 0 else 0
            
            if (profit_loss_pct >= strategy_params['profit_threshold'] or 
                profit_loss_pct <= -strategy_params['stop_loss_threshold']):
                
                sell_price_with_slippage = price * (1 - SLIPPAGE_RATE)
                capital_after_sell = btc_amount * sell_price_with_slippage
                capital += capital_after_sell * (1 - FEE_RATE)
                
                btc_amount, in_position, trade_count = 0.0, False, trade_count + 1
        
        elif predictions_buy_proba.get(date, 0) > strategy_params['prediction_confidence']:
            trade_size_usdt = capital * risk_per_trade_pct
            
            if capital > 10 and trade_size_usdt > 10:
                buy_price_with_slippage = price * (1 + SLIPPAGE_RATE)
                amount_to_buy_btc = trade_size_usdt / buy_price_with_slippage
                fee = amount_to_buy_btc * buy_price_with_slippage * FEE_RATE
                
                btc_amount = amount_to_buy_btc
                capital -= (trade_size_usdt + fee)
                
                buy_price, in_position, trade_count = buy_price_with_slippage, True, trade_count + 1

    if in_position:
        last_price = test_data_with_features['close'].iloc[-1]
        sell_price_with_slippage = last_price * (1 - SLIPPAGE_RATE)
        capital += (btc_amount * sell_price_with_slippage) * (1 - FEE_RATE)
    
    if not portfolio_values:
        return capital, -1.0

    portfolio_df = pd.DataFrame(portfolio_values).set_index('timestamp')
    
    # Usa 'h' minúsculo para ser compatível com futuras versões do pandas
    hourly_returns = portfolio_df['value'].resample('h').last().pct_change().dropna()
    
    if hourly_returns.std() == 0 or len(hourly_returns) < 2:
        sharpe_ratio = 0.0
    else:
        annualization_factor = np.sqrt(365 * 24)
        sharpe_ratio = (hourly_returns.mean() / hourly_returns.std()) * annualization_factor
        
    logger.debug(f"Backtest concluído. Capital Final: {capital:.2f}, Sharpe Horário (Anualizado): {sharpe_ratio:.2f}, Trades: {trade_count}")
    return capital, sharpe_ratio