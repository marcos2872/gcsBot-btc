# src/optimizer.py

import optuna
import pandas as pd
import numpy as np
import json
import signal
import sys
import os # Adicionado para os.path.exists
from src.model_trainer import ModelTrainer
from src.logger import logger
from src.config import (
    WFO_TRAIN_MINUTES, WFO_TEST_MINUTES, WFO_STEP_MINUTES, WFO_STATE_FILE
)

class WalkForwardOptimizer:
    def __init__(self, full_data):
        self.full_data = full_data
        self.trainer = ModelTrainer()
        self.TRADE_FEE = 0.001
        self.n_trials_for_cycle = 0 # NOVO: Inicializa a variável
        
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    def graceful_shutdown(self, signum, frame):
        if not self.shutdown_requested:
            logger.warning("\n" + "="*50)
            logger.warning("PARADA SOLICITADA! Finalizando o trial atual...")
            logger.warning("O processo será encerrado de forma limpa antes do próximo ciclo de WFO.")
            logger.warning("O progresso até o último ciclo completo foi salvo.")
            logger.warning("="*50)
            self.shutdown_requested = True

    def _save_wfo_state(self, cycle, start_index, all_results):
        state = {
            'last_completed_cycle': cycle - 1,
            'next_start_index': start_index,
            'results_so_far': all_results
        }
        with open(WFO_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Estado da WFO salvo. Ciclo #{cycle - 1} completo.")

    def _load_wfo_state(self):
        if os.path.exists(WFO_STATE_FILE):
            try:
                with open(WFO_STATE_FILE, 'r') as f:
                    state = json.load(f)
                    last_cycle = state.get('last_completed_cycle', 0)
                    logger.info("="*50)
                    logger.info(f"Estado de otimização anterior encontrado! Retomando do ciclo #{last_cycle + 1}.")
                    logger.info("="*50)
                    return state.get('next_start_index', 0), last_cycle + 1, state.get('results_so_far', [])
            except Exception as e:
                logger.error(f"Erro ao carregar estado da WFO: {e}. Começando do zero.")
        return 0, 1, []

    # NOVO: Callback corrigido
    def _progress_callback(self, study, trial):
        """Callback para mostrar o progresso do Optuna a cada trial."""
        n_trials = self.n_trials_for_cycle
        print(f"\r    - [Progresso Optuna] Trial {trial.number + 1}/{n_trials} concluído... Melhor Sharpe até agora: {study.best_value:.4f}", end="", flush=True)


    def _calculate_sharpe_ratio(self, returns):
        if returns.std() == 0 or len(returns) < 2:
            return 0.0
        annualization_factor = np.sqrt(365 * 1440)
        sharpe = (returns.mean() / returns.std()) * annualization_factor
        return sharpe

    def _run_backtest_for_period(self, model, scaler, test_data, strategy_params):
        # ... (este método permanece o mesmo) ...
        initial_capital = 100.0
        capital = initial_capital
        btc_amount = 0.0
        in_position = False
        buy_price = 0.0
        trade_count = 0
        portfolio_values = []

        temp_test_data = test_data.copy()
        X_test_full = self.trainer._prepare_features(temp_test_data)
        if X_test_full.empty:
            return initial_capital, 0, 0.0
            
        X_test_features = X_test_full[self.trainer.feature_names]
        X_test_scaled = scaler.transform(X_test_features)
        
        predictions_proba = model.predict_proba(X_test_scaled)
        
        aligned_indices = X_test_features.index
        predictions_buy_proba = pd.Series(predictions_proba[:, 1], index=aligned_indices)
        predictions_sell_proba = pd.Series(predictions_proba[:, 2], index=aligned_indices)
        
        test_prices = X_test_full['close']

        for date, price in test_prices.items():
            buy_proba = predictions_buy_proba.get(date, 0)
            sell_proba = predictions_sell_proba.get(date, 0)

            if in_position:
                if (price / buy_price - 1) >= strategy_params['profit_threshold'] or \
                   (price / buy_price - 1) <= -strategy_params['stop_loss_threshold'] or \
                   sell_proba > strategy_params['prediction_confidence']:
                    capital += (btc_amount * price) * (1 - self.TRADE_FEE)
                    btc_amount = 0.0
                    in_position = False
                    trade_count += 1
            elif buy_proba > strategy_params['prediction_confidence']:
                if capital > 10:
                    btc_amount = (capital / price) * (1 - self.TRADE_FEE)
                    capital = 0.0
                    buy_price = price
                    in_position = True
                    trade_count += 1
            
            portfolio_values.append(capital + (btc_amount * price))

        if in_position:
            capital += (btc_amount * test_prices.iloc[-1]) * (1 - self.TRADE_FEE)
        
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        
        return capital, trade_count, sharpe_ratio


    def _objective(self, trial, train_data, test_data):
        if self.shutdown_requested:
            raise optuna.exceptions.TrialPruned()
        # ... (este método permanece o mesmo) ...
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 250),
            'max_depth': trial.suggest_int('max_depth', 8, 24),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
            'future_periods': trial.suggest_int('future_periods', 10, 60),
            'profit_mult': trial.suggest_float('profit_mult', 1.0, 3.0),
            'stop_mult': trial.suggest_float('stop_mult', 1.0, 3.0),
        }
        strategy_params = {
            'profit_threshold': trial.suggest_float('profit_threshold', 0.005, 0.03),
            'stop_loss_threshold': trial.suggest_float('stop_loss_threshold', 0.005, 0.03),
            'prediction_confidence': trial.suggest_float('prediction_confidence', 0.60, 0.90)
        }
        
        model_params_copy = model_params.copy()
        model, scaler = self.trainer.train(train_data.copy(), model_params_copy)
        if model is None or scaler is None:
            return -1.0

        _, _, sharpe_ratio = self._run_backtest_for_period(model, scaler, test_data.copy(), strategy_params)
        
        return sharpe_ratio


    def run(self):
        logger.info("="*80)
        logger.info("--- INICIANDO PROCESSO DE OTIMIZAÇÃO WALK-FORWARD (WFO) ---")
        logger.info("="*80)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.full_data.sort_index(inplace=True)

        n_total = len(self.full_data)
        train_size = WFO_TRAIN_MINUTES
        test_size = WFO_TEST_MINUTES
        step_size = WFO_STEP_MINUTES

        if n_total < train_size + test_size:
            logger.error(f"Dados insuficientes. Necessário: {train_size + test_size}, Disponível: {n_total}")
            return

        start_index, cycle, all_results = self._load_wfo_state()
        
        final_model, final_scaler = None, None

        while start_index + train_size + test_size <= n_total:
            if self.shutdown_requested:
                logger.info("Processo de otimização interrompido pelo usuário.")
                break

            train_start, train_end = start_index, start_index + train_size
            test_start, test_end = train_end, train_end + test_size

            train_data = self.full_data.iloc[train_start:train_end]
            test_data = self.full_data.iloc[test_start:test_end]

            train_start_date, train_end_date = train_data.index.min().strftime('%Y-%m-%d'), train_data.index.max().strftime('%Y-%m-%d')
            test_start_date, test_end_date = test_data.index.min().strftime('%Y-%m-%d'), test_data.index.max().strftime('%Y-%m-%d')

            logger.info("\n" + "-"*80)
            logger.info(f"INICIANDO CICLO DE WFO #{cycle}")
            logger.info(f"  - Período de Treino: {train_start_date} a {train_end_date}")
            logger.info(f"  - Período de Teste:  {test_start_date} a {test_end_date}")
            logger.info(f"  - Otimizando parâmetros com Optuna...")

            try:
                # NOVO: Armazena n_trials para o callback
                n_trials_for_this_run = 25
                self.n_trials_for_cycle = n_trials_for_this_run

                study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                study.optimize(
                    lambda trial: self._objective(trial, train_data, test_data), 
                    n_trials=n_trials_for_this_run,
                    n_jobs=-1,
                    callbacks=[self._progress_callback]
                )
                print() 
            except optuna.exceptions.TrialPruned:
                logger.warning("\nTrial interrompido devido à solicitação de parada.")
                break

            if self.shutdown_requested: break

            logger.info(f"\n  - Otimização do ciclo #{cycle} concluída.")
            best_trial = study.best_trial
            logger.info(f"  - Melhor Resultado (Sharpe no Treino): {best_trial.value:.4f}")
            logger.info(f"  - Melhores Parâmetros Encontrados:")
            for key, value in best_trial.params.items():
                if isinstance(value, float): logger.info(f"    - {key}: {value:.6f}")
                else: logger.info(f"    - {key}: {value}")

            if best_trial.value <= 0:
                logger.warning(f"  - Melhor resultado do ciclo não foi positivo. Pulando para o próximo ciclo.")
                start_index += step_size
                cycle += 1
                self._save_wfo_state(cycle, start_index, all_results)
                continue

            model_params = {k: v for k, v in best_trial.params.items() if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'future_periods', 'profit_mult', 'stop_mult']}
            strategy_params = {k: v for k, v in best_trial.params.items() if k not in model_params}

            logger.info("  - Treinando modelo final do ciclo com os melhores parâmetros...")
            final_model, final_scaler = self.trainer.train(train_data.copy(), model_params)
            
            if final_model is None:
                logger.error("  - Falha ao treinar o modelo final do ciclo. Pulando.")
                start_index += step_size
                cycle += 1
                self._save_wfo_state(cycle, start_index, all_results)
                continue

            logger.info("  - Executando backtest no período de teste (fora da amostra)...")
            capital, trades, sharpe = self._run_backtest_for_period(final_model, final_scaler, test_data, strategy_params)

            all_results.append({'period': f"{test_start_date}_a_{test_end_date}", 'capital': capital, 'trades': trades, 'sharpe': sharpe})
            
            logger.info("-" * 25 + " RESULTADO DO CICLO " + str(cycle) + " " + "-" * 26)
            logger.info(f"  - Capital Final: ${capital:,.2f} (Retorno de {(capital - 100):.2f}%)")
            logger.info(f"  - Trades Executados: {trades}")
            logger.info(f"  - Sharpe Ratio (Anualizado): {sharpe:.2f}")
            logger.info("-" * 80)

            start_index += step_size
            cycle += 1
            
            self._save_wfo_state(cycle, start_index, all_results)

        logger.info("\n\n" + "="*80)
        logger.info("--- OTIMIZAÇÃO WALK-FORWARD CONCLUÍDA ---")
        logger.info("="*80)
        
        if not all_results:
            logger.warning("Nenhum ciclo de WFO foi concluído.")
            return

        logger.info("--- RESULTADO CONSOLIDADO POR PERÍODO ---")
        total_capital = 100.0
        for res in all_results:
            profit_pct = (res['capital'] - 100) / 100
            total_capital *= (1 + profit_pct)
            logger.info(f"  - Período: {res['period']} | Trades: {res['trades']:>3} | Resultado: {profit_pct:+.2%} | Sharpe: {res['sharpe']:.2f}")
        
        logger.info("\n" + "-"*80)
        logger.info("--- RESUMO ESTATÍSTICO GERAL ---")
        
        num_cycles = len(all_results)
        total_trades = sum(res['trades'] for res in all_results)
        sharpe_values = [res['sharpe'] for res in all_results if res['sharpe'] is not None]
        positive_cycles = sum(1 for res in all_results if res['capital'] > 100)
        
        logger.info(f"  - Total de Ciclos de WFO Completos: {num_cycles}")
        logger.info(f"  - Capital Final Acumulado (Simulado): ${total_capital:,.2f}")
        logger.info(f"  - Lucro Total no Período Simulado: {(total_capital - 100) / 100:+.2%}")
        logger.info(f"  - Total de Trades em todos os ciclos: {total_trades}")
        if sharpe_values:
            logger.info(f"  - Média do Sharpe Ratio nos períodos de teste: {np.mean(sharpe_values):.2f}")
            logger.info(f"  - Mediana do Sharpe Ratio nos períodos de teste: {np.median(sharpe_values):.2f}")
        logger.info(f"  - Taxa de Sucesso (Ciclos com Lucro): {positive_cycles / num_cycles if num_cycles > 0 else 0:.2%}")
        logger.info("-" * 80)

        if final_model and final_scaler:
            logger.info("Salvando o modelo e o normalizador do último ciclo bem-sucedido...")
            self.trainer.save_model(final_model, final_scaler)