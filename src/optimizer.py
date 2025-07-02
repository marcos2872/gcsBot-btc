# src/optimizer.py

import optuna
import pandas as pd
import numpy as np
import json
import signal
import os
import math
import gc

from src.model_trainer import ModelTrainer
from src.backtest import run_backtest
from src.logger import logger
from src.config import (
    WFO_TRAIN_MINUTES, WFO_TEST_MINUTES, WFO_STEP_MINUTES, WFO_STATE_FILE,
    STRATEGY_PARAMS_FILE, RISK_PER_TRADE_PCT, MODEL_FILE, SCALER_FILE
)

class WalkForwardOptimizer:
    def __init__(self, full_data):
        self.full_data = full_data
        self.trainer = ModelTrainer()
        self.n_trials_for_cycle = 100
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    def graceful_shutdown(self, signum, frame):
        if not self.shutdown_requested:
            logger.warning("\n" + "="*50 + "\nPARADA SOLICITADA! Finalizando o trial atual...\n" + "="*50)
            self.shutdown_requested = True

    def _save_wfo_state(self, cycle, start_index, all_results, cumulative_capital):
        state = {
            'last_completed_cycle': cycle - 1, 'next_start_index': start_index,
            'results_so_far': all_results, 'cumulative_capital': cumulative_capital,
        }
        with open(WFO_STATE_FILE, 'w') as f: json.dump(state, f, indent=4)
        logger.info(f"Estado da WFO salvo. Ciclo #{cycle - 1} completo.")

    def _load_wfo_state(self):
        if os.path.exists(WFO_STATE_FILE):
            try:
                with open(WFO_STATE_FILE, 'r') as f: state = json.load(f)
                last_cycle = state.get('last_completed_cycle', 0)
                cumulative_capital = state.get('cumulative_capital', 100.0)
                logger.info("="*50 + f"\nEstado de otimização anterior encontrado! Retomando do ciclo #{last_cycle + 1}.\n" + "="*50)
                return state.get('next_start_index', 0), last_cycle + 1, state.get('results_so_far', []), cumulative_capital
            except Exception as e:
                logger.error(f"Erro ao carregar estado da WFO: {e}. Começando do zero.")
        return 0, 1, [], 100.0

    def _progress_callback(self, study, trial):
        best_value = 0.0 if study.best_value is None else study.best_value
        print(f"\r    - [Progresso Optuna] Trial {trial.number + 1}/{self.n_trials_for_cycle}... Melhor Sharpe: {best_value:.4f}", end="", flush=True)

    def _objective(self, trial, train_data, test_data):
        if self.shutdown_requested: raise optuna.exceptions.TrialPruned()
        
        all_params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 30, 100),
            'max_depth': trial.suggest_int('max_depth', 7, 25),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 70),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'future_periods': trial.suggest_int('future_periods', 5, 120),
            'profit_mult': trial.suggest_float('profit_mult', 0.5, 5.0),
            'stop_mult': trial.suggest_float('stop_mult', 0.5, 5.0),
            'profit_threshold': trial.suggest_float('profit_threshold', 0.003, 0.05),
            'stop_loss_threshold': trial.suggest_float('stop_loss_threshold', 0.003, 0.05),
            'prediction_confidence': trial.suggest_float('prediction_confidence', 0.52, 0.90)
        }
        
        model, scaler = self.trainer.train(train_data.copy(), all_params)
        if model is None: return -2.0

        test_features = self.trainer._prepare_features(test_data.copy())
        if test_features.empty: return -2.0
        
        strategy_params = {k: all_params[k] for k in ['profit_threshold', 'stop_loss_threshold', 'prediction_confidence']}
        strategy_params['risk_per_trade_pct'] = RISK_PER_TRADE_PCT
        
        capital, sharpe_ratio = run_backtest(
            model=model, scaler=scaler, test_data_with_features=test_features,
            strategy_params=strategy_params, feature_names=self.trainer.feature_names
        )
        
        if sharpe_ratio == 0 and capital == 100.0:
            return -0.1

        return sharpe_ratio

    def run(self):
        logger.info("="*80 + "\n--- INICIANDO OTIMIZAÇÃO WALK-FORWARD (ESTRATÉGIA HÍBRIDA) ---\n" + "="*80)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.full_data.sort_index(inplace=True)

        n_total = len(self.full_data)
        train_size, test_size, step_size = WFO_TRAIN_MINUTES, WFO_TEST_MINUTES, WFO_STEP_MINUTES
        if n_total < train_size + test_size:
            return logger.error(f"Dados insuficientes. Necessário: {train_size + test_size}, Disponível: {n_total}")

        total_cycles = math.floor((n_total - train_size - test_size) / step_size) + 1
        logger.info(f"Otimização será executada em aproximadamente {total_cycles} ciclos.")

        start_index, cycle, all_results, cumulative_capital = self._load_wfo_state()

        while start_index + train_size + test_size <= n_total:
            if self.shutdown_requested: break
            
            train_data = self.full_data.iloc[start_index : start_index + train_size]
            test_data = self.full_data.iloc[start_index + train_size : start_index + train_size + test_size]

            logger.info("\n" + "-"*80)
            logger.info(f"INICIANDO CICLO DE WFO #{cycle}/{total_cycles} ({(cycle/total_cycles)*100:.1f}%)")
            logger.info(f"  - Período de Treino: {train_data.index.min():%Y-%m-%d} a {train_data.index.max():%Y-%m-%d}")
            logger.info(f"  - Período de Teste:  {test_data.index.min():%Y-%m-%d} a {test_data.index.max():%Y-%m-%d}")
            
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
            study.optimize(lambda trial: self._objective(trial, train_data, test_data), n_trials=self.n_trials_for_cycle, n_jobs=-1, callbacks=[self._progress_callback])
            print()

            best_trial = study.best_trial
            logger.info(f"\n  - Otimização do ciclo #{cycle} concluída. Melhor Sharpe: {best_trial.value:.4f}")

            if best_trial.value > 0.1:
                logger.info("  - Modelo promissor encontrado! Treinando e salvando modelo final do ciclo...")
                final_model, final_scaler = self.trainer.train(train_data.copy(), best_trial.params)
                
                if final_model:
                    self.trainer.save_model(final_model, final_scaler)
                    strategy_params = {k: best_trial.params[k] for k in ['profit_threshold', 'stop_loss_threshold', 'prediction_confidence']}
                    with open(STRATEGY_PARAMS_FILE, 'w') as f: json.dump(strategy_params, f, indent=4)
                    
                    logger.info("  - Validando performance no período de teste...")
                    # No backtest final, preparamos as features novamente para garantir consistência
                    test_features_final = self.trainer._prepare_features(test_data.copy())
                    capital, sharpe = run_backtest(final_model, final_scaler, test_features_final, strategy_params, self.trainer.feature_names)
                    result_pct = (capital - 100) / 100.0
                    
                    new_cumulative_capital = cumulative_capital * (1 + result_pct)
                    logger.info(f"  - RESULTADO DO CICLO: {result_pct:+.2%} | Sharpe: {sharpe:.2f} | Capital Acumulado: ${new_cumulative_capital:,.2f}")
                    cumulative_capital = new_cumulative_capital
                else:
                    logger.error("  - Falha ao treinar o modelo final do ciclo.")
            else:
                logger.warning(f"  - Melhor resultado do ciclo não foi promissor (Sharpe <= 0.1). Nenhum modelo salvo para este ciclo.")
            
            start_index += step_size
            cycle += 1
            self._save_wfo_state(cycle, start_index, all_results, cumulative_capital)
            gc.collect()

        logger.info("\n\n" + "="*80 + "\n--- OTIMIZAÇÃO WALK-FORWARD CONCLUÍDA ---\n" + "="*80)