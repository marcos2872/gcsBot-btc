# src/optimizer.py (VERSÃO FINAL COM INTELIGÊNCIA COMPLETA)

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
    STRATEGY_PARAMS_FILE, MODEL_FILE, SCALER_FILE
)
# A importação do RISK_PER_TRADE_PCT do config não é mais necessária, pois ele será otimizado
from src.confidence_manager import AdaptiveConfidenceManager

class WalkForwardOptimizer:
    def __init__(self, full_data):
        self.full_data = full_data
        self.trainer = ModelTrainer()
        self.n_trials_for_cycle = 0
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

    # --- Funções de controle (graceful_shutdown, _save_wfo_state, etc.) ---
    # Nenhuma alteração necessária aqui, elas já estão corretas.
    def graceful_shutdown(self, signum, frame):
        if not self.shutdown_requested:
            logger.warning("\n" + "="*50)
            logger.warning("PARADA SOLICITADA! Finalizando o trial atual...")
            logger.warning("="*50)
            self.shutdown_requested = True
            
    def _save_wfo_state(self, cycle, start_index, all_results, cumulative_capital):
        state = {
            'last_completed_cycle': cycle - 1,
            'next_start_index': start_index,
            'results_so_far': all_results,
            'cumulative_capital': cumulative_capital
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
                    cumulative_capital = state.get('cumulative_capital', 100.0)
                    logger.info("="*50)
                    logger.info(f"Estado de otimização anterior encontrado! Retomando do ciclo #{last_cycle + 1}.")
                    logger.info(f"Capital acumulado até o momento: ${cumulative_capital:,.2f}")
                    logger.info("="*50)
                    return state.get('next_start_index', 0), last_cycle + 1, state.get('results_so_far', []), cumulative_capital
            except Exception as e:
                logger.error(f"Erro ao carregar estado da WFO: {e}. Começando do zero.")
        return 0, 1, [], 100.0

    def _progress_callback(self, study, trial):
        n_trials = self.n_trials_for_cycle
        best_value = 0.0 if study.best_value is None else study.best_value
        print(f"\r    - [Progresso Optuna] Trial {trial.number + 1}/{n_trials} concluído... Melhor Sharpe (Validação): {best_value:.4f}", end="", flush=True)

    # --- A MENTE DO OTIMIZADOR ---
    def _objective(self, trial, train_data, validation_data):
        if self.shutdown_requested: raise optuna.exceptions.TrialPruned()
        
        all_params = {
            # === PARÂMETROS DO MODELO DE MACHINE LEARNING (PREENCHIDOS) ===
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
            
            # === PARÂMETROS DA CRIAÇÃO DE LABELS (O QUE O BOT DEVE APRENDER A PROCURAR) ===
            'future_periods': trial.suggest_int('future_periods', 5, 120),
            'profit_mult': trial.suggest_float('profit_mult', 0.5, 5.0),
            'stop_mult': trial.suggest_float('stop_mult', 0.5, 5.0),
            
            # === PARÂMETROS DA ESTRATÉGIA DE TRADING (COMO O BOT DEVE AGIR) ===
            'profit_threshold': trial.suggest_float('profit_threshold', 0.003, 0.05),
            'stop_loss_threshold': trial.suggest_float('stop_loss_threshold', 0.003, 0.05),
            'initial_confidence': trial.suggest_float('initial_confidence', 0.51, 0.75),
            
            ### NOVO: OTIMIZAÇÃO DOS PARÂMETROS DE INTELIGÊNCIA ADAPTATIVA ###
            # O bot vai aprender qual o melhor risco para o período
            'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.01, 0.10), # De 1% a 10% de risco
            # O bot vai aprender quão rápido ele deve ajustar sua própria confiança
            'confidence_learning_rate': trial.suggest_float('confidence_learning_rate', 0.01, 0.20)
        }
        
        model, scaler = self.trainer.train(train_data.copy(), all_params)
        if model is None: return -2.0

        validation_features = self.trainer._prepare_features(validation_data.copy())
        if validation_features.empty: return -2.0
        
        # Passa todos os parâmetros relevantes para a simulação de backtest
        strategy_params = {
            'profit_threshold': all_params['profit_threshold'],
            'stop_loss_threshold': all_params['stop_loss_threshold'],
            'initial_confidence': all_params['initial_confidence'],
            'risk_per_trade_pct': all_params['risk_per_trade_pct'],
            'confidence_learning_rate': all_params['confidence_learning_rate']
        }
        
        capital, sharpe_ratio = run_backtest(
            model=model, scaler=scaler,
            test_data_with_features=validation_features,
            strategy_params=strategy_params,
            feature_names=self.trainer.feature_names
        )
        return sharpe_ratio

    # --- O LOOP PRINCIPAL WALK-FORWARD ---
    def run(self):
        logger.info("="*80)
        logger.info("--- INICIANDO OTIMIZAÇÃO WALK-FORWARD COM INTELIGÊNCIA COMPLETA ---")
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.full_data.sort_index(inplace=True)
        n_total = len(self.full_data)
        train_val_size, test_size, step_size = WFO_TRAIN_MINUTES, WFO_TEST_MINUTES, WFO_STEP_MINUTES
        if n_total < train_val_size + test_size:
            return logger.error(f"Dados insuficientes. Necessário: {train_val_size + test_size}, Disponível: {n_total}")

        total_cycles = math.floor((n_total - train_val_size - test_size) / step_size) + 1
        start_index, cycle, all_results, cumulative_capital = self._load_wfo_state()

        while start_index + train_val_size + test_size <= n_total:
            if self.shutdown_requested: break

            validation_pct = 0.20
            train_val_data = self.full_data.iloc[start_index : start_index + train_val_size]
            test_data = self.full_data.iloc[start_index + train_val_size : start_index + train_val_size + test_size]
            validation_size = int(len(train_val_data) * validation_pct)
            train_data = train_val_data.iloc[:-validation_size]
            validation_data = train_val_data.iloc[-validation_size:]
            
            logger.info("\n" + "-"*80)
            logger.info(f"INICIANDO CICLO DE WFO #{cycle} / {total_cycles}")
            logger.info(f"  - Período de Treino:      {train_data.index.min():%Y-%m-%d} a {train_data.index.max():%Y-%m-%d}")
            logger.info(f"  - Período de Validação:   {validation_data.index.min():%Y-%m-%d} a {validation_data.index.max():%Y-%m-%d}")
            logger.info(f"  - Período de Teste Final: {test_data.index.min():%Y-%m-%d} a {test_data.index.max():%Y-%m-%d}")

            self.n_trials_for_cycle = 100
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self._objective(trial, train_data, validation_data), n_trials=self.n_trials_for_cycle, n_jobs=-1, callbacks=[self._progress_callback])
            if self.shutdown_requested: break
            
            best_trial = study.best_trial
            logger.info(f"\n  - Otimização do ciclo concluída. Melhor Sharpe na VALIDAÇÃO: {best_trial.value:.4f}")

            if best_trial.value > 0.1:
                logger.info("  - Treinando modelo final do ciclo com os melhores parâmetros...")
                # No treino final, usamos todos os parâmetros encontrados, incluindo os do ML
                final_model, final_scaler = self.trainer.train(train_val_data.copy(), best_trial.params)
                
                if final_model:
                    # Salva apenas os parâmetros da ESTRATÉGIA para o bot de trading usar
                    strategy_params = {
                        'profit_threshold': best_trial.params['profit_threshold'],
                        'stop_loss_threshold': best_trial.params['stop_loss_threshold'],
                        'initial_confidence': best_trial.params['initial_confidence'],
                        'risk_per_trade_pct': best_trial.params['risk_per_trade_pct'],
                        'confidence_learning_rate': best_trial.params['confidence_learning_rate']
                    }
                    self.trainer.save_model(final_model, final_scaler)
                    with open(STRATEGY_PARAMS_FILE, 'w') as f: json.dump(strategy_params, f, indent=4)
                    
                    logger.info("  - Executando backtest final no período de TESTE...")
                    test_features_final = self.trainer._prepare_features(test_data.copy())
                    
                    capital, sharpe = run_backtest(
                        model=final_model, scaler=final_scaler, 
                        test_data_with_features=test_features_final, 
                        strategy_params=strategy_params, 
                        feature_names=self.trainer.feature_names
                    )
                    
                    result_pct = (capital - 100) / 100 if capital > 0 else 0
                    all_results.append({'period': f"{test_data.index.min():%Y-%m-%d}_a_{test_data.index.max():%Y-%m-%d}", 'capital': capital, 'sharpe': sharpe})
                    
                    new_cumulative_capital = cumulative_capital * (1 + result_pct)
                    logger.info("-" * 25 + f" RESULTADO REAL DO CICLO {cycle} " + "-" * 26)
                    logger.info(f"  - Resultado do Período de Teste: {result_pct:+.2%}")
                    logger.info(f"  - Capital Simulado Acumulado: ${cumulative_capital:,.2f} -> ${new_cumulative_capital:,.2f}")
                    logger.info(f"  - Sharpe Ratio (Anualizado): {sharpe:.2f}")
                    cumulative_capital = new_cumulative_capital
                else:
                    logger.error("  - Falha ao treinar o modelo final do ciclo. Pulando.")
            else:
                logger.warning(f"  - Melhor resultado na validação não foi positivo o suficiente. Pulando para o próximo ciclo.")
            
            start_index += step_size
            cycle += 1
            self._save_wfo_state(cycle, start_index, all_results, cumulative_capital)
            gc.collect()

        logger.info("\n\n" + "="*80 + "\n--- OTIMIZAÇÃO WALK-FORWARD CONCLUÍDA ---")