# 📈 gcsBot - Framework de Trading Quantitativo para BTC/USDT

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker) ![License](https://img.shields.io/badge/License-MIT-green.svg)

Um framework de ponta para pesquisa, validação e execução de estratégias de trading algorítmico no par BTC/USDT. Este projeto vai além de um simples bot, oferecendo um pipeline completo de Machine Learning, desde a otimização de estratégias com dados históricos até a operação autônoma e adaptativa na Binance.

---

## 📋 Tabela de Conteúdos

- [🌟 Sobre o Projeto](#-sobre-o-projeto)
- [✨ Features de Destaque](#-features-de-destaque)
- [🧠 A Filosofia do Bot: Como Ele Pensa?](#-a-filosofia-do-bot-como-ele-pensa)
- [⚙️ Ecossistema do Bot: Como os Módulos Interagem](#️-ecossistema-do-bot-como-os-módulos-interagem)
- [🚀 Guia de Início Rápido](#-guia-de-início-rápido)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [🔧 Configuração do Ambiente (`.env`)](#-configuração-do-ambiente-env)
- [▶️ O Workflow Profissional: Como Usar](#️-o-workflow-profissional-como-usar)
  - [Fase 1: Pesquisa e Otimização (`optimize`)](#fase-1-pesquisa-e-otimização-optimize)
  - [Fase 2: Validação Fora da Amostra (`backtest`)](#fase-2-validação-fora-da-amostra-backtest)
  - [Fase 3: Operação em Ambiente de Teste (`test`)](#fase-3-operação-em-ambiente-de-teste-test)
  - [Fase 4: Operação em Produção (`trade`)](#fase-4-operação-em-produção-trade)
  - [Comandos de Gerenciamento](#comandos-de-gerenciamento)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [📜 Licença](#-licença)

---

## 🌟 Sobre o Projeto

Este repositório contém um sistema de trading algorítmico completo, projetado para ser robusto, inteligente e metodologicamente correto. Diferente de bots baseados em regras fixas, o gcsBot utiliza um modelo de **Machine Learning (LightGBM)** para encontrar padrões preditivos e uma arquitetura sofisticada para se adaptar às dinâmicas do mercado.

O núcleo do projeto é um processo de **Walk-Forward Optimization (WFO)** que garante que a estratégia seja constantemente reavaliada e otimizada em dados novos, evitando o overfitting e a estagnação. O resultado é um agente autônomo que não apenas opera, mas aprende e se ajusta.

---

## ✨ Features de Destaque

- **🧠 Inteligência Autoadaptativa:**

  - **Confiança Dinâmica:** O bot ajusta sua própria "coragem" (`prediction_confidence`) com base em seus lucros e prejuízos recentes, tornando-se mais ousado em sequências de vitórias e mais cauteloso após perdas.
  - **Risco Dinâmico (Bet Sizing):** O tamanho de cada operação é proporcional à convicção do modelo no sinal, arriscando mais em oportunidades de alta probabilidade.
  - **Otimização da Personalidade:** O sistema utiliza `Optuna` para encontrar não apenas os melhores parâmetros de modelo, mas a melhor "personalidade" para o bot, incluindo seu apetite de risco e velocidade de aprendizado.

- **🤖 Metodologia de Nível Profissional:**

  - **Validação Robusta (Train/Validate/Test):** O processo de otimização utiliza uma metodologia rigorosa que impede o vazamento de dados do futuro (_look-ahead bias_), garantindo que os resultados dos testes sejam honestos.
  - **Rotulagem de 3 Classes (Buy/Sell/Hold):** O modelo aprende a identificar mercados laterais e a ficar de fora, reduzindo trades desnecessários e focando em sinais de alta qualidade.
  - **Análise de Regime de Mercado:** O bot utiliza features de longo prazo (`SMA200`, `ATR`) para entender o contexto do mercado (tendência vs. lateralidade, alta vs. baixa volatilidade) antes de tomar decisões.

- **⚙️ Engenharia de Ponta:**
  - **Backtest Realista:** Todas as simulações incluem custos operacionais (taxas de `0.1%` e derrapagem de `0.05%`) para uma avaliação de performance fiel à realidade.
  - **Integração de Dados Macroeconômicos:** Utiliza a variação de indicadores como DXY (dólar), VIX (volatilidade), Ouro e Títulos de 10 anos para um contexto de mercado mais rico.
  - **Deployment com Docker:** Ambiente 100% conteinerizado para uma execução consistente e livre de problemas de dependências.
  - **Cache Inteligente e Modo Offline:** Processa e armazena dados para inicializações futuras quase instantâneas e permite rodar o modo de otimização completamente offline.

---

## 🧠 A Filosofia do Bot: Como Ele Pensa?

A tomada de decisão do gcsBot segue uma hierarquia de inteligência em quatro etapas:

1.  **Contexto (O Cenário):** Primeiro, o bot analisa o **regime de mercado**. "Estamos em uma tendência de alta ou de baixa? A volatilidade está alta ou baixa?" Isso é feito através das features de regime (`regime_tendencia`, `regime_volatilidade`).

2.  **Sinal (A Oportunidade):** Dentro desse contexto, o modelo de Machine Learning busca por um **padrão preditivo** de curto prazo, uma ineficiência que sugira uma oportunidade de compra.

3.  **Convicção (A Coragem):** Uma vez que um sinal é encontrado, o bot consulta seu **nível de confiança adaptativo**. "Baseado na minha performance recente, eu deveria arriscar neste sinal ou é melhor ter paciência?"

4.  **Ação (O Tamanho da Posição):** Se a convicção for alta o suficiente, o bot calcula o **tamanho do risco** a ser tomado, proporcional à força do sinal. Um sinal "ok" recebe uma alocação pequena; um sinal "perfeito" recebe uma alocação maior.

Este processo transforma o bot de um simples executor de regras em um agente estratégico que pensa em múltiplas camadas.

---

## ⚙️ Ecossistema do Bot: Como os Módulos Interagem

- **`optimizer.py`**: O cérebro da pesquisa. Gerencia o WFO, chama o `model_trainer` e o `backtest`, e usa o `Optuna` para encontrar os melhores parâmetros.
- **`model_trainer.py`**: O "cientista de dados". Prepara todas as features (técnicas, macro e de regime) e treina o modelo LightGBM.
- **`confidence_manager.py`**: O "psicólogo" do bot. Implementa a lógica para ajustar a confiança com base nos resultados.
- **`backtest.py`**: O simulador. Executa a estratégia de forma realista, utilizando o `confidence_manager` para testar o desempenho da estratégia adaptativa.
- **`quick_tester.py`**: O "auditor". Permite validar um modelo já treinado em um período de tempo futuro completamente novo.
- **`trading_bot.py`**: O "piloto". Módulo que opera no mercado real, utilizando os artefatos (`.pkl`, `.json`) gerados pela otimização.

---

## 🚀 Guia de Início Rápido

Siga estes passos para colocar o bot em funcionamento.

### Pré-requisitos

- [Python 3.11+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (em execução)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/gcsbot-btc.git](https://github.com/SEU_USUARIO/gcsbot-btc.git)
    cd gcsbot-btc
    ```
2.  **Execute o Setup Automático:**
    Este comando irá verificar o ambiente, instalar as dependências e criar o seu arquivo de configuração `.env`.

    ```bash
    python run.py setup
    ```

    > ⚠️ **Atenção:** Após o setup, abra o arquivo `.env` recém-criado e preencha **todas** as variáveis, especialmente suas chaves de API.

3.  **Construa a Imagem Docker:**
    ```bash
    python run.py build
    ```

---

## 🔧 Configuração do Ambiente (`.env`)

O arquivo `.env` é o painel de controle principal do bot.

- **`MODE`**: Modo de operação: `optimize`, `backtest`, `test`, ou `trade`.
- **`FORCE_OFFLINE_MODE`**: `True` ou `False`. Impede o bot de acessar a internet (útil para otimizações).

#### Chaves de API

- `BINANCE_API_KEY` & `BINANCE_API_SECRET`: Chaves da conta **real**.
- `BINANCE_TESTNET_API_KEY` & `BINANCE_TESTNET_API_SECRET`: Chaves da conta **Testnet**.

#### Gestão de Portfólio (Para os modos `test` e `trade`)

- `MAX_USDT_ALLOCATION`: O **MÁXIMO** de capital em USDT que o bot tem permissão para gerenciar.
- `LONG_TERM_HOLD_PCT`: Percentual do capital para holding de longo prazo (ex: `0.50` para 50%).

#### Parâmetros de Backtest e Fallback

- `RISK_PER_TRADE_PCT`: **Fallback** do risco por operação, caso o valor não seja encontrado nos parâmetros otimizados.
- `BACKTEST_START_DATE` & `BACKTEST_END_DATE`: Período para a simulação do modo `backtest`.

> ⚠️ **NUNCA** envie seu arquivo `.env` para repositórios públicos! O `.gitignore` já está configurado para ignorá-lo.

---

## ▶️ Como Usar (Workflow Profissional)

A interação com o bot é feita através do orquestrador `run.py`. Siga estas fases na ordem correta.

### Fase 1: Otimização

O passo mais importante. O bot irá estudar todo o histórico para encontrar a melhor estratégia e criar os arquivos de modelo.

```bash
python run.py optimize
```

Este processo é longo e pode levar horas ou dias. Ao final, os arquivos `trading_model.pkl`, `scaler.pkl` e `strategy_params.json` serão salvos na pasta /data.

---

### Fase 2: Backtest Rápido

Após a otimização, valide a estratégia no mercado ao vivo com dinheiro de teste.

```bash
python run.py backtest
```

O bot irá rodar a simulação no período definido no .env e imprimir um relatório de performance mês a mês no terminal.

---

### Fase 3: Validação em Testnet

Após a otimização, valide a estratégia no mercado ao vivo com dinheiro de teste.

```bash
python run.py test
```

O bot iniciará em segundo plano e rodará 24/7. Ele usará o modelo e os parâmetros criados na Fase 1. Deixe rodando por pelo menos 1-2 semanas para obter dados estatísticos relevantes.

---

### Fase 3: Trading Real

Após a otimização, valide a estratégia no mercado ao vivo com dinheiro de teste.

```bash
python run.py trade
```

O bot operará da mesma forma que no modo test, mas utilizando sua conta real da Binance.

---

## Comandos Adicionais

- Ver os Logs em Tempo Real:

```bash
python run.py logs
```

- Parar o Bot (Modo `test` ou `trade`):

```bash
python run.py stop
```

---

# 📂 Estrutura do Projeto

```bash
gcsbot-btc/
├── data/                  # Dados gerados (CSVs, modelos, estados) - Ignorado pelo Git
├── logs/                  # Arquivos de log diários - Ignorado pelo Git
├── src/                   # Código fonte do projeto
│   ├── __init__.py
│   ├── backtest.py        # Motor de simulação realista (usado pela otimização)
│   ├── config.py          # Gerenciador de configurações do .env
│   ├── confidence_manager.py # Cérebro da confiança adaptativa
│   ├── data_manager.py    # Gerenciador de coleta e cache de dados
│   ├── logger.py          # Configuração do sistema de logs
│   ├── model_trainer.py   # Prepara features e treina o modelo de ML
│   ├── optimizer.py       # Orquestrador do Walk-Forward Optimization (WFO)
│   ├── quick_tester.py    # Lógica para o modo de backtest rápido (validação)
│   └── trading_bot.py     # Lógica de operação real e gestão de portfólio
├── .dockerignore          # Arquivos a serem ignorados pelo Docker
├── .env.example           # Exemplo do arquivo de configuração
├── .gitignore             # Arquivos a serem ignorados pelo Git
├── Dockerfile             # Define o ambiente Docker para o bot
├── main.py                # Ponto de entrada principal (usado pelo Docker)
├── README.md              # Esta documentação
├── requirements.txt       # Dependências Python
└── run.py                 # Orquestrador principal e ponto de entrada do usuário
```
