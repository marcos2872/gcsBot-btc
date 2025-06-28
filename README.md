# 📈 gcsBot - Bot de Trading para BTC/USDT com Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

Um bot de trading algorítmico de nível profissional para o par BTC/USDT na Binance, que utiliza técnicas avançadas de Machine Learning e gestão de portfólio dinâmica para otimizar estratégias e operar de forma autônoma.

---

## 📋 Tabela de Conteúdos

- [Sobre o Projeto](#-sobre-o-projeto)
- [✨ Core Features](#-core-features)
- [🧠 Como o Bot "Pensa"? (A Estratégia)](#-como-o-bot-pensa-a-estratégia)
- [⚙️ O Ecossistema do Bot: Como os Módulos Interagem](#️-o-ecossistema-do-bot-como-os-módulos-interagem)
- [🚀 Começando](#-começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [🔧 Configuração do Ambiente (`.env`)](#-configuração-do-ambiente-env)
- [▶️ Como Usar (Workflow Profissional)](#️-como-usar-workflow-profissional)
  - [Fase 1: Otimização (`MODE=optimize`)](#fase-1-otimização-modeoptimize)
  - [Fase 2: Validação (`MODE=test`)](#fase-2-validação-modetest)
  - [Fase 3: Produção (`MODE=trade`)](#fase-3-produção-modetrade)
  - [Comandos Adicionais](#comandos-adicionais)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [📜 Licença](#-licença)

---

## 🤖 Sobre o Projeto

Este não é um bot de trading comum. Ele foi projetado para tomar decisões baseadas em dados e estatísticas, não em regras fixas. O sistema utiliza um pipeline completo de Machine Learning e um gerenciador de portfólio para:

1.  **Aprender** com um vasto histórico de dados de mercado para prever oportunidades.
2.  **Gerenciar o Risco** de forma dinâmica, ajustando o tamanho de cada operação com base no capital disponível.
3.  **Otimizar** seus próprios parâmetros através de um processo robusto de Walk-Forward Optimization (WFO).
4.  **Operar** de forma autônoma nos ambientes de Teste (Testnet) ou Produção (Conta Real) da Binance.

O objetivo é encontrar e explorar ineficiências no mercado, combinando análise técnica e macroeconômica, sempre sob uma camada de gestão de capital disciplinada.

---

## ✨ Core Features

- **🧠 Modelo Preditivo (LightGBM):** Utiliza um modelo de Gradient Boosting rápido e eficiente.
- **💼 Gestão de Portfólio Dinâmica:** Gerencia o capital de forma inteligente, separando fundos para holding e para trading, com cálculo de risco dinâmico por operação.
- **🔍 Otimização de Hiperparâmetros (Optuna):** Encontra a melhor combinação de parâmetros para o modelo e para a estratégia.
- **🛡️ Walk-Forward Optimization (WFO):** A metodologia de backtesting mais robusta, que simula o desempenho do bot em condições de mercado dinâmicas.
- **ρεαλισμός Backtest Realista:** A simulação de backtest inclui **custos operacionais** (taxas e slippage) e é livre de **look-ahead bias**, garantindo que os resultados da otimização sejam honestos e representativos do mundo real.
- **💵 Correlação com DXY:** Incorpora a variação do Índice do Dólar (DXY) como uma feature para contexto macroeconômico.
- **🐳 Deployment com Docker:** Empacotado em um container Docker para um deployment fácil, portátil e robusto.
- **▶️ Orquestrador Inteligente (`run.py`):** Um ponto de entrada único que gerencia todo o ciclo de vida do bot.
- **📝 Logging Detalhado:** Sistema de logs inteligente que registra não apenas os trades, mas o estado completo do portfólio.

---

## 🧠 Como o Bot "Pensa"? (A Estratégia)

O bot é um especialista em encontrar **padrões numéricos** nos dados de mercado. Ele analisa uma combinação de "impressões digitais" (features) para tomar uma decisão. Quando o modelo encontra um padrão com alta probabilidade estatística de sucesso, ele passa a decisão para o **Gerenciador de Portfólio**, que calcula o tamanho exato da posição com base nas regras de risco definidas, garantindo que nenhuma operação individual possa comprometer o capital total.

---

## ⚙️ O Ecossistema do Bot: Como os Módulos Interagem

O bot opera em dois "modos mentais" principais, utilizando diferentes combinações de arquivos.

#### Modo de Otimização (`optimize`)

Neste modo, o bot está em seu "laboratório de pesquisa". Ele não opera no mercado real.

- **`optimizer.py`**: É o cérebro da operação. Ele gerencia o processo de Walk-Forward.
- **`model_trainer.py`**: É chamado pelo otimizador para treinar um novo modelo a cada ciclo, usando as features realistas (sem olhar para o futuro).
- **`backtest.py`**: É a peça-chave. Para cada modelo treinado, ele executa uma simulação **realista** nos dados de teste, calculando a performance com taxas e slippage. O resultado (Sharpe Ratio) é devolvido ao otimizador.
- **Resultado Final:** A criação dos arquivos `trading_model.pkl`, `scaler.pkl` e `strategy_params.json` na pasta `/data`.

#### Modos de Operação (`test` e `trade`)

Neste modo, o bot está "em campo", operando no mercado ao vivo.

- **`trading_bot.py`**: É o único módulo ativo. Ele é o piloto.
- **Arquivos de Inteligência**: Ele carrega os arquivos `.pkl` e `.json` gerados pela otimização para saber _como_ e _quando_ operar.
- **`PortfolioManager`**: Uma classe dentro do `trading_bot.py` que gerencia ativamente o capital, calcula o tamanho das posições com base no risco e protege a carteira.
- **Conexão com a Binance**: Ele usa as chaves de API definidas no `.env` para enviar ordens reais (para a Testnet no modo `test`, ou para a conta real no modo `trade`).

---

## 🚀 Começando

Siga estes passos para colocar o bot em funcionamento.

### Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/giakomogcs/gcsbot-btc.git](https://github.com/giakomogcs/gcsbot-btc.git)
    cd gcsbot-btc
    ```
2.  **Execute o Setup Automático:**
    ```bash
    python run.py setup
    ```
    ⚠️ **Atenção:** Este comando criará um arquivo `.env`. **Você deve abri-lo e preencher todas as variáveis necessárias.**
3.  **Construa a Imagem Docker:**
    ```bash
    python run.py build
    ```

---

## 🔧 Configuração do Ambiente (`.env`)

O arquivo `.env` é o painel de controle principal do bot.

- **`MODE`**: Define o modo de operação. Use `optimize`, `test`, ou `trade`.
- **`SYMBOL`**: O par de moedas a ser operado (ex: `BTCUSDT`).

#### Chaves de API

- `BINANCE_API_KEY` & `BINANCE_API_SECRET`: Suas chaves da conta **real**.
- `BINANCE_TESTNET_API_KEY` & `BINANCE_TESTNET_API_SECRET`: Suas chaves da conta **Testnet**.

#### Gestão de Portfólio

- `MAX_USDT_ALLOCATION`: O **MÁXIMO** de capital em USDT que o bot tem permissão para gerenciar. Ele usará o menor valor entre este e o seu saldo real na Binance.
- `LONG_TERM_HOLD_PCT`: Percentual do capital que será usado para comprar e manter BTC como holding de longo prazo (o bot não vende essa parte). Ex: `0.50` para 50%.
- `RISK_PER_TRADE_PCT`: Do capital de **trading** restante, qual a porcentagem de risco por operação? Ex: `0.02` para arriscar 2% em cada operação.

> ⚠️ **Nunca compartilhe ou envie seu arquivo `.env` para repositórios públicos!**

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

### Fase 2: Validação em Testnet

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
│   ├── backtest.py        # Módulo de backtesting realista com custos
│   ├── config.py          # Carrega e gerencia as configurações
│   ├── data_manager.py    # Gerencia a coleta e atualização de dados
│   ├── logger.py          # Configuração do sistema de logs
│   ├── model_trainer.py   # Prepara features e treina o modelo de ML
│   ├── optimizer.py       # Orquestra o Walk-Forward Optimization
│   └── trading_bot.py     # Lógica de operação e gestão de portfólio
├── .dockerignore          # Arquivos a serem ignorados pelo Docker
├── .env.example           # Exemplo do arquivo de configuração
├── .gitignore             # Arquivos a serem ignorados pelo Git
├── Dockerfile             # Define o ambiente Docker para o bot
├── main.py                # Ponto de entrada legado (usado pelo Docker)
├── README.md              # Esta documentação
├── requirements.txt       # Dependências Python
└── run.py                 # Orquestrador principal e ponto de entrada do usuário
```
