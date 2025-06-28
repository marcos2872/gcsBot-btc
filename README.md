# 📈 gcsBot - Bot de Trading para BTC/USDT com Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

Um bot de trading algorítmico de nível profissional para o par BTC/USDT na Binance, que utiliza técnicas avançadas de Machine Learning para otimizar estratégias e operar de forma autônoma.

---

## 📋 Tabela de Conteúdos

- [Sobre o Projeto](#-sobre-o-projeto)
- [✨ Core Features](#-core-features)
- [🧠 Como o Bot "Pensa"? (A Estratégia)](#-como-o-bot-pensa-a-estratégia)
- [🚀 Começando](#-começando)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [⚙️ Configuração](#️-configuração)
- [▶️ Como Usar (Workflow Profissional)](#️-como-usar-workflow-profissional)
  - [Fase 1: Otimização](#fase-1-otimização)
  - [Fase 2: Validação em Testnet](#fase-2-validação-em-testnet)
  - [Fase 3: Trading Real](#fase-3-trading-real)
  - [Comandos Adicionais](#comandos-adicionais)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)

---

## 🤖 Sobre o Projeto

Este não é um bot de trading comum. Ele foi projetado desde o início para tomar decisões baseadas em dados e estatísticas, não em regras fixas. O sistema utiliza um pipeline completo de Machine Learning para:

1.  **Aprender** com um vasto histórico de dados de mercado.
2.  **Otimizar** seus próprios parâmetros através de um processo robusto de Walk-Forward Optimization (WFO).
3.  **Operar** de forma autônoma nos ambientes de Teste (Testnet) ou Produção (Conta Real) da Binance.

O objetivo é encontrar e explorar ineficiências e padrões no mercado, combinando indicadores técnicos clássicos com dados macroeconômicos, como a força do dólar (DXY).

---

## ✨ Core Features

- **🧠 Modelo Preditivo (LightGBM):** Utiliza um modelo de Gradient Boosting rápido e eficiente para prever a direção do mercado.
- **🔍 Otimização de Hiperparâmetros (Optuna):** Encontra a melhor combinação de parâmetros para o modelo e para a estratégia de forma automática.
- **🛡️ Walk-Forward Optimization (WFO):** A metodologia de backtesting mais robusta, que simula o desempenho do bot em condições de mercado dinâmicas, retreinando o modelo periodicamente.
- **🎯 Labeling com Barreira Tripla:** Utiliza a metodologia profissional "Triple-Barrier" para ensinar o modelo, criando alvos de lucro e prejuízo dinâmicos baseados na volatilidade do mercado (ATR).
- **💵 Correlação com DXY:** Incorpora a variação do Índice do Dólar (DXY) como uma feature, permitindo que o modelo aprenda sobre o contexto macroeconômico.
- **🐳 Deployment com Docker:** Empacotado em um container Docker para um deployment fácil, portátil e robusto (com reinicialização automática).
- **▶️ Orquestrador Inteligente (run.py):** Um ponto de entrada único que gerencia o setup, build, otimização e execução do bot, simplificando o fluxo de trabalho.
- **📝 Logging Detalhado:** Sistema de logs inteligente que separa informações por nível e modo de operação, facilitando a análise financeira e a depuração.

---

## 🧠 Como o Bot "Pensa"? (A Estratégia)

O bot não "sabe" sobre notícias ou eventos geopolíticos. Em vez disso, ele é um especialista em encontrar **padrões numéricos** que esses eventos deixam nos dados de mercado.

Ele analisa uma combinação de "impressões digitais" para tomar uma decisão:

| Categoria da Pista          | Features Utilizadas                  | O que o Bot "Vê"?                                                                    |
| --------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------ |
| **Tendência do Mercado**    | SMA, MACD                            | "O preço está em uma tendência de alta ou de baixa no curto/médio prazo?"            |
| **Volatilidade**            | ATR, Largura das Bandas de Bollinger | "O mercado está calmo e previsível ou agitado e perigoso?"                           |
| **Força do Movimento**      | RSI, Oscilador Estocástico           | "Esta alta está perdendo força e prestes a reverter? Esta baixa já chegou ao fundo?" |
| **Contexto Macroeconômico** | Variação do DXY                      | "O que o dólar está fazendo? Historicamente, isso afeta o BTC de que forma?"         |

O processo de **otimização** ensina ao modelo qual combinação dessas pistas leva a um resultado lucrativo, usando o método de Barreira Tripla como gabarito. Ao final, o bot opera com base em pura probabilidade e estatística.

---

## 🚀 Começando

Siga estes passos para colocar o bot em funcionamento na sua máquina local.

### Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Instalação

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/giakomogcs/gcsbot-btc.git
    cd gcsbot-btc
    ```

2.  **Execute o Setup Automático:**
    Nosso orquestrador `run.py` fará o resto. Ele vai criar o arquivo de configuração e instalar as dependências.

    ```bash
    python run.py setup
    ```

    ⚠️ **Atenção:** Este comando criará um arquivo `.env`. **Você deve abri-lo e preencher suas chaves de API da Binance (Real e Testnet).**

3.  **Construa a Imagem Docker:**
    Este comando empacota o bot e suas dependências em um ambiente pronto para ser executado.

    ```bash
    python run.py build
    ```

---

## ⚙️ Configuração

Toda a configuração do bot é gerenciada através do arquivo `.env`.

- `BINANCE_API_KEY`: Sua chave de API da conta **real** da Binance.
- `BINANCE_API_SECRET`: Seu segredo de API da conta **real** da Binance.
- `BINANCE_TESTNET_API_KEY`: Sua chave de API da conta **Testnet**.
- `BINANCE_TESTNET_API_SECRET`: Seu segredo de API da conta **Testnet**.
- `SYMBOL`: O par de moedas a ser operado (padrão: BTCUSDT).
- `TRADE_AMOUNT_USDT`: O valor em USDT para cada operação (padrão: 100.0).

> ⚠️ **Nunca compartilhe ou envie seu arquivo `.env` para repositórios públicos!** Ele já está incluído no `.gitignore` para sua segurança.

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
│   ├── config.py          # Carrega e gerencia as configurações
│   ├── data_manager.py    # Gerencia a coleta e atualização de dados
│   ├── logger.py          # Configuração do sistema de logs
│   ├── model_trainer.py   # Prepara features e treina o modelo de ML
│   ├── optimizer.py       # Orquestra o Walk-Forward Optimization
│   └── trading_bot.py     # Lógica de operação em tempo real
├── .dockerignore          # Arquivos a serem ignorados pelo Docker
├── .env.example           # Exemplo do arquivo de configuração
├── .gitignore             # Arquivos a serem ignorados pelo Git
├── Dockerfile             # Define o ambiente Docker para o bot
├── main.py                # Ponto de entrada legado (agora usado pelo Docker)
├── README.md              # Esta documentação
├── requirements.txt       # Dependências Python
└── run.py                 # Orquestrador principal e ponto de entrada do usuário
```
