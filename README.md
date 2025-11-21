# Projeto de Credit Scoring - Análise de Risco de Crédito

Este repositório contém uma solução end-to-end de Ciência de Dados para a concessão de crédito. O objetivo principal foi desenvolver uma ferramenta capaz de auxiliar instituições financeiras a prever a probabilidade de inadimplência de um cliente, classificando-o como "Bom" ou "Mau" pagador.

Acesse através do link - https://credit-scoring-ebac.onrender.com

## Sobre o Projeto

A análise de crédito é um processo crítico para bancos e fintechs. Errar na aprovação pode significar prejuízo (calote) ou perda de lucro (negar crédito a um bom pagador).

Neste projeto, desenvolvi um modelo de Machine Learning que analisa dados pessoais, financeiros e demográficos para calcular um "score" de risco. Todo o processo foi automatizado, desde o tratamento dos dados até a disponibilização da interface para o usuário final.

## Tecnologias e Abordagem

Utilizei a linguagem Python (versão 3.10) e foquei em ferramentas de produtividade e performance:

* PyCaret: Utilizado para agilizar a comparação de modelos, tratamento de dados (inputação de nulos, normalização) e tunagem de hiperparâmetros.
* LightGBM: Foi o algoritmo escolhido para o modelo final devido ao seu excelente desempenho em dados tabulares e velocidade de processamento. O modelo alcançou uma métrica AUC de 0.78.
* Streamlit: Escolhido para construir a interface web, permitindo que qualquer pessoa interaja com o modelo sem precisar entender de código.

## Funcionalidades da Aplicação

Ao acessar o aplicativo, o usuário encontra um formulário na barra lateral para inserir informações do cliente, como:
* Renda mensal e tempo de emprego.
* Dados demográficos (idade, estado civil, educação).
* Bens (posse de veículo ou imóvel).

O sistema processa esses dados em tempo real e retorna:
1.  A decisão sugerida (Aprovar ou Negar Crédito).
2.  Uma barra de progresso visual indicando a probabilidade exata calculada pelo modelo.

## Como executar este projeto localmente

Para rodar a aplicação no seu computador, siga as instruções abaixo. Recomendo o uso de um ambiente virtual (como Anaconda) para garantir a compatibilidade das versões.

1. Clone este repositório ou baixe os arquivos.

2. Crie um ambiente virtual com Python 3.10 (essencial para compatibilidade do PyCaret):
   conda create -n credit-scoring python=3.10
   conda activate credit-scoring

3. Instale as dependências listadas no arquivo requirements.txt:
   pip install -r requirements.txt

4. Execute o aplicativo através do Streamlit:
   streamlit run app_pycaret.py

## Estrutura dos Arquivos

* app_pycaret.py: O código principal da aplicação web. Contém a lógica da interface e as chamadas para o modelo.
* modelo_credit_scoring_pycaret.pkl: O arquivo binário do modelo treinado. Ele contém não apenas o algoritmo matemático, mas todo o pipeline de pré-processamento dos dados.
* requirements.txt: Lista de bibliotecas necessárias para que o projeto rode em qualquer máquina ou servidor.

---

Lucas Ponte e Silva
