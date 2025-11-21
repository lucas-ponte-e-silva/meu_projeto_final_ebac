import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import os

# 1. Configuração da Página
st.set_page_config(page_title="Credit Scoring App", layout="centered")

# 2. Carregar o Modelo
# Usa caminho GitHub
@st.cache_resource
def carregar_modelo():
    try:
        return load_model('modelo_credit_scoring_pycaret')
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Verifique se o arquivo .pkl está na pasta.")
        st.error(f"Detalhe do erro: {e}")
        return None

model = carregar_modelo()

# 3. Título e Descrição
st.title("📊 Predição de Credit Scoring")
st.markdown("Simulação de risco de crédito utilizando Machine Learning (PyCaret). Preencha os dados na barra lateral para realizar uma análise.")

if model is None:
    st.warning("⚠️ Aguardando carregamento do modelo...")
    st.stop()

# 4. Formulário de Entrada (Barra Lateral)
st.sidebar.header("Perfil do Cliente")

def get_user_input():
    # Dados Demográficos
    sexo = st.sidebar.selectbox("Sexo", ['M', 'F'])
    idade = st.sidebar.slider("Idade", 18, 80, 30)
    estado_civil = st.sidebar.selectbox("Estado Civil", ['Solteiro', 'Casado', 'Viúvo', 'Separado', 'União'])
    educacao = st.sidebar.selectbox("Escolaridade", ['Fundamental', 'Médio', 'Superior incompleto', 'Superior completo', 'Pós graduação'])
    
    st.sidebar.markdown("---")
    
    # Dados Financeiros e Patrimoniais
    renda = st.sidebar.number_input("Renda Mensal (R$)", min_value=0.0, value=5000.0, step=100.0)
    tipo_renda = st.sidebar.selectbox("Fonte de Renda", ['Assalariado', 'Empresário', 'Pensionista', 'Servidor público', 'Bolsista'])
    tempo_emprego = st.sidebar.slider("Tempo de Emprego (anos)", 0.0, 40.0, 2.0)
    
    posse_de_veiculo = st.sidebar.selectbox("Possui Veículo?", ['S', 'N'])
    posse_de_imovel = st.sidebar.selectbox("Possui Imóvel Próprio?", ['S', 'N'])
    
    st.sidebar.markdown("---")
    
    # Dados Residenciais
    tipo_residencia = st.sidebar.selectbox("Tipo de Moradia", ['Casa', 'Com os pais', 'Governamental', 'Aluguel', 'Estúdio', 'Comunitário'])
    # step=1 garante número inteiro
    qt_pessoas_residencia = st.sidebar.number_input("Pessoas na Residência", min_value=1, max_value=15, value=1, step=1)
    qtd_filhos = st.sidebar.number_input("Quantidade de Filhos", min_value=0, max_value=15, value=0, step=1)
    
    # Dicionário de dados para o modelo
    data = {
        'sexo': sexo,
        'posse_de_veiculo': posse_de_veiculo,
        'posse_de_imovel': posse_de_imovel,
        'qtd_filhos': qtd_filhos,
        'tipo_renda': tipo_renda,
        'educacao': educacao,
        'estado_civil': estado_civil,
        'tipo_residencia': tipo_residencia,
        'idade': idade,
        'tempo_emprego': tempo_emprego,
        'renda': renda,
        'qt_pessoas_residencia': qt_pessoas_residencia
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Exibe um resumo dos dados preenchidos
with st.expander("Ver dados selecionados", expanded=False):
    st.dataframe(input_df)

# 5. Botão de Predição
if st.button('📊 Calcular Risco', use_container_width=True):
    try:
        # O modelo faz o pré-processamento automático
        prediction = predict_model(model, data=input_df)
        
        # Extraindo os resultados
        classe_predita = prediction['prediction_label'].iloc[0]
        score = prediction['prediction_score'].iloc[0]
        
        st.divider()
        
        # Lógica de Resultado (Assumindo 1 = Mau Pagador)
        if classe_predita == 1:
            st.error("🚨 **Resultado: Alto Risco (Crédito Negado)**")
            st.write(f"Probabilidade de Inadimplência: **{score:.2%}**")
            st.progress(int(score * 100), text="Risco Alto")
        else:
            st.success("✅ **Resultado: Baixo Risco (Crédito Aprovado)**")
            # Se a classe é 0 (Bom), o score é a certeza de ser Bom.
            # Risco = 1 - certeza
            risco = 1 - score
            st.write(f"Score de Confiança: **{score:.2%}**")
            st.progress(int(score * 100), text="Segurança do Crédito")
            
    except Exception as e:
        st.error("Ocorreu um erro ao processar a previsão.")
        st.write("Detalhes do erro:", e)

