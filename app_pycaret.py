import streamlit as st
import pandas as pd
import os
from pycaret.classification import load_model, predict_model

# 1. Configuração da Página
st.set_page_config(page_title="Credit Scoring EBAC", layout="centered")

# --- CONFIGURAÇÃO DO CAMINHO DO ARQUIVO ---
# Usamos 'r' antes das aspas para indicar uma string "crua" (raw string),
# ignorando caracteres de escape do Windows.
CAMINHO_PASTA = r'C:\Users\ottop\OneDrive\Área de Trabalho\projeto_final_ebac'
NOME_ARQUIVO_MODELO = 'modelo_credit_scoring_pycaret' # Sem o .pkl, o PyCaret adiciona sozinho

# Junta a pasta com o nome do arquivo de forma segura
caminho_completo = os.path.join(CAMINHO_PASTA, NOME_ARQUIVO_MODELO)

# 2. Carregar o Modelo
@st.cache_resource
def carregar_modelo_seguro():
    try:
        return load_model(caminho_completo)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo no caminho: {caminho_completo}.pkl")
        st.error(f"Detalhe do erro: {e}")
        return None

model = carregar_modelo_seguro()

# 3. Interface
st.title("📊 Predição de Credit Scoring")
st.markdown(f"Modelo carregado de: `{CAMINHO_PASTA}`")

if model is None:
    st.warning("⚠️ O arquivo do modelo não foi encontrado. Verifique se o arquivo .pkl está na pasta correta.")
    st.stop() # Para a execução se não tiver modelo

# 4. Formulário de Entrada
st.sidebar.header("Dados do Cliente")

def get_user_input():
    # Variáveis comuns de crédito
    sexo = st.sidebar.selectbox("Sexo", ['M', 'F'])
    posse_de_veiculo = st.sidebar.selectbox("Possui Veículo?", ['S', 'N'])
    posse_de_imovel = st.sidebar.selectbox("Possui Imóvel?", ['S', 'N'])
    
    # MUDANÇA AQUI: step=1 e valores inteiros garantem que não haja decimais
    qtd_filhos = st.sidebar.number_input("Qtd Filhos", min_value=0, max_value=20, value=0, step=1)
    
    tipo_renda = st.sidebar.selectbox("Tipo de Renda", ['Assalariado', 'Empresário', 'Pensionista', 'Servidor público', 'Bolsista'])
    educacao = st.sidebar.selectbox("Educação", ['Fundamental', 'Médio', 'Superior incompleto', 'Superior completo', 'Pós graduação'])
    estado_civil = st.sidebar.selectbox("Estado Civil", ['Solteiro', 'Casado', 'Viúvo', 'Separado', 'União'])
    tipo_residencia = st.sidebar.selectbox("Tipo de Residência", ['Casa', 'Com os pais', 'Governamental', 'Aluguel', 'Estúdio', 'Comunitário'])
    
    idade = st.sidebar.slider("Idade", 18, 100, 30)
    tempo_emprego = st.sidebar.slider("Tempo de Emprego (anos)", 0.0, 50.0, 5.0)
    renda = st.sidebar.number_input("Renda Mensal", min_value=0.0, value=5000.0)
    
    # MUDANÇA AQUI TAMBÉM: Removi o .0 e adicionei step=1
    qt_pessoas_residencia = st.sidebar.number_input("Pessoas na Residência", min_value=1, max_value=15, value=2, step=1)
    
    # Dicionário de dados
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

# Mostrar dados
st.subheader("Dados Selecionados:")
st.dataframe(input_df)

# 5. Botão de Predição
if st.button('Calcular Risco'):
    # O PyCaret faz o pré-processamento automático
    prediction = predict_model(model, data=input_df)
    
    # Recupera resultados
    try:
        # Tenta pegar pelas colunas padrão do PyCaret
        resultado = prediction['prediction_label'].iloc[0]
        probabilidade = prediction['prediction_score'].iloc[0]
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Decisão do Modelo", f"{resultado}")
            
        with col2:
            st.metric("Probabilidade", f"{probabilidade:.2%}")

        if resultado == 1: # Assumindo 1 como Mau Pagador
            st.error("🚨 Classificação: **Mau Pagador**")
        else:
            st.success("✅ Classificação: **Bom Pagador**")
            
    except Exception as e:
        st.error("Erro ao ler a predição. Verifique as colunas de saída do modelo.")
        st.write(prediction) # Mostra o dataframe bruto para debug