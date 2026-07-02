# ==============================================================================
# SCRIPT: inspect_dataset.py
# PROPÓSITO: Realizar a análise exploratória estrutural básica do arquivo bruto.
# OBJETIVO:  Mapear nomes de colunas, tipos de dados e volumetria de nulos
#            para desenhar o modelo de dados (DDL) no PostgreSQL.
# ==============================================================================

import os            # Biblioteca nativa para manipulação de caminhos de arquivos e diretórios
import pandas as pd  # Biblioteca padrão para análise e manipulação de tabelas de dados

# Captura o caminho absoluto de onde este script atual está localizado (pasta /scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sobe um nível para a raiz do projeto (portfolio_ecommerce) e encontra a pasta do dataset
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "dataset_CoSupplyChain", "DataCoSupplyChainDataset.csv")

print("🔍 Iniciando a inspeção estrutural do DataCo Supply Chain Dataset...\n")

try:
    # Carrega o dataset especificando a codificação 'ISO-8859-1' para evitar erros de caracteres globais
    df = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1')
    
    print("✅ Arquivo carregado com sucesso!")
    print(f"📊 Volumetria Total: {df.shape[0]} linhas e {df.shape[1]} colunas.\n")
    
    print("📋 Resumo das Colunas Detectadas e Tipos de Dados:")
    print("-" * 80)
    
    # Cria o DataFrame de diagnóstico contendo o Tipo de Dado e a contagem de valores nulos
    info_colunas = pd.DataFrame({
        'Tipo_Dado': df.dtypes,                                    # Captura se a coluna é int64, float64 ou object (texto)
        'Valores_Nulos': df.isnull().sum(),                        # Conta quantas linhas vazias existem na coluna
        'Percentual_Nulos %': (df.isnull().sum() / len(df)) * 100  # Calcula a proporção de perda de dados
    })
    
    # Exibimos a lista completa de colunas na ordem original do arquivo CSV.
    print(info_colunas.to_string())
    print("-" * 80)
    
    print("\n💡 Principais colunas logísticas de interesse detectadas:")
    # Lista algumas colunas estratégicas conhecidas desse dataset para validação visual rápida
    colunas_chave = ['Order Status', 'Delivery Status', 'Late_delivery_risked', 'Days for shipping (real)', 'Days for shipment (scheduled)']
    for col in colunas_chave:
        if col in df.columns:
            print(f"  -> {col}: Encontrada com sucesso!")

except Exception as e:
    # Tratamento de erro Fail-Fast
    print(f"❌ Erro crítico ao tentar ler o arquivo CSV: {e}")
