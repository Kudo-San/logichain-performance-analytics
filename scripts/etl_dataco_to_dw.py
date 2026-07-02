# ==============================================================================
# SCRIPT: etl_dataco_to_dw.py
# PROJETO: LogiChain-Intelligence
# PROPÓSITO: Pipeline de extração, tratamento, modelagem dimensional e carga (ETL).
#            Lê o CSV bruto, limpa inconsistências e popula o Star Schema no Postgres.
# ==============================================================================

import os                             # Biblioteca nativa para manipulação de caminhos e arquivos do sistema
import sys                            # Controle do sistema para chamadas de interrupção (Fail-Fast)
import pandas as pd                   # Manipulação eficiente de estruturas tabulares em memória
from sqlalchemy import create_engine  # Motor de conexão unificado para SQL tradicional

# Captura o diretório atual do script (/scripts) de forma dinâmica e absoluta
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define o caminho para buscar o arquivo bruto subindo um nível para a raiz do projeto
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "dataset_CoSupplyChain", "DataCoSupplyChainDataset.csv")

print("⚡ Iniciando o Processamento ETL para o LogiChain-Intelligence...\n")

# ------------------------------------------------------------------------------
# STEP 1: EXTRAÇÃO E TRATAMENTO PRELIMINAR (FAIL-FAST)
# ------------------------------------------------------------------------------
print("[1/4] Lendo arquivo bruto do Kaggle e tratando strings...")

if not os.path.exists(CSV_PATH):
    print(f"❌ [ERRO] Arquivo não localizado no caminho esperado: {CSV_PATH}")
    sys.exit(1)

try:
    # Carrega os dados aplicando o encoding correto para caracteres internacionais
    df_raw = pd.read_csv(CSV_PATH, encoding='ISO-8859-1')
    print(f"✅ Sucesso! {len(df_raw)} linhas carregadas para processamento.")
except Exception as e:
    print(f"❌ Erro crítico ao abrir o arquivo CSV: {e}")
    sys.exit(1)

# Normalização dos nomes das colunas: remove espaços, caracteres especiais e padroniza em snake_case
# Isso evita problemas de sintaxe com aspas duplas no PostgreSQL tradicional
df_raw.columns = (
    df_raw.columns.str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('(', '')
    .str.replace(')', '')
)

# ------------------------------------------------------------------------------
# STEP 2: CONSTRUÇÃO DAS DIMENSÕES (MODELAGEM DIMENSIONAL IN-MEMORY)
# ------------------------------------------------------------------------------
print("\n[2/4] Modelando as tabelas de dimensões (Star Schema)...")

# --- DIMENSÃO CLIENTES ---
# Agrupa os atributos exclusivos do cliente e remove duplicatas de ID
dim_clientes = df_raw[[
    'customer_id', 'customer_fname', 'customer_lname', 
    'customer_segment', 'customer_city', 'customer_state', 'customer_country'
]].drop_duplicates(subset=['customer_id']).copy()

# Renomeia campos para manter o padrão sem ambiguidade
dim_clientes.rename(columns={'customer_id': 'id_cliente'}, inplace=True)

# --- DIMENSÃO PRODUTOS ---
# Isola as características dos produtos comercializados na cadeia
dim_produtos = df_raw[[
    'product_card_id', 'product_name', 'product_price', 
    'category_name', 'department_name'
]].drop_duplicates(subset=['product_card_id']).copy()

dim_produtos.rename(columns={'product_card_id': 'id_produto'}, inplace=True)

# --- DIMENSÃO LOCALIZAÇÃO DE ENTREGA ---
# Cria uma dimensão geográfica limpa com base nos destinos dos pedidos
dim_localizacao = df_raw[[
    'order_city', 'order_state', 'order_country', 'order_region', 'market'
]].drop_duplicates().copy()

# Como essa dimensão não vem com um ID de fábrica, cria uma Chave Substituta (Surrogate Key) sequencial
dim_localizacao['id_localizacao'] = range(1, len(dim_localizacao) + 1)

# Mapeia a tabela original para incluir o novo ID de localização antes de montar a tabela Fato
# Isso funciona como um PROCV (Merge) baseado nos eixos geográficos
df_raw = df_raw.merge(dim_localizacao, on=['order_city', 'order_state', 'order_country', 'order_region', 'market'], how='left')


# ------------------------------------------------------------------------------
# STEP 3: CONSTRUÇÃO DA TABELA FATO (MÉTRICAS E CHAVES)
# ------------------------------------------------------------------------------
print("\n[3/4] Estruturando a tabela Fato de pedidos e logística...")

# Seleciona as chaves de ligação e as métricas de performance operacionais/financeiras
fato_pedidos = df_raw[[
    'order_id', 'order_item_id', 'customer_id', 'product_card_id', 'id_localizacao',
    'order_date_dateorders', 'shipping_date_dateorders', 'days_for_shipping_real', 
    'days_for_shipment_scheduled', 'shipping_mode', 'order_status', 'delivery_status',
    'late_delivery_risk', 'sales', 'order_item_quantity', 'order_item_total', 'order_profit_per_order'
]].copy()

# Renomeia chaves para manter a integridade referencial clara
fato_pedidos.rename(columns={
    'customer_id': 'id_cliente',
    'product_card_id': 'id_produto',
    'order_date_dateorders': 'data_pedido',
    'shipping_date_dateorders': 'data_envio',
    'days_for_shipping_real': 'dias_envio_real',
    'days_for_shipment_scheduled': 'dias_envio_planejado',
    'order_item_total': 'receita_liquida',
    'order_profit_per_order': 'lucro_real'
}, inplace=True)

# Converte os campos de data que vieram como texto puro para o formato Datetime nativo do banco
fato_pedidos['data_pedido'] = pd.to_datetime(fato_pedidos['data_pedido'])
fato_pedidos['data_envio'] = pd.to_datetime(fato_pedidos['data_envio'])

print(f"📊 Modelagem dimensional finalizada:")
print(f"   -> dim_clientes: {len(dim_clientes)} registros únicos.")
print(f"   -> dim_produtos: {len(dim_produtos)} registros únicos.")
print(f"   -> dim_localizacao: {len(dim_localizacao)} rotas exclusivas.")
print(f"   -> fato_pedidos: {len(fato_pedidos)} itens de movimentação de carga.")


# ------------------------------------------------------------------------------
# STEP 4: CARGA NO POSTGRESQL (DOCKER / WSL2)
# ------------------------------------------------------------------------------
print("\n[4/4] Estabelecendo conexão e injetando tabelas no PostgreSQL Docker...")

try:
    # Abre o pool de conexões com o banco de dados target
    engine = create_engine('postgresql://user_ecommerce:senhaforte!@localhost:5432/ecomm_datawarehouse')
    
    # Injeta a dimensão de clientes substituindo tabelas antigas caso existam (if_exists='replace')
    print("   -> Gravando dim_clientes...")
    dim_clientes.to_sql('dim_clientes', engine, if_exists='replace', index=False)
    
    # Injeta a dimensão de produtos configurando os tipos relacionais automáticos
    print("   -> Gravando dim_produtos...")
    dim_produtos.to_sql('dim_produtos', engine, if_exists='replace', index=False)
    
    # Injeta a dimensão de localizações geográficas globais
    print("   -> Gravando dim_localizacao...")
    dim_localizacao.to_sql('dim_localizacao', engine, if_exists='replace', index=False)
    
    # Injeta a tabela Fato. Por ter quase 200k linhas, usamos o chunksize para não estourar a RAM do Docker
    print("   -> Gravando fato_pedidos (Operação em massa, aguarde)...")
    fato_pedidos.to_sql('fato_pedidos', engine, if_exists='replace', index=False, chunksize=50000)
    
    print("\n🏁 Processo ETL Concluído com Sucesso absoluto!")
    print("🚀 O Star Schema está de pé e populado dentro do PostgreSQL no Docker.")

except Exception as e:
    print(f"\n❌ [ERRO CRÍTICO NO PASSO 4] Falha ao tentar injetar dados no PostgreSQL.")
    print(f"⚠️ Detalhes: {e}")
    sys.exit(1)
