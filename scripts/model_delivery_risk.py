# =======================================================================================================================================
# SCRIPT: model_delivery_risk.py
# PROJETO: LogiChain-Intelligence
# PROPÓSITO: Pipeline automatizado de Machine Learning para prever riscos de atraso na cadeia de suprimentos global (Late Delivery Risk).
# ARQUITETURA: Scikit-Learn Pipelines + ColumnTransformer + Stratified K-Fold.
# POLÍTICA DE OPERAÇÃO: Fail-Fast (Interrupção imediata em caso de erro crítico).
# =======================================================================================================================================

import os                             # Biblioteca nativa para manipulação de caminhos e diretórios do sistema
import sys                            # Biblioteca nativa para encerramento forçado do script em caso de falha (sys.exit)
import joblib                         # Biblioteca padrão para serializar e salvar modelos treinados em disco (.pkl)
import numpy as np                    # Biblioteca para operações matemáticas e manipulação de vetores/matrizes
import pandas as pd                   # Biblioteca padrão para análise, filtros e manipulação de tabelas (DataFrames)
from sqlalchemy import create_engine  # Motor de conexão para bancos de dados relacionais via Python

# Componentes de Validação e Divisão Estatística de Dados
from sklearn.model_selection import StratifiedKFold  # Fatiador de dados que preserva a proporção exata da classe alvo

# Componentes de Transformação de Dados e Engenharia de Features
from sklearn.compose import ColumnTransformer        # Aplica transformações isoladas dependendo do tipo da coluna
from sklearn.pipeline import Pipeline                # Encapsula etapas de pré-processamento e modelagem num fluxo protegido
from sklearn.preprocessing import StandardScaler     # Normaliza dados numéricos (Média = 0, Desvio Padrão = 1)
from sklearn.preprocessing import OneHotEncoder      # Converte textos categóricos em colunas de chaves binárias (0 e 1)
from sklearn.impute import SimpleImputer             # Trata e substitui valores ausentes ou nulos (NaN) da base

# Modelos Avançados de Classificação para Competição e Benchmarking
from sklearn.linear_model import LogisticRegression  # Modelo linear clássico para servir de base comparativa (Baseline)
from sklearn.ensemble import RandomForestClassifier  # Algoritmo de árvores ensacadas (Bagging) de alta robustez
from xgboost import XGBClassifier                    # Algoritmo de Boosting de Gradiente Extremo (Estado da arte para tabelas)

# Métricas de Avaliação Estratégicas para Classes com Desbalanceamento Logístico
from sklearn.metrics import (
    roc_auc_score,            # Área sob a curva ROC (Mede a capacidade global de separação do modelo)
    average_precision_score,  # Métrica PR-AUC (Foco total na precisão e sensibilidade da classe de risco)
    f1_score,                 # Média harmônica balanceada entre Precisão e Recall
    precision_score,          # Mede a taxa de acerto entre os alertas disparados (Evita falsos alarmes logísticos)
    recall_score,             # Mede a capacidade do modelo de capturar os atrasos reais (Evita quebras na cadeia)
    confusion_matrix          # Matriz cruzada de contagem exata de erros e acertos do negócio
)

print("🚀 Iniciando o Pipeline de Inteligência Preditiva de Supply Chain...\n")

# ------------------------------------------------------------------------------
# STEP 1: CONEXÃO E EXTRAÇÃO DA VIEW DE NEGÓCIO (FAIL-FAST)
# ------------------------------------------------------------------------------
print("[1/6] Conectando ao PostgreSQL e extraindo dados da v_delivery_performance...")

try:
    # Estabelece a conexão com o pool do PostgreSQL rodando dentro do container Docker
    engine = create_engine('postgresql://user_ecommerce:senhaforte!@localhost:5432/ecomm_datawarehouse')
    
    # Executa a leitura diretamente em cima da View mestre de performance operacional
    query = "SELECT * FROM v_delivery_performance"
    df = pd.read_sql(query, engine)
    print(f"✅ Dados reais carregados com sucesso! Total de movimentações de carga: {len(df)}")

except Exception as e:
    # Aplicação estrita da política Fail-Fast para blindar o fluxo contra contaminações
    print(f"\n❌ [ERRO CRÍTICO NO PASSO 1] Falha ao tentar ler a View analítica no banco.")
    print(f"⚠️ Detalhes Técnicos: {e}")
    print("🛑 Encerrando o pipeline de Machine Learning imediatamente.")
    sys.exit(1)

# Converte a coluna booleana 'risco_atraso' calculada pela View para formato numérico inteiro (1 ou 0)
df['risco_atraso'] = df['risco_atraso'].astype(int)

# Isola as variáveis preditivas (X) removendo o alvo e campos identificadores puros (IDs/Datas)
# Deixendo apenas o que o modelo consegue usar para aprender padrões operacionais
remover_colunas = ['id_pedido', 'id_item_pedido', 'id_cliente', 'id_produto', 'id_localizacao', 
                      'data_pedido', 'data_envio', 'dias_envio_real', 'dias_desvio_prazo',
                     'entrega_no_prazo', 'entrega_completa', 'kpi_otif', 'lucro_em_risco', 'risco_atraso',
                     'delivery_status', 'status_entrega', 'order_status', 'status_pedido']

colunas_para_dropar = [c for c in remover_colunas if c in df.columns]
X = df.drop(columns=colunas_para_dropar)
y = df['risco_atraso']  # Define o novo Target (1 = Pedido Atrasou / 0 = Chegou no Prazo)


# Avalia a proporção real de desbalanceamento logístico presente na base de dados
classes_count = np.bincount(y)
ratio_desbalanceamento = float(classes_count[0] / classes_count[1])
print(f"📊 Distribuição do Target: No Prazo (0) = {classes_count[0]} | Atrasados (1) = {classes_count[1]} (Razão: {ratio_desbalanceamento:.2f})")


# ------------------------------------------------------------------------------
# STEP 2: DESIGN DA ESTEIRA DE PRÉ-PROCESSAMENTO (ANTI-DATA LEAKAGE)
# ------------------------------------------------------------------------------
print("\n[2/6] Configurando ColumnTransformer corporativo para isolamento de características...")

# Mapeamento explícito das features remanescentes da View por tipo de dado
features_numericas = ['dias_envio_planejado', 'faturamento_bruto', 'quantidade_itens', 'receita_liquida', 'lucro_real']
features_categoricas = ['modo_envio']

# Configura o tratamento numérico isolado (Imputação da Mediana + Padronização de Escala Vetorial)
pipeline_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Configura o tratamento textual isolado (Imputação da Moda + One-Hot Encoding com tratamento de novas categorias)
pipeline_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Consolida os transformadores garantindo que cada coluna receba o tratamento matemático correto
preprocessor = ColumnTransformer(transformers=[
    ('num', pipeline_numerico, features_numericas),
    ('cat', pipeline_categorico, features_categoricas)
])


# ------------------------------------------------------------------------------
# STEP 3: VALIDAÇÃO CRUZADA E COMPETIÇÃO DE ALGORITMOS (BENCHMARKING)
# ------------------------------------------------------------------------------
print("\n[3/6] Iniciando loop de Validação Cruzada Estratificada (5-Folds)...")

# Configura a divisão em 5 blocos misturados mantendo a reprodutibilidade através do random_state
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instancia os competidores aplicando parâmetros rígidos de regularização (max_depth) e pesos de classe
dict_modelos = {
    "Regressão Logística (Baseline)": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Random Forest (Regularizada)": RandomForestClassifier(class_weight='balanced', max_depth=6, min_samples_leaf=15, random_state=42, n_jobs=-1),
    "XGBoost (Regularizado)": XGBClassifier(scale_pos_weight=ratio_desbalanceamento, max_depth=4, subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)
}

# A lista que guardará os sumários de performance média de cada algoritmo avaliado
resultados_consolidados = []

try:
    # Itera sobre os dicionários de modelos para iniciar a competição estatística
    for nome_modelo, algoritmo in dict_modelos.items():
        print(f"🤖 Avaliando comportamento de: {nome_modelo}...")
        
        # Listas de controle para armazenar as métricas calculadas em cada um dos folds isolados
        fold_roc_aucs, fold_pr_aucs = [], []
        fold_f1s, fold_precisions, fold_recalls = [], [], []
        
        # Divide a base real dinamicamente nos índices de treino e validação do fold atual
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Constrói o Pipeline encapsulado do fold corrente (Pré-processamento + Algoritmo)
            # Isso garante isolamento absoluto e impede vazamento de dados de validação para o fit do scaler
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', algoritmo)
            ])
            
            # Treina as transformações e o classificador exclusivamente com a fatia de treino do fold
            model_pipeline.fit(X_train_fold, y_train_fold)
            
            # Executa as predições de classe e extrai as probabilidades da classe positiva (risco)
            preds_classes = model_pipeline.predict(X_val_fold)
            preds_probs = model_pipeline.predict_proba(X_val_fold)[:, 1]
            
            # Computa as métricas reais do fold e anexa nas listas de rastreamento
            fold_roc_aucs.append(roc_auc_score(y_val_fold, preds_probs))
            fold_pr_aucs.append(average_precision_score(y_val_fold, preds_probs))
            fold_f1s.append(f1_score(y_val_fold, preds_classes, zero_division=0))
            fold_precisions.append(precision_score(y_val_fold, preds_classes, zero_division=0))
            fold_recalls.append(recall_score(y_val_fold, preds_classes, zero_division=0))
            
        # Consolida a média matemática dos 5 folds e injeta no repositório de resultados do benchmark
        resultados_consolidados.append({
            "Modelo": nome_modelo,
            "ROC-AUC Média": float(np.mean(fold_roc_aucs)),
            "PR-AUC Média (Crítica)": float(np.mean(fold_pr_aucs)),
            "F1-Score Médio": float(np.mean(fold_f1s)),
            "Precision Média": float(np.mean(fold_precisions)),
            "Recall Médio": float(np.mean(fold_recalls))
        })
except Exception as e:
    print(f"\n❌ [ERRO CRÍTICO NO PASSO 3] Falha inesperada durante a validação cruzada.")
    print(f"⚠️ Detalhes: {e}")
    sys.exit(1)


# ------------------------------------------------------------------------------
# STEP 4: EMISSÃO DO RANKING CONSOLIDADO DE COMPUTAÇÃO
# ------------------------------------------------------------------------------
print("\n[4/6] 🏆 Exibindo Ranking Consolidado de Supply Chain (Ordenado por PR-AUC):")

# Transforma a lista de resultados em DataFrame e ordena de forma decrescente pela PR-AUC Média
df_ranking = pd.DataFrame(resultados_consolidados).sort_values(by="PR-AUC Média (Crítica)", ascending=False)

# Imprime a tabela de performance formatando os decimais com 4 casas de precisão
print(df_ranking.to_string(index=False, formatters={
    "ROC-AUC Média": "{:.4f}".format,
    "PR-AUC Média (Crítica)": "{:.4f}".format,
    "F1-Score Médio": "{:.4f}".format,
    "Precision Média": "{:.4f}".format,
    "Recall Médio": "{:.4f}".format
}))

# Extrai o nome do modelo que liderou o ranking de performance preditiva
melhor_nome_modelo = df_ranking.iloc[0]['Modelo']
print(f"\n🎯 O algoritmo eleito para governar as predições foi: {melhor_nome_modelo}")


# ------------------------------------------------------------------------------
# STEP 5: TREINAMENTO DO PIPELINE FINAL E ANÁLISE DE IMPACTO OPERACIONAL
# ------------------------------------------------------------------------------
print(f"\n[5/6] Ajustando o Pipeline Definitivo do {melhor_nome_modelo} com a base cheia...")

try:
    # Resgata a instância limpa do algoritmo campeão para acoplamento final
    melhor_algoritmo = dict_modelos[melhor_nome_modelo]
    
    pipeline_final = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', melhor_algoritmo)
    ])

    # Vai treina o pipeline final utilizando 100% dos dados reais extraídos do DW
    pipeline_final.fit(X, y)
    
    # Executa a predição na base total para mapeamento da Matriz de Confusão corporativa
    final_preds = pipeline_final.predict(X)
    cm = confusion_matrix(y, final_preds)

    # Exibe os quadrantes traduzindo jargões estatísticos em dores de impacto financeiro do negócio
    print("\n📊 Matriz de Confusão Final (Volume Consolidado do DW):")
    print(f"   [Verdadeiros Negativos (Rotas Aprovadas no Prazo): {cm[0][0]}]   [Falsos Positivos (Custos de Alarme Falso):  {cm[0][1]}]")
    print(f"   [Falsos Negativos (Atrasos Não Detectados/Prejuízo): {cm[1][0]}]   [Verdadeiros Positivos (Gargalos Mitigados):   {cm[1][1]}]")

except Exception as e:
    print(f"\n❌ [ERRO CRÍTICO NO PASSO 5] Falha no ajuste ou avaliação da matriz definitiva.")
    print(f"⚠️ Detalhes: {e}")
    sys.exit(1)


# ------------------------------------------------------------------------------
# STEP 6: SERIALIZAÇÃO DO PIPELINE DE PRODUÇÃO (DEPLOY READY)
# ------------------------------------------------------------------------------
print("\n[6/6] Exportando o Pipeline unificado completo para o ambiente de produção...")

try:
    # Define o nome padrão do binário completo que será lido pelos servidores ou pelo Watchdog
    nome_arquivo_pipeline = "pipeline_logichain_final.pkl"
    
    # Salva o fluxo contendo o ColumnTransformer e o Modelo em uma única chamada de escrita estável
    joblib.dump(pipeline_final, nome_arquivo_pipeline)
    print(f"✅ Sucesso absoluto! O arquivo binário '{nome_arquivo_pipeline}' foi gerado.")
    print("🏁 Projeto LogiChain-Intelligence: Pipeline de Machine Learning Finalizado.")

except Exception as e:
    print(f"\n❌ [ERRO CRÍTICO NO PASSO 6] Falha ao gravar o arquivo binário do pipeline (.pkl).")
    print(f"⚠️ Detalhes: {e}")
    sys.exit(1)