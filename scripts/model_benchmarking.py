# ==============================================================================
# SCRIPT: model_benchmarking.py
# NÍVEL: Pleno / Senior (Maturidade de Engenharia de ML)
# PROPÓSITO: Pipeline robusto de Benchmarking com Validação Cruzada, Pipelines do
#            Scikit-Learn, Tratamento de Desbalanceamento e Métricas Avançadas.
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

# Componentes do Scikit-Learn e XGBoost
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Modelos para Benchmarking
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Métricas de Avaliação
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, # PR-AUC
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)

print("🚀 A iniciar o Pipeline Avançado de ML Antifraude (v2) ...\n")

# ------------------------------------------------------------------------------
# 1. CONEXÃO E EXTRAÇÃO DE DADOS
# ------------------------------------------------------------------------------
print("[1/6] A conectar ao Data Warehouse e extrair a v_analise_vendas...")
try:
    engine = create_engine('postgresql://user_ecommerce:senhaforte!@localhost:5432/ecomm_datawarehouse')
    query = "SELECT * FROM v_analise_vendas"
    df = pd.read_sql(query, engine)
    print(f"✅ Dados carregados! Total de registos: {len(df)}")
except Exception as e:
    print(f"⚠️ Erro ao conectar ao banco: {e}")
    print("👉 A simular execução com dados mockados para validação do script...")
    # Fallback apenas para garantir que o script não quebra se testado fora do ambiente Docker
    np.random.seed(42)
    n_rows = 1096
    df = pd.DataFrame({
        'receita_liquida_pedido': np.random.exponential(scale=200, size=n_rows),
        'custo_frete': np.random.uniform(15, 80, size=n_rows),
        'dias_para_despachar': np.random.randint(1, 5, size=n_rows),
        'cliente_estado': np.random.choice(['SP', 'RJ', 'MG', 'AM', 'BA'], size=n_rows),
        'produto_categoria': np.random.choice(['Eletrônicos', 'Vestuário', 'Acessórios', 'Casa'], size=n_rows),
        'canal_venda': np.random.choice(['Website', 'App', 'RedSocial'], size=n_rows),
        'foi_chargeback': np.random.choice([0, 1], size=n_rows, p=[0.94, 0.06])
    })

# Converter target para inteiro por segurança
df['foi_chargeback'] = df['foi_chargeback'].astype(int)

# Separar X (Features) e y (Target)
X = df.drop(columns=['foi_chargeback'])
y = df['foi_chargeback']

# Calcular a razão de desbalanceamento para o XGBoost (Negativos / Positivos)
classes_count = np.bincount(y)
ratio_desbalanceamento = classes_count[0] / classes_count[1]
print(f"📊 Proporção de Classes: Saudáveis={classes_count[0]} | Chargebacks={classes_count[1]} (Ratio: {ratio_desbalanceamento:.2f})")

# ------------------------------------------------------------------------------
# 2. ARQUITETURA DE PRÉ-PROCESSAMENTO (ColumnTransformer + Pipelines)
# ------------------------------------------------------------------------------
print("\n[2/6] A configurar ColumnTransformer (Evita Data Leakage e Inconsistências)...")

features_numericas = ['receita_liquida_pedido', 'custo_frete', 'dias_para_despachar']
features_categoricas = ['cliente_estado', 'produto_categoria', 'canal_venda']

# Pipeline Numérico: Imputação + Padronização (Essencial para Regressão Logística)
pipeline_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline Categórico: Imputação + One-Hot Encoding robusto
pipeline_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', pipeline_numerico, features_numericas),
    ('cat', pipeline_categorico, features_categoricas)
])

# ------------------------------------------------------------------------------
# 3. VALIDAÇÃO CRUZADA E BENCHMARKING DE MODELOS
# ------------------------------------------------------------------------------
print("\n[3/6] A iniciar Validação Cruzada Estratificada (5-Folds)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modelos com estratégias nativas de balanceamento de classe (Resolver o GAP 2 e 5)
dict_modelos = {
    "Regressão Logística (Baseline)": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(scale_pos_weight=ratio_desbalanceamento, eval_metric='logloss', random_state=42, n_jobs=-1)
}

resultados_consolidados = []

for nome_modelo, algoritmo in dict_modelos.items():
    print(f"🤖 A avaliar: {nome_modelo}...")
    
    fold_roc_aucs, fold_pr_aucs = [], []
    fold_f1s, fold_precisions, fold_recalls = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # O fit do scaler/encoder ocorre APENAS nos dados de treino de cada fold!
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', algoritmo)
        ])
        
        model_pipeline.fit(X_train_fold, y_train_fold)
        
        preds_classes = model_pipeline.predict(X_val_fold)
        preds_probs = model_pipeline.predict_proba(X_val_fold)[:, 1]
        
        fold_roc_aucs.append(roc_auc_score(y_val_fold, preds_probs))
        fold_pr_aucs.append(average_precision_score(y_val_fold, preds_probs))
        fold_f1s.append(f1_score(y_val_fold, preds_classes, zero_division=0))
        fold_precisions.append(precision_score(y_val_fold, preds_classes, zero_division=0))
        fold_recalls.append(recall_score(y_val_fold, preds_classes, zero_division=0))
        
    resultados_consolidados.append({
        "Modelo": nome_modelo,
        "ROC-AUC Média": np.mean(fold_roc_aucs),
        "PR-AUC Média": np.mean(fold_pr_aucs),
        "F1-Score Médio": np.mean(fold_f1s),
        "Precision Média": np.mean(fold_precisions),
        "Recall Médio": np.mean(fold_recalls),
    })

# ------------------------------------------------------------------------------
# 4. RANKING CONSOLIDADO DOS MODELOS
# ------------------------------------------------------------------------------
print("\n[4/6] 🏆 Ranking Consolidado (Ordenado por PR-AUC):")
df_ranking = pd.DataFrame(resultados_consolidados).sort_values(by="PR-AUC Média", ascending=False)
print(df_ranking.to_string(index=False, formatters={
    "ROC-AUC Média": "{:.4f}".format,
    "PR-AUC Média": "{:.4f}".format,
    "F1-Score Médio": "{:.4f}".format,
    "Precision Média": "{:.4f}".format,
    "Recall Médio": "{:.4f}".format
}))

melhor_nome_modelo = df_ranking.iloc[0]['Modelo']
print(f"\n🎯 O modelo escolhido para produção foi: {melhor_nome_modelo}")

# ------------------------------------------------------------------------------
# 5. MATRIZ DE CONFUSÃO E TREINO FINAL
# ------------------------------------------------------------------------------
print(f"\n[5/6] A treinar o Pipeline Final do {melhor_nome_modelo} com todos os dados...")

melhor_algoritmo = dict_modelos[melhor_nome_modelo]
pipeline_final = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', melhor_algoritmo)
])

pipeline_final.fit(X, y)
final_preds = pipeline_final.predict(X)
cm = confusion_matrix(y, final_preds)

print("\n📊 Matriz de Confusão Final (Dados Totais):")
print(f"   [Verdadeiros Negativos (Clientes Bons): {cm[0][0]}]   [Falsos Positivos (Bloqueios Injustos): {cm[0][1]}]")
print(f"   [Falsos Negativos (Fraudes Passaram):  {cm[1][0]}]   [Verdadeiros Positivos (Fraudes Pegas):  {cm[1][1]}]")

# ------------------------------------------------------------------------------
# 6. EXPORTAÇÃO (DEPLOY READY)
# ------------------------------------------------------------------------------
print("\n[6/6] A exportar o Pipeline unificado para ambiente de produção...")
nome_arquivo_pipeline = "pipeline_antifraude_final.pkl"
joblib.dump(pipeline_final, nome_arquivo_pipeline)
print(f"✅ Sucesso absoluto! O ficheiro '{nome_arquivo_pipeline}' contém o pré-processamento e o modelo integrados.")