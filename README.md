![LogiChain Banner](images/logichain_banner.png)

<div align="center">
  <br />
  <p align="center">
    <img src="https://img.shields.io/badge/LOGICHAIN-Supply%20Chain%20Risk%20Intelligence-112E47?style=for-the-badge&logo=databricks&logoColor=white" height="40" />
  </p>
  <p align="center">
    <b>Transformando operações logísticas globais em inteligência de risco financeiro</b>
  </p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
  [![Docker](https://img.shields.io/badge/Docker-Latest-blue.svg?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
  [![Power BI](https://img.shields.io/badge/Power_BI-Analytics-yellow.svg?style=flat-square&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
  <br />
  <hr />
</div>

### ⚡ TL;DR

Plataforma end-to-end de dados que transforma operações logísticas em inteligência de risco financeiro usando Data Warehouse, Machine Learning e BI.

* 📉 OTIF global crítico: 6.18%

* 💸 Exposição a risco: $3.24M

* 🤖 Economia estimada com IA: $2.75M

* 🧱 Stack: Docker • Postgres • Python • SQL • Power BI • ML (Random Forest)

---

### Problema de Negócio

A DataCo opera com uma cadeia global de e-commerce e logística enfrentando um colapso operacional crítico:

* Baixa taxa de entregas no prazo e completas (OTIF = 6.18%)
* Falta de visibilidade sobre gargalos reais da operação
* Perdas financeiras ocultas por falhas logísticas recorrentes
* Decisões baseadas em média agregada distorcida

👉 O problema não era transporte. Era inteligência operacional inexistente.

---

### Solução

O LogiChain foi projetado como uma plataforma analítica completa com 4 camadas:

1. Infraestrutura de Dados (Docker + PostgreSQL)
2. Data Warehouse em Star Schema
3. Camada Analítica com SQL Views
4. Machine Learning para previsão de risco
5. Dashboard executivo em Power BI

---

### Arquitetura de Dados & Data Pipeline
Arquitetura foca na descentralização e performance, utilizando o banco de dados relacional para processar regras de negócio pesadas via SQL Views, aliviando a carga cognitiva e de memória do Power BI.

```mermaid
graph TD
    A[1. Infraestrutura Docker<br/><i>docker-compose.yml</i>] -->|Subida do Container| B[(PostgreSQL Database<br/><i>LogiChain_DB</i>)]
    
    subgraph "Camada de Dados & DW (Postgres)"
        B -->|Ingestão via Python| C{2. Star Schema / Modelo Estrela}
        
        subgraph "Dimensões & Fatos Relacionais"
            C --> D1(public.dim_produtos)
            C --> D2(public.dim_clientes)
            C --> D3(public.dim_localizacao)
            C --> D4(public.fct_pedidos_logistica)
        end
        
        D1 & D2 & D3 & D4 -->|Agregação SQL| E[3. Core Analítico<br/><i>SQL View: v_delivery_performance</i>]
    end

    subgraph "Camada de Inteligência Artificial"
        E -->|Leitura Histórica| F[4. Machine Learning Pipeline<br/><i>Random Forest .pkl</i>]
        F -->|Prevenção de Riscos| H[5. Métrica Predict<br/><i>Perda Evitável com IA</i>]
    end
    
    E & H -->|Conexão Localhost| G["✨ 6. Consumo Analítico<br/><i>PowerBI Performance Dashboard</i>"]

    subgraph "Arquitetura do Dashboard (3 Abas)"
        G --> G1(Aba 1: Executive Summary)
        G --> G2(Aba 2: Logistics Performance)
        G --> G3(Aba 3: Risk Analytics)
    end

    classDef yellowBox fill:#E6B800,stroke:#B38F00,stroke-width:2px,color:#000000;
    classDef blueBox fill:#2B85FF,stroke:#1A52A3,stroke-width:2px,color:#FFFFFF;
    classDef dbBox fill:#1F4E79,stroke:#112E47,stroke-width:2px,color:#FFFFFF;
    classDef greenBox fill:#54B435,stroke:#377D22,stroke-width:2px,color:#FFFFFF;
    classDef purpleBox fill:#6F38C5,stroke:#4A2485,stroke-width:2px,color:#FFFFFF;
    classDef dimBox fill:#F2F4F7,stroke:#D0D5DD,stroke-width:1px,color:#101828;

    class A yellowBox;
    class B,C dbBox;
    class E greenBox;
    class F,H blueBox;
    class G purpleBox;
    class D1,D2,D3,D4,G1,G2,G3 dimBox;
```

📊 Data Warehouse Model

Modelo estrela otimizado para performance analítica:

* Dimensões:
    * public.dim_produtos
    * public.dim_clientes
    * public.dim_localização
* Fato:
    * public.dim_pedidos_logistica
* Camada semântica:
    * SQL Views para performance e abstração de regras de negócio

---

### Inteligência de Negócio: Wireframe do Dashboard (3 Abas)
O Dashboard foi construído seguindo padrões de UX/UI, estruturado em uma jornada analítica que vai do sumário executivo até a causa raiz de perdas financeiras.

```mermaid
graph TD
    subgraph Dashboard ["📊 WIREFRAME: LOGICHAIN PERFORMANCE & RISK DASHBOARD (16:9)"]
        direction TB

        subgraph Header ["Filtros Globais Interativos (Top Control Layer)"]
            Title["ℹ️ LogiChain Control Center: Gestão de Prazos, Eficiência Logística e ROI de IA"]
            F1["📅 Year of Request"] --- F2["👤 Customer Segment"] --- F3["🌍 Regional Market (Cross-Filtering via Visuals)"]
        end

        subgraph KPI_Layer ["Aba 1: Executive Summary - KPIs de Alto Impacto (High-Level Cards)"]
            direction LR
            K1["💵 Faturamento Bruto<br/><b>SUM(faturamento_bruto)</b><br/><i>$36,78 Mi</i>"]
            K2["📦 Itens Despachados<br/><b>COUNT(id_item_pedido)</b><br/><i>181 Mil</i>"]
            K3["🔴 Lucro em Risco<br/><b>CALCULATE(Lucro, OTIF=0)</b><br/><i>$3,24 Mi (A Dor)</i>"]
            K4["🟢 Economia com IA<br/><b>[Lucro em Risco] * 84.88%</b><br/><i>$2,75 Mi (O ROI)</i>"]
            K5["🎯 Taxa OTIF Global<br/><b>AVERAGE(kpi_otif)</b><br/><i>6,18% (Eficiência)</i>"]
        end

        subgraph Ops_Layer ["Aba 2: Logistics Performance - Análise de Processos & Tempo"]
            direction LR
            G1["📈 Linhas: Sazonalidade Temporal<br/><b>X: Mês do Pedido | Y: Taxa OTIF</b><br/><i>(Mapeia gargalos de Agosto e Dezembro)</i>"]
            G2["📊 Linhas/Colunas: Promessa de Prazo<br/><b>X: dias_envio_planejado | Y: Gap Médio e OTIF</b><br/><i>(Prova o colapso de 24h e o efeito do gap 0,6)</i>"]
            G3["🗂️ Matriz: Drill-down de Rotas<br/><b>Linhas: market > order_country</b><br/><i>(Isola a Taxa On-Time vs Taxa In-Full)</i>"]
        end

        subgraph Risk_Layer ["Aba 3: Risk Analytics - Tomada de Decisão Estratégica"]
            direction LR
            G4["🎯 Pareto: Concentração de Atrasos<br/><b>X: order_country | Y: Itens Atrasados & % Acumulado</b><br/><i>(Regra 80/20: Isola as 7 nações críticas)</i>"]
            G5["📈 Dispersão: Correlação Desvio x Perda<br/><b>X: Gap Médio (Dias) | Y: Lucro em Risco</b><br/><i>(Identifica o outlier do Fan Shop com $1,5 Mi)</i>"]
        end

    end

    classDef headerStyle fill:#112E47,stroke:#1F4E79,stroke-width:2px,color:#FFFFFF;
    classDef filterStyle fill:#F2F4F7,stroke:#D0D5DD,stroke-width:1px,color:#101828;
    classDef cardStyle fill:#1F4E79,stroke:#112E47,stroke-width:2px,color:#FFFFFF;
    classDef alertCard fill:#990000,stroke:#660000,stroke-width:2px,color:#FFFFFF;
    classDef successCard fill:#377D22,stroke:#2A6319,stroke-width:2px,color:#FFFFFF;
    classDef chartStyle fill:#FFFFFF,stroke:#E4E7EC,stroke-width:1px,color:#344054;
    classDef containerStyle fill:#F8F9FA,stroke:#D0D5DD,stroke-width:2px,color:#101828;

    class Title headerStyle;
    class F1,F2,F3 filterStyle;
    class K1,K2,K5 cardStyle;
    class K3 alertCard;
    class K4 successCard;
    class G1,G2,G3,G4,G5 chartStyle;
    class Dashboard,Header,KPI_Layer,Risk_Layer,Ops_Layer containerStyle;
```
📈 Inteligência de Negócio (KPIs)

| Métrica             | Valor     | Interpretação        |
| ------------------- | --------- | -------------------- |
| OTIF Global         | 6.18% 🔴  | Eficiência crítica   |
| Pedidos processados | 181K 📦   | Volume operacional   |
| Lucro em risco      | $3.24M 💸 | Exposição financeira |
| Economia com IA     | $2.75M 🟢 | ROI potencial        |

---

### Principais Insights
1. Problema estrutural mascarado por média

    O gap médio de entrega (~0.6 dias) escondia atrasos críticos devido à compensação estatística entre adiantamentos e atrasos.

2. Falha total em entregas expressas

    Pedidos com promessa de 1 dia apresentam 0% OTIF, revelando colapso em SLAs agressivos.

3. Gargalo interno, não logístico

    Mesmo com transporte eficiente, OTIF permanece baixo (~6%), indicando falhas em:

    * separação
    * faturamento
    * processos internos

4. Concentração de risco (Pareto)

    7 países concentram ~80% dos atrasos globais, permitindo foco operacional direcionado.

5. Categoria crítica: Fan Shop

    Responsável por $1.5M em risco financeiro, sendo o maior hotspot da operação.

---

### Machine Learning (Risk Engine)

Foi desenvolvido um modelo de classificação para previsão de atrasos logísticos:

* Algoritmo: Random Forest
* Métrica principal: PR-AUC = 0.7978
* Abordagem: dados históricos + features de rota e tempo de processamento
* Objetivo: antecipar falhas antes da execução logística

💰 Impacto estimado:
Redução potencial de perdas financeiras em $2.75M

--- 

### Dashboard (Power BI)

O painel foi estruturado em 3 níveis analíticos:

#### 1. Executive Summary
![Executive Summary](images/aba1_executive_summary.png)
* KPIs financeiros e operacionais
* Visão de saúde global da operação
#### 2. Logistics Performance
![Logistics Performance](images/aba2_logistics_performance.png)
* Sazonalidade de atrasos
* Gap entre promessa e entrega
* Drill-down por região
#### 3. Risk Analytics
![Risk Analytics](images/aba3_risk_analytics.png)
* Pareto de países críticos
* Correlação entre atraso e perda financeira
* Identificação de outliers operacionais

---

### Tecnologias Utilizadas
Data Engineering
* Docker
* PostgreSQL
* SQL (views e modelagem dimensional)

Data Science
* Python 3.10
* Pandas
* Scikit-learn
* Random Forest

Business Intelligence
* Power BI
* DAX
* Modelagem analítica e UX de dashboards

--- 

### Como Executar

1. Clone do repositório
```bash
git clone https://github.com/seu-usuario/logichain-performance-analytics.git
```
2. Inicialize o banco de dados no container Docker:
``` bash
docker-compose up -d

Passos seguintes:

a. O banco de dados PostgreSQL subirá automaticamente em localhost:5432.

b. O container etl_service iniciará em sequência para popular o Data Warehouse.

c. Abra o arquivo .pbix localizado na pasta / powerbi para explorar os dados.
```

3. Baixe os dados do Kaggle:
``` bash
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
```

---

### Resultado Final

O LogiChain transforma dados logísticos brutos em inteligência acionável de risco financeiro, permitindo:

Identificação de gargalos invisíveis
Priorização de regiões críticas
Previsão de falhas operacionais
Quantificação do impacto financeiro da ineficiência

---

### Visão

Este projeto simula uma arquitetura real de uma plataforma de dados corporativa moderna:

Data Warehouse + Analytics + Machine Learning + BI = decisão orientada a valor

---

### 👨‍💻 Autor

Desenvolvido por Marcelo Kudo