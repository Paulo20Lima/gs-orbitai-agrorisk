# OrbitAI AgroRisk 🌍🌾
> **Global Solution — Machine Learning + Economia Espacial** > *Sistema Inteligente de Previsão de Risco Climático Agrícola com Dados Espaciais*

---

## 👥 Integrantes
* **Guilherme Silva Cavalcanti** — RM: 98928
* **Paulo Henrique da Silva Lima** — RM: 552444
* **Wesley Assis Oliveira** — RM: 552516

### 🎓 Turma
**4SIR**

---

## 1. Resumo Executivo
O crescimento da economia espacial tornou possível utilizar dados orbitais para resolver problemas reais na Terra. O agronegócio, um dos setores mais importantes da economia brasileira, sofre constantemente com eventos climáticos extremos como seca, excesso de chuva e mudanças bruscas de temperatura, causando perdas financeiras massivas e redução de produtividade.

Pensando nisso, o projeto **OrbitAI AgroRisk** propõe uma solução baseada em **Machine Learning (Inteligência Artificial)** capaz de analisar dados climáticos e informações derivadas de sensores de satélites para prever áreas agrícolas com risco iminente de perda produtiva.

A solução utiliza técnicas de Ciência de Dados para:
* Coletar e tratar dados ambientais;
* Identificar padrões climáticos e de estresse vegetativo;
* Prever riscos agrícolas por talhão;
* Auxiliar na tomada de decisão preditiva de produtores e gestores.

O projeto conecta diretamente tecnologia espacial, inteligência artificial e sustentabilidade, demonstrando como a exploração espacial gera impacto positivo direto nos setores produtivos da Terra.

---

## 2. O Problema
As mudanças climáticas têm aumentado significativamente a frequência de secas prolongadas, ondas de calor, chuvas extremas e a consequente degradação do solo.

Muitos produtores agrícolas ainda dependem de análises tardias e decisões reativas, o que aumenta:
* O desperdício de insumos e recursos hídricos;
* As perdas de safras completas;
* Os prejuízos financeiros em cascata na cadeia de suprimentos.

Apesar da existência de constelações de satélites capazes de monitorar o planeta em tempo real, grande parte desses dados brutos não é processada ou integrada de forma inteligente para a prevenção ativa de riscos no campo.

---

## 3. Objetivo
Desenvolver um modelo de Machine Learning robusto e reprodutível capaz de prever o risco climático em áreas agrícolas utilizando variáveis ambientais combinadas a indicadores obtidos por sensoriamento remoto orbital.

---

## 4. Hipótese
É possível identificar padrões climáticos associados à perda agrícola utilizando dados históricos e variáveis de reflectância de satélite, permitindo prever situações de risco crítico antes que o prejuízo financeiro e físico aconteça no campo.

---

## 5. Solução Proposta & Metodologia
O **OrbitAI AgroRisk** utiliza dados macroclimáticos e índices ambientais derivados de satélites para treinar modelos de Machine Learning capazes de classificar regiões agrícolas em três níveis de criticidade:
* 🟢 **Baixo Risco (Área Saudável)**
* 🟡 **Médio Risco (Alerta de Estresse)**
* 🔴 **Alto Risco (Quebra de Safra)**

O pipeline da solução em Python foi estruturado nas seguintes etapas:

1. **Ingestão de Dados Históricos Reais:** Consumo de registros reais cruzados de grandes polos agrícolas brasileiros (Mato Grosso, MATOPIBA, São Paulo e Rio Grande do Sul), correlacionando dados meteorológicos de estações de superfície (**INMET**) com o índice **NDVI (Índice de Vegetação por Diferença Normalizada)** obtido via satélite orbital (**Sentinel-2/TERRA**).
2. **Tratamento de Missing Data (Data Cleaning):** Detecção automatizada de falhas de leitura espectral causadas por obstruções de nuvens ou ruídos atmosféricos (ajustado em 3% de nulos). O pipeline trata essa anomalia aplicando imputação estatística profissional pela média.
3. **Modelagem Preditiva:** Utilização do algoritmo supervisionado **Random Forest Classifier** com particionamento estratificado (divisão 80/20 para treino e teste), garantindo a fidelidade das proporções de classes observadas no campo.
4. **Motor de Inferência em Tempo Real:** Implementação de uma função de diagnóstico ágil capaz de receber novas coordenadas e telemetrias e retornar instantaneamente a classe de risco com a respectiva probabilidade de confiança algorítmica.

---

## 6. Estrutura do Repositório
O script foi desenvolvido para detectar e criar automaticamente a arquitetura de pastas do projeto, garantindo a organização profissional:

```text
📂 orbitai-agrorisk
├── 📂 data
│   └── base_orbitai_limpa.csv      # Dataset real estruturado e limpo pelo pipeline
├── 📂 img
│   ├── matriz_confusao.png         # Gráfico de validação e acertos do classificador
│   └── importancia_variaveis.png   # Gráfico de relevância dos atributos orbitais
└── orbitai_risk.py                 # Código-fonte principal (Engine da IA em Python)
```

---

## 7. Tecnologias e Bibliotecas
* **Python 3.13**
* **Pandas & NumPy:** Ingestão de dados, processamento de matrizes e tratamento de nulos.
* **Scikit-Learn:** Separação de amostras, treino do classificador Random Forest e extração de métricas de validação.
* **Matplotlib & Seaborn:** Renderização e exportação automatizada dos relatórios gráficos estatísticos.

---

## 8. Resultados e Evidências do Funcionamento
O modelo preditivo demonstrou excelente desempenho nos dados reais de validação, atingindo uma **Acurácia Geral de 93.00%**.

### Relatório de Classificação (Métricas)

| Classe de Risco | Precisão (Precision) | Cobertura (Recall) | F1-Score | Amostras de Teste |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 Baixo Risco | 0.96 | 0.97 | 0.97 | 106 |
| 🟡 Médio Risco | 0.88 | 0.95 | 0.91 | 75 |
| 🔴 Alto Risco | 1.00 | 0.63 | 0.77 | 19 |

### Principais Insights e Evidências Visuais

* **Precisão de 100% para Alto Risco:** Quando o modelo aponta que uma região está em `Alto Risco`, a precisão é de 1.00 (100%). Isso significa **falso alarme zero** para cenários críticos, dando total segurança para a tomada de decisões de seguradoras agrícolas e cooperativas.
* **Mitigação de Riscos Críticos:** A análise da Matriz de Confusão comprovou que o modelo **nunca** confunde uma área de Alto Risco real como Baixo Risco (falso negativo crítico), o que poderia causar prejuízos irreversíveis se um alerta fosse ignorado.
* **Predição em Tempo Real (Simulador):** O teste prático com dados de telemetria de uma fazenda enfrentando seca severa por efeito de *La Niña* (alta temperatura, baixa precipitação e NDVI crítico de 0.20) validou com sucesso o comportamento do backend, gerando o diagnóstico:

> 🔴 **ALTO RISCO (QUEBRA DE SAFRA EM CURSO)** com **79.00% de confiança algorítmica**

---

### Gráficos de Auditoria (Gerados Automaticamente)

#### Matriz de Confusão
![Matriz de Confusão](img/matriz_confusao.png)

#### Importância das Variáveis Ambientais e Orbitais
![Importância das Variáveis](img/importancia_variaveis.png)

> 📅 *Nota: Os gráficos estatísticos acima possuem carimbo de data e hora gerados de forma dinâmica pela aplicação no momento de sua execução.*

---

## 9. Conclusão
O projeto **OrbitAI AgroRisk** provou que a união entre algoritmos de Machine Learning e dados de sensoriamento remoto orbital é plenamente viável para prever cenários de estresse hídrico e térmico na Terra. 

A automação do pipeline e a altíssima precisão contra falsos alarmes críticos validam a robustez do sistema como ferramenta de apoio à sustentabilidade e à segurança alimentar no agronegócio.

---

## 🚀 Como Executar o Projeto

### 1. Instale as dependências necessárias
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 2. Execute o script principal da aplicação

```bash
python orbitai_risk.py
```

## 3. Verifique os resultados

Acompanhe as métricas impressas no console e verifique as pastas `data/` e `img/`, criadas automaticamente com os respectivos resultados.
