# Projeto kobe

### Link para o repositório: https://github.com/FTNeves80/Projeto_Kobe

## 1.Diagrama do projeto
![Diagrama](docs/Diagrama.png)

### 2. Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos:
   #### 2.1 Rastreamento de experimentos:
    Com o MLFlow é possível registrar as métricas, artefatos e parâmetros de forma automática. O MLFlow possui uma interface onde você pode consultas e comparar todas as informações registradas tanto atuais como historicamente. Ele cria uma espece de central de informações muito útil para a utilização de controle e registro do ciclo de desenvolvimento dos modelos.
                
   #### 2.2 Funções de treinamento:
    O Scikit-Learn é uma biblioteca poderosa e bastante utilizada no mercado, com uma grande variedade de algoritmos para treinamento de modelos. Já o PyCaret ajuda a automatizar esse processo, permitindo treinar diversos modelos, comparar resultados e ajustar hiperparâmetros com poucas linhas de código. Isso economiza tempo e facilita a obtenção de bons resultados de forma mais eficiente.
                
   #### 2.3 Monitoramento da saúde do modelo:
    O monitoramento da saúde do modelo pode ser feito registrando periodicamente as métricas de performance no MLflow, como AUC, F1-score, Log Loss, entre outras. Isso ajuda a perceber se o modelo está piorando com o tempo (drift de dados ou de conceito). Com isso, conseguimos identificar quando é hora de reavaliar ou atualizar o modelo. Ferramentas como o PyCaret facilitam essa parte, pois permitem reusar facilmente o pipeline de treino e avaliação para comparar com versões anteriores.
    
   #### 2.4 Atualização de modelo:
    Utilizando PyCaret e MLflow, a atualização de um modelo pode ser feita de forma automatizada. É possível configurar um pipeline para retreinar o modelo com dados mais recentes ou com novos parâmetros, registrando uma nova versão no MLflow. A partir daí, você pode promover essa nova versão para produção, mantendo o histórico e comparando com versões anteriores.
                
   #### 2.5 Provisionamento (Deployment).
    O MLflow permite que você sirva o modelo em um servidor, gerando uma API que recebe inputs e retorna previsões. Já o Streamlit pode ser usado para construir uma interface gráfica (front-end) que consome essa API, permitindo a visualização e interação com o modelo de forma simples e intuitiva.

### 4 Com base no diagrama realizado na questão 2, aponte os artefatos que serão criados ao longo de um projeto. Para cada artefato, a descrição detalhada de sua composição.
![Diagrama_artefatos](docs/Diagrama_artefatos.png)
![Diagrama_artefatos](docs/Diagrama_artefatos2.png)


## Overview

This is your new Kedro project, which was generated using `kedro 0.19.12`.

exemplo de sintaxe de MD [Kedro documentation](https://docs.kedro.org) to get started.

### JupyterLab
## Project dependencies


```
Exmplo de marcaçao MD kedro jupyter notebook
```

### Como executar o projeto Ajudar essa respostas
1-kedro run
2-subir o modelo
3-moninotar via mlflow
4-Visualizar o front-end


Analisando a tabela, vamos comparar os principais indicadores de desempenho entre os modelos lr (Logistic Regression) e dt (Decision Tree):

Métrica	Logistic Regression (lr)	Decision Tree (dt)	Melhor
AUC	0.5921	0.5923	Empate (quase idênticos)
Accuracy	0.5682	0.5883	dt
F1 Score	0.5397	0.4644	lr
Recall	0.5296	0.3758	lr
Log Loss	0.6821	0.7171	lr (menor é melhor)
🧠 Análise
F1 Score e Recall são métricas muito importantes quando o dataset é desbalanceado ou você quer equilibrar falsos positivos e falsos negativos.

Log Loss penaliza muito previsões com alta confiança e erradas — e o lr também teve melhor resultado aqui.

A AUC praticamente empatou, com vantagem minúscula para a árvore.

A Accuracy foi maior na dt, mas isso pode ser enganoso em problemas com desequilíbrio nas classes.

✅ Conclusão
Você deve escolher o Logistic Regression (lr), porque:

Ele teve melhor F1, melhor Recall, menor Log Loss.

A pequena perda em Accuracy e AUC não compensa a perda nos outros pontos, principalmente se seu problema é sensível a erros de classificação.

Se quiser, posso te ajudar a registrar isso no MLflow ou usar como base para uma próxima inferência!