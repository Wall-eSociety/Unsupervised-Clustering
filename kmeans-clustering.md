```python
# Configure to show multiples outputs from a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
from scipy.sparse import csc_matrix

doc = open('../docword.kos.txt')
docs_count = int(doc.readline().replace('\n',''))
dictionary_count = int(doc.readline().replace('\n',''))
word_count = int(doc.readline().replace('\n',''))
trainData = pd.read_csv(doc, delimiter=' ', names=['row', 'col', 'value'])
doc.close()
# csc = csc_matrix((trainData.value.tolist(), (trainData.row.tolist(), trainData.col.tolist())))

print('docs: {}\ndictionary_count: {}\nwords: {}'.format(docs_count, dictionary_count, word_count))

trainData.head(5)
# Read words of dictionary
vocabulary = pd.read_csv('../vocab.kos.txt', names=['vocab'])
# Set the vocabulary index row begin in 1 instead 0
vocabulary.index = vocabulary.index+1

vocabulary.head()
```

```python
x = trainData.groupby(['col'])['value']
counts, sums = x.count(), x.sum()

counts.head()
sums.head()
```

```python
vocabulary['count'] = counts
vocabulary['sum'] = sums

vocabulary

```

```python
X = trainData.iloc[:, [1,2]]
```

```python
docId = trainData.iloc[:, 0]
vocabularyId = trainData.iloc[:, 1]
words = trainData.iloc[:, 2]
```

```python
%%time
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.figure(figsize=[15, 7])
plt.scatter(vocabularyId, words, c='darkgreen', marker='o', s=20, alpha=0.8, label='Word Count')
plt.ylabel('Word count')
plt.xlabel('Word ID')
plt.xlim(0, dictionary_count+1)
plt.legend()
plt.title('Word Histogram first 100 docs')
plt.show()
```

# K Means
## Como escolher o número de clusters?

### Método Elbow

Escolhemos o número de *clusters* de acordo com o primeiro ponto do gráfico que
possui menor diferença se comparado com seus vizinhos.

```python
%%time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0, n_jobs=-1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=[15, 5])
plt.plot(range(1,11), wcss)
plt.title('Método Elbow')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()
```

## Aplicando o melhor número de clusters:
O número de *clusters* determina diretamente o quão complexo o modelo ficara.
Determinar a quantidade de *clusters* a se utilizar evita que o modelo produzido
caia em:

* **Underfitting**: Pode acontecer quando o modelo é produzido com menos
*clusters* que o necessário, o que torna o modelo altamente generalizado.

* **Overfitting**: Pode acontecer quando o modelo é produzido com mais
*clusters* que o necessário, o que torna o modelo altamente complexo e
específico para a base de treinamento

```python
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
y_kmeans = kmeans.fit_predict(X)
```

```python
# Plot for 2D dataset

# %%time
# plt.figure(figsize=[15, 7])
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'red', marker='v', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'blue', marker='*', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'orange', marker='s', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'cyan', marker='o', label = 'Cluster 4')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
# plt.title('NIPS Clusters')
# plt.legend()
# plt.show()
```

# Validando o modelo

Para avaliar a performance do modelo, deve-se fazer uma análise dos dados e a
relação dos clusters gerados com os pontos disponíveis. Para tal, deve-se fazer
uma análise dos dados dentro de cada cluster (intra cluster) e a relação entre
os diferentes clusters (inter cluster)[1]. Outra métrica possível para realizar
uma medição dos clusters é avaliar o _silhouette coeficient_ [2].

## Intra e Inter

A biblioteca do scikit learn fornece uma métrica chamada
[inertia\_](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMe
ans.html#sklearn.cluster.KMeans) que calcula a soma do quadrado da distância dos
dados para o cluster mais próximo. Isso quer dizer que quanto menor este valor,
mais próximo dos centroids do cluster estão as amostras. Essa métrica é
utilizada para escolher o melhor K no método [elbow](#Método-Elbow).

Os cálculos de inter e intra cluster são descritos em [1].
\begin{equation*}
inertia = intra = \sum_{i=1}^K \sum_{x \in C_i}^n ||x - C_i||^2 \\
inter = min(||z_i - z_j||^2) \quad onde \quad \big\{ i = 1 ... ( K - 1 ) \ e\  j
= ( i + 1 ) ... K
\end{equation*}

Nas equações acima, __K__ é o número de clusters, __C_i__ é um centroid cluster
qualquer, __x__ é uma amostra dos dados e z é a representação dos centroids de
um cluser.

## Silhouette Coefficient

É uma métrica que calcula junto os valores de coesão inter e intra cluster.  O
cálculo para esta métrica é destrito como
\begin{equation*}
    silhouette = \frac{(b - a)}{max(a,b)}
\end{equation*}

em que o valor de __a__ é o valor da média da distância intra-cluster e __b__ é
a média do cluster mais próximo a amostra.

```python
%%time

from scipy.spatial import distance_matrix
inter_distances = None
def inter_cluster(kmeans_model):
    clusters = kmeans_model.cluster_centers_
    inter_distances = distance_matrix(clusters, clusters)
    inter_distances[ inter_distances == 0 ] = np.inf
    return inter_distances.min()

min_inter_cluster = inter_cluster(kmeans)
print(min_inter_cluster, kmeans.inertia_ / min_inter_cluster) 
```

# Referências

[1] [Determination of Number of Clusters in K-Means Clustering and Application
in Colour Image Segmentation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=
10.1.1.587.3517&rep=rep1&type=pdf)

[2] [Silhouette](https://cs.fit.edu/~pkc/classes/ml-internet/silhouette.pdf)
