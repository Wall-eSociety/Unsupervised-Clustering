```python
print('kos enron nips')
base = input()
```

```python
# Configure to show multiples outputs from a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
from scipy.sparse import csc_matrix

doc = open('../docword.{}.txt'.format(base))
docs_count = int(doc.readline().replace('\n',''))
dictionary_count = int(doc.readline().replace('\n',''))
word_count = int(doc.readline().replace('\n',''))
trainData = pd.read_csv(doc, delimiter=' ', names=['row', 'col', 'value'])
doc.close()
# csc = csc_matrix((trainData.value.tolist(), (trainData.row.tolist(), trainData.col.tolist())))

print('docs: {}\ndictionary_count: {}\nwords: {}'.format(docs_count, dictionary_count, word_count))

trainData.head()
# Read words of dictionary
vocabulary = pd.read_csv('../vocab.{}.txt'.format(base), names=['vocab', 'count', 'sum'])
# Set the vocabulary index row begin in 1 instead 0
vocabulary.index = vocabulary.index+1


vocabulary.head()
```

```python
%%time

InteractiveShell.ast_node_interactivity = "last"
pivot = trainData.pivot_table('value', ['row'], 'col').fillna(0)
none_df = pd.DataFrame(0, range(1, pivot.shape[0]+1), range(1, vocabulary.shape[0]+1))
pivot = pivot.combine_first(none_df)
print(pivot.shape, vocabulary.shape)
pivot.columns = vocabulary.vocab.values
X = pivot.values
pivot.head()
```

```python
pivot = pivot.drop(vocabulary[ (vocabulary['sum'] > 100) ].vocab.values, axis=1)
X = pivot.values
pivot
```

```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X)

print('X_train before PCA:\n', X_train)
```

```python
%%time
from sklearn.decomposition import PCA
# PCA Dimensionality Reduction

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_

explained_variance
print('X_train after PCA to 2 dimensions:\n', X_train)
```

```python
# Count = Number of unique documents word was used on 
# Sums = Number of times word was used throughout the documents

x = trainData.groupby(['col'])['value']
counts, sums = x.count(), x.sum()

counts.head()
sums.head()
```

```python
vocabulary['count'] = counts
vocabulary['sum'] = sums
vocabulary[ (vocabulary['count'] < 170) & (vocabulary['sum'] < 100)]
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

```python
# pivot = pivot.drop(vocabulary[ (vocabulary['sum'] > 100) ].vocab.values, axis=1)
# X = pivot.values
# pivot
```

```python
filtered_df = []
max_value = 200
min_value = 50

for index, row in vocabulary.iterrows():
    if((row['count'] > 200 and row['sum'] > 100) or (row['count'] < 50 and row['sum'] > 200)):
        print(row['vocab'])
    else:
        filtered_df.append((row['vocab'], row['count'], row['sum']))

filtered_df = pd.DataFrame(filtered_df, columns=['vocab', 'count', 'sum'])
filtered_df
```

# K Means
## Como escolher o número de clusters?

### Método Elbow

Escolhemos o número de *clusters* de acordo com o primeiro ponto do gráfico que
possui menor diferença se comparado com seus vizinhos.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

```python
%%time
wcss = []
for i in range(1, 9):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1, random_state=0, n_jobs=-1)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=[15, 5])
plt.plot(range(1,9), wcss)
plt.scatter(3, 79000, c='black', s=100, alpha=1.0)
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
K = 3
kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
y_kmeans = kmeans.fit(X_train)
```

```python
# Plot for 2D dataset

# %%time
plt.figure(figsize=[15, 7])
plt.scatter(X_train[y_kmeans == 0, 0], X_train[y_kmeans == 0, 1], s = 100, c = 'red', marker='v', label = 'Cluster 1')
plt.scatter(X_train[y_kmeans == 1, 0], X_train[y_kmeans == 1, 1], s = 100, c = 'blue', marker='*', label = 'Cluster 2')
plt.scatter(X_train[y_kmeans == 2, 0], X_train[y_kmeans == 2, 1], s = 100, c = 'orange', marker='s', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'cyan', marker='o', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters')
plt.legend()
plt.show()
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
inertia = \sum_{i=1}^K \sum_{x \in C_i}^n ||x - C_i||^2 \\
inter = \frac{intertia}{N} \\
inter = min(||z_i - z_j||^2) \quad onde \quad \big\{ i = 1 ... ( K - 1 ) \ e\  j
= ( i + 1 ) ... K
\end{equation*}

Nas equações acima, __K__ é o número de clusters, __C_i__ é um centroid cluster
qualquer, __x__ é uma amostra dos dados, __z__ é a representação dos centroids
de
um cluser e __N__ é o total de clusters.

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

max_intra_cluster = kmeans.inertia_ / K
min_inter_cluster = inter_cluster(kmeans)
print((min_inter_cluster, max_intra_cluster), max_intra_cluster / min_inter_cluster)
```

# Resultado

Neste problema de clusterização, não é possível determinar se o modelo fez o
trabalho correto em agrupar os dados ou não. Basea-se então na modelagem
matemática da análise inter cluster e intra cluster para trazer mais confiança.
Como é possível observar no valor da célula anterior da relação intra/inter. Em
que busca-se minimizar o valor de intra e maximizar o valor do inter.

A seguir, mostraremos um gráfico representativo de cada cluster.

```python
InteractiveShell.ast_node_interactivity = 'last'
plt.figure(figsize=[15, 7])

for cluster in kmeans.cluster_centers_:
    plt.plot(np.arange(1, len(cluster) + 1), cluster)
plt.show()

```

## Palavras representativas

Com a caracterização dos clusters vistos acima, é perceptível que há algumas
palavras que mais caracterizam um determinado cluster. Então, para tentar
averiguar se a clusterização foi realizada com uma boa acurácia, iremos tentar
identificar se as palavras que aparecem irão ter algum sentido lógico para
representar o cluster.

```python
import collections
clusters_amount = collections.Counter(kmeans.labels_)
clusters = pd.DataFrame(kmeans.cluster_centers_)

# indexes = clusters.iloc[0, :].sort_values(ascending=False).index
# vocabulary.iloc[indexes]

for cluster in clusters.iterrows():
    idxs = cluster[1].sort_values(ascending=False)[:10].index
    cluster_size = clusters_amount[cluster[0]]
    top10_words = vocabulary.iloc[idxs].vocab.str.cat(sep=', ')
    print("cluster {} ({}): {}".format(cluster[0], cluster_size, top10_words))
```

# Referências

[1] [Determination of Number of Clusters in K-Means Clustering and Application
in Colour Image Segmentation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=
10.1.1.587.3517&rep=rep1&type=pdf)

[2] [Silhouette](https://cs.fit.edu/~pkc/classes/ml-internet/silhouette.pdf)
