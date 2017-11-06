```python
import pandas as pd
from scipy.sparse import csc_matrix

trainData = pd.read_csv('../docword.nips.txt', delimiter=' ')

# csc = csc_matrix((trainData.value.tolist(), (trainData.row.tolist(), trainData.col.tolist())))

print('Training data size:', len(trainData))

trainData.head(5)
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
plt.xlim(0, 100)
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
