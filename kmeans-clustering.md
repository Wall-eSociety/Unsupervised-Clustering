```python
import pandas as pd

trainData = pd.read_csv('../docword.nips.txt', delimiter=' ')

print('Training data size:', len(trainData))
```

```python
X = trainData.iloc[:, :].values
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
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0, n_jobs=-1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Método Elbow')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

print('Melhor número de clusters: 4')
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
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
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
