```{.python .input  n=1}
import pandas as pd
from scipy.sparse import csc_matrix

trainData = pd.read_csv('../docword.nips.txt', delimiter=' ')

csc=csc_matrix( (trainData.value.tolist(), (trainData.row.tolist(), trainData.col.tolist())))

print('Training data size:', len(trainData))
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Training data size: 746316\n"
 }
]
```

```{.python .input  n=2}
X = trainData.iloc[:, :].values
X = csc
```

# K Means
## Como escolher o número de clusters?

### Método Elbow

Escolhemos o número de *clusters* de acordo com o primeiro ponto do gráfico que
possui menor diferença se comparado com seus vizinhos.

```{.python .input  n=3}
%%time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Método Elbow')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

print('Melhor número de clusters: 4')
```

```{.json .output n=3}
[
 {
  "ename": "KeyboardInterrupt",
  "evalue": "",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m-------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m                       Traceback (most recent call last)",
   "\u001b[0;32m<timed exec>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n",
   "\u001b[0;32m~/UnB/ml/.env/lib/python3.6/site-packages/sklearn/cluster/k_means_.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y)\u001b[0m\n\u001b[1;32m    894\u001b[0m                 \u001b[0mtol\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtol\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mrandom_state\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy_x\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy_x\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    895\u001b[0m                 \u001b[0mn_jobs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mn_jobs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0malgorithm\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0malgorithm\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 896\u001b[0;31m                 return_n_iter=True)\n\u001b[0m\u001b[1;32m    897\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    898\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/UnB/ml/.env/lib/python3.6/site-packages/sklearn/cluster/k_means_.py\u001b[0m in \u001b[0;36mk_means\u001b[0;34m(X, n_clusters, init, precompute_distances, n_init, max_iter, verbose, tol, random_state, copy_x, n_jobs, algorithm, return_n_iter)\u001b[0m\n\u001b[1;32m    361\u001b[0m                                    \u001b[0;31m# Change seed to ensure variety\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    362\u001b[0m                                    random_state=seed)\n\u001b[0;32m--> 363\u001b[0;31m             for seed in seeds)\n\u001b[0m\u001b[1;32m    364\u001b[0m         \u001b[0;31m# Get results with the lowest inertia\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    365\u001b[0m         \u001b[0mlabels\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minertia\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcenters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_iters\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mresults\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/UnB/ml/.env/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, iterable)\u001b[0m\n\u001b[1;32m    787\u001b[0m                 \u001b[0;31m# consumption.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    788\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_iterating\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 789\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mretrieve\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    790\u001b[0m             \u001b[0;31m# Make sure that we get a last message telling us we are done\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    791\u001b[0m             \u001b[0melapsed_time\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtime\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtime\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_start_time\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/UnB/ml/.env/lib/python3.6/site-packages/sklearn/externals/joblib/parallel.py\u001b[0m in \u001b[0;36mretrieve\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    697\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    698\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mgetattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_backend\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'supports_timeout'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 699\u001b[0;31m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_output\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mextend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mjob\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    700\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    701\u001b[0m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_output\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mextend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mjob\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/lib64/python3.6/multiprocessing/pool.py\u001b[0m in \u001b[0;36mget\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    636\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    637\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 638\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwait\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    639\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mready\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    640\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mTimeoutError\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/lib64/python3.6/multiprocessing/pool.py\u001b[0m in \u001b[0;36mwait\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    633\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    634\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mwait\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 635\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_event\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwait\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    636\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    637\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mget\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/lib64/python3.6/threading.py\u001b[0m in \u001b[0;36mwait\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    549\u001b[0m             \u001b[0msignaled\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_flag\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    550\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0msignaled\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 551\u001b[0;31m                 \u001b[0msignaled\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_cond\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwait\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    552\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0msignaled\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    553\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/usr/lib64/python3.6/threading.py\u001b[0m in \u001b[0;36mwait\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    293\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m    \u001b[0;31m# restore state no matter what (e.g., KeyboardInterrupt)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    294\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mtimeout\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 295\u001b[0;31m                 \u001b[0mwaiter\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0macquire\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    296\u001b[0m                 \u001b[0mgotit\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    297\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
  ]
 }
]
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

```{.python .input}
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0, n_jobs=-1)
y_kmeans = kmeans.fit_predict(X)
```

```{.python .input}
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
