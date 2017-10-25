# Unsupervised-Clustering

Este é o quarto problema da disciplina de machine learning - FGA-UNB

## Objetivo

Este problema tem o objetivo atacar a area de Machine Learning não supervisionado. Mais especificamente Clusterização. Mais especificamente serão utilizados dados do tipo de *Texto*.

Os dados utilizados nesta interação da disciplina do grupo podem ser encontrados [Aqui.](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words)

## Referencial teorico

###Clusterização

Clusterização consiste no agrupamento de dados em grupos semelhantes e significativos (*Clusters*), dessa forma capturando a estrutura natural dos dados apresentados.
O objetivo natural da clusterização é que os grupos intra cluster sejam semelhantes entre si, e que os clusters sejam tenham diferenças significativas com relação aos outros clusters. Quanto maior for o grau de semelhança intra cluster e a diferença entre cluster, melhor é considerado o modelo montado.

A sobreposição de grupos é chamado de ruido. Dependendo da analise é necessário retirar as amostras que causam ruido. Caso os grupos estejam sobrepostos, é possível que a configuração dos parametros do algorítmo não esteja corretos ou o algoritmo utilizado não seja o recomendado.

Para mais informações acerca de clusterização acesse este [artigo](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.99.7799&rep=rep1&type=pdf).

### Quantificação vetorial

Quantificação vetorial é uma técnica para processamento de sinais que permite a modelagem de funções de densidade de probabilidade pela distribuição de vetores protótipos. Ela trabalha dividindo um grande número de pontos fixos(vetores) em grupos que tem aproximadamente o mesmo número de pontos entre eles. Cada grupo é representado por o seu ponto centroide, como em k-means e outros algoritmos de clusterização.

Kohonen(Biblioteca para quantificação vetorial) (https://pypi.python.org/pypi/kohonen/1.1.2).
