Esse é um projeto remake de https://github.com/JVictorDias/DeepCars em Python.
Da mesma forma do original decidi fazer para facilitar os estudos em redes neurais.

![Deep cars running]([http://url/to/img.png](https://github.com/rmaugusto/deepcars/blob/master/deepcars.gif))

Eu mantive o código original no diretorio "original" para facilitar a tradução.

Referências:
* https://github.com/JVictorDias/DeepCars
* https://www.youtube.com/watch?v=gnfkfUQvKDw

# Funcionalidades iguais:
* Rede neural (18, 6, 4) + 1 bias com função de ativação ReLU, levou 138 gerações para treinar o modelo
* *Rede neural (18, 6, 6, 4) + 1 bias com função de ativação ReLU, levou 60 gerações para treinar o modelo
* Função de mutação aleatória
* Grafico da rede neural do melhor carro
* Obstaculos
* Zonas de aceleração e desaceleração
* Armazenamento do melhor cerebro
* Laser destruidor com aceleração
* Serialização da matriz de distancias na primeira execução (distance_matrix.npy)

# Funcionalidades a mais:
* População 200 carros (Por capacidade inferior do meu notebook)
* Estatisticas na tela
* Identificador do melhor carro
* Renderização do mapa após chegar na linha de chegada (Apertar a tecla SPACE força a exibição)
* Quando chega na linha de chegada exibe o laiser destruidor e os 18 sensores
* O algoritimo de distancia foi usada dijkstra
* No final da execução é gravado deepcars.prof para estatisticas de performance

# Funcionalidades não implementadas:
* Graficos maneiros de explosão do laser destruidor
* Otimizações graficas com GPU
* Otimizações de IA com frameworks como PyTorch

# Instalação
{ pip install --user pipenv }
{ pipenv install }
{ python deepcars/main.py }
