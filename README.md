# Vetorizador fonético
Projeto de PIBIC (UNICAMP, 2018). Vetorizador que usa como input banco de dados fonético e tem como output um modelo bi- ou tridimensional de relações estatísticas entre fones. Assim, tenta traçar, por meio de aprendizagem de máquina, relações fonéticas relevantes entre esses fones apenas com seu uso nas palavras.

Utiliza a versatilidade do algoritmo de aprendizagem de máquina Word2Vec para mapear fones a vetores, e não palavras. Se palavras mantém relações semânticas nesse modelo de algoritmo, é possivel estabelecer relações fonéticas?

O script formata corpora para serem usados de maneira comparativa, usa o algoritmo Word2Vec para mapear cada fone a um vetor de 100 dimensões, reduz a quantidade de dimensões para 2 ou 3 por meio da análise de componentes principais (PCA) e gera um gráfico.

Demonstração do funcionamento: https://youtu.be/sKQi3t46JIk

![exemplo](https://github.com/alexandre-barroso/vetorizador_fonetico/blob/master/exemplo.gif)
