# VETORIZADOR DE FONES (EM PYTHON3)
#por Alexandre Menezes Barroso (Linguística -- IEL/UNICAMP) 

#RODAR COMO '$ PYTHONHASHSEED=1234 python3 -i vetorizador.py' NO CMD PARA MANTER INTEGRIDADE DOS TESTES
#EXPLICACOES DETALHADAS NO PDF QUE ACOMPANHA

import matplotlib
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from nltk.tokenize import word_tokenize
import numpy as np
import numpy.random
import seaborn as sns
from mpl_toolkits.mplot3d import proj3d
import re

#---------------------------------------------------------------------------------------------------------------------

#ABRIR CORPORA E REMOÇAO DE CARACTERES NAO-FONETICOS:

#CORPUS FONÉTICO:
with open('corpus_fonetico.txt','r') as Y:
	corpus_fonetico = Y.read().lower() #COLOCA TUDO EM MINUSCULA
	corpus_fonetico = corpus_fonetico[:-1] #REMOVE \n FINAL

	corpus_fonetico = corpus_fonetico.replace('\n','')
	corpus_fonetico = corpus_fonetico.replace('[','')
	corpus_fonetico = corpus_fonetico.replace(']','')
	corpus_fonetico = corpus_fonetico.replace('.','')
	corpus_fonetico = corpus_fonetico.replace(',','')
	corpus_fonetico = corpus_fonetico.replace(':','')
	corpus_fonetico = corpus_fonetico.replace(';','')
	corpus_fonetico = corpus_fonetico.replace('"','')
	corpus_fonetico = corpus_fonetico.replace('&','')
	corpus_fonetico = corpus_fonetico.replace('!','')
	corpus_fonetico = corpus_fonetico.replace('?','')
	corpus_fonetico = corpus_fonetico.replace('$','')
	corpus_fonetico = corpus_fonetico.replace('<','')
	corpus_fonetico = corpus_fonetico.replace('>','')
	corpus_fonetico = corpus_fonetico.replace("'",'')
	corpus_fonetico = corpus_fonetico.replace("ˈ",'')
	corpus_fonetico = corpus_fonetico.replace("-",'')
	corpus_fonetico = corpus_fonetico.replace("(",'')
	corpus_fonetico = corpus_fonetico.replace(")",'')


#CORPUS FONÉTICO 2:
with open('corpus_fonetico2.txt','r') as Z:
	corpus_fonetico2 = Z.read().lower() #COLOCA TUDO EM MINUSCULA
	corpus_fonetico2 = corpus_fonetico2[:-1] #REMOVE \n FINAL'ˈ

	corpus_fonetico2 = corpus_fonetico2.replace('\n','')
	corpus_fonetico2 = corpus_fonetico2.replace('[','')
	corpus_fonetico2 = corpus_fonetico2.replace(']','')
	corpus_fonetico2 = corpus_fonetico2.replace('.','')
	corpus_fonetico2 = corpus_fonetico2.replace(',','')
	corpus_fonetico2 = corpus_fonetico2.replace(':','')
	corpus_fonetico2 = corpus_fonetico2.replace(';','')
	corpus_fonetico2 = corpus_fonetico2.replace('"','')
	corpus_fonetico2 = corpus_fonetico2.replace('&','')
	corpus_fonetico2 = corpus_fonetico2.replace('!','')
	corpus_fonetico2 = corpus_fonetico2.replace('?','')
	corpus_fonetico2 = corpus_fonetico2.replace('$','')
	corpus_fonetico2 = corpus_fonetico2.replace('<','')
	corpus_fonetico2 = corpus_fonetico2.replace('>','')
	corpus_fonetico2 = corpus_fonetico2.replace("'",'')
	corpus_fonetico2 = corpus_fonetico2.replace("ˈ",'')
	corpus_fonetico2 = corpus_fonetico2.replace("-",'')
	corpus_fonetico2 = corpus_fonetico2.replace("(",'')
	corpus_fonetico2 = corpus_fonetico2.replace(")",'')

#---------------------------------------------------------------------------------------------------------------------

#PROCESSAMENTO FONÉTICO DO CORPUS 1 & 2:

#SEPARA INDIVIDUALMENTE CADA CARACTERE DO CORPUS 1 & DO CORPUS 2
fones = list(corpus_fonetico)
fones2 = list(corpus_fonetico2)

#CRIA OS MODELOS DA REDE NEURAL WORD2VEC
modelo_fonetico = Word2Vec(fones, size=100, window=3, min_count=1, sg=0, hs=1, iter=200, batch_words=5, workers=1, seed=1234) 
modelo_fonetico2 = Word2Vec(fones2, size=100, window=3, min_count=1, sg=0, hs=1, iter=200, batch_words=5, workers=1, seed=1234) 

###LEGENDAS:

#sg=0 para CBOW | sg=1 para SKIP-GRAM

#size == DIMENSIONALIDADE DO FEATURE VECTOR

#window == JANELA DO CONTEXTO

#hs == Hierarchical Softmax (MELHOR PARA INFOS FREQUENTES), 1 pra SIM e 0 pra NAO

#iter == EPOCHS/ITERAÇÕES DO TREINO DA REDE NEURAL

#workers == THREADS, MANTER EM 1 PARA REMOVER COMPONENTE ALEATORIO

#seed == QUALQUER NUMERO, NA CONDICAO QUE SEMPRE SE MANTENHA O MESMO ENTRE TESTES (P REMOVER ALEATORIEDADE)


#---------------------------------------------------------------------------------------------------------------------
#NO COMEÇO DE CADA GRÁFICO, REDUZO AS DIMENSOES DOS VETORES (PCA)


#GRÁFICOS DO CORPUS 1:

def tres_dimensoes_fonetico():
	X = modelo_fonetico[modelo_fonetico.wv.vocab]
	pca = PCA(n_components=3)
	resultado = pca.fit_transform(X)
	fig = pyplot.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(resultado[:,0], resultado[:,1],resultado[:,2],'r,',alpha=0.0)
	ax.set_title('MAPEAMENTO DE FONES A VETORES <GRÁFICO TRIDIMENSIONAL>')
	palavras = list(modelo_fonetico.wv.vocab)
	for i, word in enumerate(palavras):
		ax.text(resultado[i, 0], resultado[i, 1], resultado[i,2], '%s' %(str(word)),size=16,zorder=1,color='k',alpha=0.7)
	ax.set_xlabel('Eixo X')
	ax.set_ylabel('Eixo Y')
	ax.set_zlabel('Eixo Z')
	return pyplot.show()

def duas_dimensoes_fonetico():
	Z = modelo_fonetico[modelo_fonetico.wv.vocab]
	pca = PCA(n_components=2)
	resultado = pca.fit_transform(Z)
	pyplot.plot(resultado[:,0], resultado[:,1],'w,')
	pyplot.title('MAPEAMENTO DE FONES A VETORES <GRÁFICO BIDIMENSIONAL>')
	palavras = list(modelo_fonetico.wv.vocab)
	for i, word in enumerate(palavras):
		pyplot.annotate(word, xy=(resultado[i,0], resultado[i,1]),size=16)
	pyplot.xlabel('Eixo X')
	pyplot.ylabel('Eixo Y')
	return pyplot.show()

#---------------------------------------------------------------------------------------------------------------------


#GRÁFICOS DO CORPUS 2:

def tres_dimensoes_fonetico2():
	X = modelo_fonetico2[modelo_fonetico2.wv.vocab]
	pca = PCA(n_components=3)
	resultado = pca.fit_transform(X)
	fig = pyplot.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(resultado[:,0], resultado[:,1],resultado[:,2],'r,',alpha=0.0)
	ax.set_title('MAPEAMENTO DE FONES A VETORES <GRÁFICO TRIDIMENSIONAL>')
	palavras = list(modelo_fonetico2.wv.vocab)
	for i, word in enumerate(palavras):
		ax.text(resultado[i, 0], resultado[i, 1], resultado[i,2], '%s' %(str(word)),size=16,zorder=1,color='k',alpha=0.7)
	ax.set_xlabel('Eixo X')
	ax.set_ylabel('Eixo Y')
	ax.set_zlabel('Eixo Z')
	return pyplot.show()

def duas_dimensoes_fonetico2():
	Z = modelo_fonetico2[modelo_fonetico2.wv.vocab]
	pca = PCA(n_components=2)
	resultado = pca.fit_transform(Z)
	pyplot.plot(resultado[:,0], resultado[:,1],'w,')
	pyplot.title('MAPEAMENTO DE FONES A VETORES <GRÁFICO BIDIMENSIONAL>')
	palavras = list(modelo_fonetico2.wv.vocab)
	for i, word in enumerate(palavras):
		pyplot.annotate(word, xy=(resultado[i,0], resultado[i,1]),size=16)
	pyplot.xlabel('Eixo X')
	pyplot.ylabel('Eixo Y')
	return pyplot.show()

#---------------------------------------------------------------------------------------------------------------------


#GRÁFICOS COMPARATIVOS (CORPUS 1 + CORPUS 2):

def tres_dimensoes_fonetico_conjunto():
	X = modelo_fonetico[modelo_fonetico.wv.vocab]
	pca = PCA(n_components=3)
	resultado = pca.fit_transform(X)

	Z = modelo_fonetico2[modelo_fonetico2.wv.vocab]
	pca2 = PCA(n_components=3)
	resultado2 = pca2.fit_transform(Z)

	fig = pyplot.figure()

	ax = fig.add_subplot(111, projection='3d')

	ax.plot(resultado[:,0], resultado[:,1],resultado[:,2],'ro',alpha=0.8,label='I. AMERICANO')
	ax.plot(resultado2[:,0], resultado2[:,1],resultado2[:,2],'bo',alpha=0.8,label='I. BRITÂNICO')

	ax.set_title('<GRÁFICO TRIDIMENSIONAL>')

	palavras = list(modelo_fonetico.wv.vocab)
	for i, word in enumerate(palavras):
		ax.text(resultado[i, 0], resultado[i, 1], resultado[i,2], '%s' %(str(word)),size=16,zorder=1,color='r',alpha=0.7)

	palavras2 = list(modelo_fonetico2.wv.vocab)
	for i, word in enumerate(palavras2):
		ax.text(resultado2[i, 0], resultado2[i, 1], resultado2[i,2], '%s' %(str(word)),size=16,zorder=1,color='b',alpha=0.7)

	pyplot.legend(loc='upper right')
	
	ax.set_xlabel('Eixo X')
	ax.set_ylabel('Eixo Y')
	ax.set_zlabel('Eixo Z')
	return pyplot.show()

def duas_dimensoes_fonetico_conjunto():

	Z = modelo_fonetico[modelo_fonetico.wv.vocab]
	pca = PCA(n_components=2)
	resultado = pca.fit_transform(Z)

	Y = modelo_fonetico2[modelo_fonetico2.wv.vocab]
	pca2 = PCA(n_components=2)
	resultado2 = pca2.fit_transform(Y)

	pyplot.plot(resultado[:,0], resultado[:,1],'r+',label='I. AMERICANO')
	pyplot.plot(resultado2[:,0], resultado2[:,1],'b+',label='I. BRITÂNICO')

	pyplot.title('MAPEAMENTO DE FONES A VETORES <GRÁFICO BIDIMENSIONAL>')
	palavras2 = list(modelo_fonetico2.wv.vocab)
	for i, word in enumerate(palavras2):
		pyplot.annotate(word, xy=(resultado[i,0], resultado[i,1]),color='red',size=16,)

	palavras = list(modelo_fonetico.wv.vocab)
	for i, word in enumerate(palavras):
		pyplot.annotate(word, xy=(resultado2[i,0], resultado2[i,1]),color='blue',size=16,)

	pyplot.legend(loc='upper right')

	pyplot.xlabel('Eixo X')
	pyplot.ylabel('Eixo Y')
	return pyplot.show()


###FIM
