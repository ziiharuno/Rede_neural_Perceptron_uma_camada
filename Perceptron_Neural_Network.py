
import numpy as np
import pandas as pd

class perceptron():
    
    def __init__(self,learning_rate=0.01, num_treinos=50):
        #Taxa de aprendizagem 
        self.learning_rate = learning_rate 
        #Numero de interações na rede
        self.num_treinos = num_treinos
     
    def train(self, treino_entradas, treino_saidas):
        #vetor de pesos com a bias na primeira posicao
        self.peso_sinaptico = np.zeros(1 + entradas_treinar.shape[1])

        for i in range(self.num_treinos):
            erros = 0
            #Inicio do BackPropagation
            for x, target in zip(treino_entradas,treino_saidas):
                erro = (target - self.think(x))
                erros += int(erro != 0.0)

                #ajustes do vetor de peso
                ajustes = self.learning_rate * erro
                self.peso_sinaptico[1:] += ajustes * x
                self.peso_sinaptico[0] += ajustes

        return self
        
    #funcao de ativacao
    def sigmoid(self,y): 
        return 1 / (1 + np.exp(-y))
    
    def activation(self,entrada): 
        y =  np.dot(entrada, self.peso_sinaptico[1:])
        return self.sigmoid(y)

    #classificacao, retorna 1 caso funcao de ativacao > 0.5, caso contrario retorn -1
    def think(self, entradas):
        return np.where(self.activation(entradas) > 0.5, 1, -1)
    
    #faz a contagem de previsao por acerto, retornando no final a porcentagem de acerto
    def acuracia(self,entradas,saidas):
        i=count=0
        for x,y in zip(entradas,saidas):
            i+=1
            if self.think(x) == int(y): 
                count+=1

        return (count/i)
    
if __name__ == "__main__":
    
    #construindo a classe perceptron como NNP
    NNP = perceptron(num_treinos=30)

    #------------- Tratamento de dados---------------
    
    #lendo os dados 
    df= pd.read_csv("iris.csv")
    
    #Separando 70% das as entradas e saidas para treinar a rede 
    entradas_treinar = df.iloc[0:35,[0,1,2,3]]
    entradas_treinar= entradas_treinar.append(df.iloc[50:85,[0,1,2,3]], ignore_index=True).values

    saidas_treinar = df.iloc[0:35,[4]]
    saidas_treinar= saidas_treinar.append(df.iloc[50:85,[4]], ignore_index=True)
    #binarizando Setosa para -1 e Versicolor para 1
    saidas_treinar = np.where(saidas_treinar == 'Setosa', -1, 1)

    #Separando os 30% restantes das as entradas e saidas para testes e acurácia
    entradas_teste = df.iloc[35:50,[0,1,2,3]]
    entradas_teste = entradas_teste.append(df.iloc[85:100,[0,1,2,3]], ignore_index=True).values
    
    saidas_teste = df.iloc[35:50,[4]]
    saidas_teste= saidas_teste.append(df.iloc[85:100,[4]], ignore_index=True)
    #binarizando Setosa para -1 e Versicolor para 1
    saidas_teste= np.where(saidas_teste == 'Setosa', -1, 1) 
    #-------------Fim do tratamento de dados---------------
    
    
    #Treinando a rede
    NNP.train(entradas_treinar, saidas_treinar)
    
    #resultado dos pesos
    print("Pesos da rede ",NNP.peso_sinaptico)
   
    #acurácia
    print("acuracia de ",NNP.acuracia(entradas_teste,saidas_teste))
    
    #Testando a rede
    entrada_1 = input("sepal length	: ")
    entrada_2 = input("sepal width : ")
    entrada_3 = input("petal length	: ")
    entrada_4 = input("petal width : ")
    entrada = np.array([entrada_1,entrada_2,entrada_3,entrada_4])
    print("A flor é do tipo :")
    if NNP.think(list(map(float,entrada))) == -1:
         print("Setosa")
    elif NNP.think(list(map(float,entrada))) == 1:
         print("Versicolor")
 
# -*- coding: utf-8 -*-

