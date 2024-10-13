# Criando o meu próprio algoritimo de regressão linear

from numpy import *

class LinearRegression:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.__correlation_coeficient = self.__correlacao()
            self.__inclination = self.__inclinacao()
            self.__intercept = self.__interceptacao()

        def __correlacao(self): 
            covariacao = cov(self.x, self.y, bias=True)[0][1]
            variancia_x = var(self.x)
            variancia_y = var(self.y)
            return covariacao / sqrt(variancia_x * variancia_y)

        def __inclinacao(self):
            desvio_x = std(self.x)
            desvio_y =  std(self.y)
            return self.__correlation_coeficient * (desvio_y/desvio_x)
        

        def __interceptacao(self):
            media_x = mean(self.x)
            media_y = mean(self.y)
            return media_y - self.__inclination * media_x

        def previsao(self, valor): 
            return self.__intercept + (self.__inclination * valor)

x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])

regressao_linear = LinearRegression(x, y)
previsao = regressao_linear.previsao(15)



print(f'A previsão feita através da regressão Linear é {previsao}')