import numpy as np
import matplotlib.pyplot as plt

#Definimos el prior con las regiones en las que han de estar los parametros
def logprior(v):
    p = -np.inf
    for i in range(len(v)):
        if v[i] > 0 and v[i] < 100:
            p = 0.0
        else:
            p= - np.inf
            break
    return p

#El modelo matematico que sigue la caida libre de un objeto con una velocidad inicial
def modelo(v, x):
    n_dim = len(v)-1
    notas = np.ones(len(x))
    notas = v[0]
    for i in range(n_dim):
        notas += v[i+1]*x[i]
    return notas

#Comparacion de la propuesta con los datos experimentales en logaritmo
def likelihood(y, x, v, sigmas):
    L = np.ones(len(y))
    for i in range(len(y)):
        L *= (1.0/np.sqrt(2.0*np.pi*sigmas[i]**2))*np.exp(-0.5*(modelo(v,x[i,:])-y[i])**2/(sigmas[i]**2))
    return L

def logposterior(L,P,y):
    print(L)
    post =  np.multiply(L,P)
    evidencia = np.trapz(post, y)
    logpost = np.log(post/evidencia)
    return  logpost

#Datos observados
data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]
Sigmas = np.ones(len(Y))*0.1

#Numero con el que se realizara el MCMH
N = 100
lista_v = [[1,1,1,1]]
sigma_v=1
for i in range(1,N):
    #Proponemos un nuevo beta en funcion del anterior mas una distribucion normal
    propuesta_v  = lista_v[i-1] + np.random.normal(loc=0.0, scale=sigma_v, size=4)
    
    #Se crean los Posteriors nuevo, y viejo con los anteriores, con el fin de tener el criterio de comparacion
  
    logposterior_viejo = logposterior(likelihood(Y,X,lista_v[i-1],Sigmas),logprior(lista_v[i-1]),Y)
    logposterior_nuevo = logposterior(likelihood(Y,X,propuesta_v,Sigmas),logprior(propuesta_v),Y)
                      
    #criterio de comparacion
    r = min(np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    
    #indexamos la propuesta 
    if(alpha<r):
        lista_v.append(propuesta_v)
    #No se indexa e indexamos el anterior
    else:
        lista_v.append(lista_v[i-1])

#Convertimos el todo en arrays para poder graficarlo en el histograma
lista_v = np.array(lista_v)
#print(lista_v)
#lista_theta = np.array(lista_theta)

#El histograma con los datos de los bins para encontrar el maximo
#a, b, c = plt.hist(lista_v, bins=100, density=True)

#Encontramos el maximo del histograma (la maxima probabilidad)
#bin_max = np.where(a == a.max())
#desv = np.std(lista_v)
#mV = np.mean(lista_v)

#plt.title('Velocidad de salida '+ str(b[bin_max][0]) + ' m/s. El valor medio es ' + str(mV) + ' m/s. Y la desviacion estandar es ' + str(desv))
#Guardamos la figura
#plt.savefig('casiqueno.pdf')