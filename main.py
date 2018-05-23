import matriz as mtz
import numpy as np
from scipy.sparse.linalg import cg
import time


def grad_conj(m, b, x0=None, eps=1e-5, maxiter=100):

    n = len(b)
    if not x0:
        x0 = np.zeros(n)
    grad0 = np.dot(m, x0) - b        
    d = - grad0                      
    
    for i in range(maxiter):
        alpha = np.dot(grad0.T, grad0) / np.dot(np.dot(d.T, m), d)
        x0 = x0 + d*(alpha[0])
        gradi = grad0 + np.dot(m*(alpha[0]), d)

        print(i, np.linalg.norm(gradi))

        if np.linalg.norm(gradi) < eps:
            return x
        betai = np.dot(gradi.T, gradi) / np.dot(grad0.T, grad0)
        d = - gradi + (betai[0])*d
        grad0 = gradi
    return x0


if __name__ == '__main__':
    coeficientes = input("Insira o nome do primeiro arquivo (coeficientes da Matriz):")
    valores = input("Insira o nome do segundo arquivo (valores b):")
    mat = np.array(mtz.criarmatriz(coeficientes))
    val = np.array(mtz.criarmatriz(valores))
    start_time = time.time()
    gc = grad_conj(mat, val)
    res = open("resposta.txt", "w")
    for i in range(len(gc[0])):
        res.write(str(gc[i][0]))
        print(gc[i][0])
        res.write("\n")
    res.close()
    print("--- %s segundos de execução ---" % (time.time() - start_time))
