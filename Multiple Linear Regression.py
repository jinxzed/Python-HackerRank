import numpy
import math

def getxy():
    features, n = input().split()
    x, y = [] , []

    for i in range(int(n)):
        data = [float(j) for j in input().split()]
        x.append(data[:-1])
        y.append(data[-1])
    x = numpy.array(x)
    y = numpy.array(y)
    
    return (x,y)

def get_xnew():
    n1 = int(input())
    xnew = []
    for i in range(n1):
        xnew.append([float(j) for j in input().split()])
    return numpy.array(xnew)

def calc_theta(x, y):
    x = numpy.column_stack( (numpy.ones(x.shape[0]), x) )
    inv = numpy.linalg.inv(x.transpose().dot(x))
    dotxy = x.transpose().dot(y)
    theta = inv.dot(dotxy)
    return theta

def make_predict(x, theta):
    x = numpy.column_stack( (numpy.ones(x.shape[0]), x) )
    ypred = x.dot(theta)

    return ypred

def mlr(x, y):

    return calc_theta(x, y)

x, y = getxy()

ypred = make_predict( get_xnew(), mlr(x,y) )

for i in ypred:
    print(i)