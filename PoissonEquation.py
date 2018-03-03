# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.constants import pi
from mpl_toolkits.mplot3d import axes3d


def read(filename):
    with open(filename) as f:
        lines = f.readlines()
        rows = [np.fromstring(line, sep = ' ', dtype=int) for line in lines]
    Den = np.zeros((rows[-1].size,rows[-1].size), dtype=int)
    for i, row in enumerate(rows):
        Den[i,:row.size] = row
    
    return Den

a=100
b=100


class Plotter: 
    def __init__(self, title, xlabel, ylabel): 
        self.fig = plt.figure(facecolor='w')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title) 
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def plot(self, x, y, Mat): 
        return self.ax.pcolormesh(x, y, Mat)

    def show(self):
        plt.show()
        
class AdvancedPlotter:
    def __init__(self, title, xlabel, ylabel):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title) 
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
    def plot(self, xx, yy, Mat):
        x,y=np.meshgrid(xx,yy)
        CS=plt.contour(x,y,Mat,colors=('indigo','purple','b','m','violet','aqua'), linewidths=0.8)
        return self.ax.plot_wireframe(x, y, Mat, color='k', linewidth=0.3)

    def show(self):
        pylab.show()
        
class SolvePoisson:
    def __init__(self,M,N,Step,Density,NumberOfIterrations,LeftBorder,RightBorder,TopBorder,BottomBorder,Epsilon,plotter):
        self.n=N
        self.m=M
        self.s=Step
        self.P=Density
        self.ni=NumberOfIterrations
        self.lb=LeftBorder
        self.rb=RightBorder
        self.tb=TopBorder
        self.bb=BottomBorder
        self.eps=Epsilon
        self.plotter=plotter
        
    def Analitic(self, Num):
        A=np.zeros((self.m, self.n))
        Ak=np.zeros((self.m, self.n))
        for k in range(Num):
            for i in range(self.n):
                for j in range(self.m):
                    Ak[j,i]=400./(pi*(2*k+1))*(np.cosh(pi/self.n*(2*k+1)*i)-1/np.tanh(pi*(2*k+1))*np.sinh(pi/self.n*(2*k+1)*i))*np.sin(pi/self.m*(2*k+1)*j)
            A=A+Ak
        x=np.linspace(0.,self.n*self.s,self.n)
        y=np.linspace(0.,self.m*self.s,self.m)
        plotter.plot(x,y,A)
        return (A)
        
    def Jakobi(self):
        A=np.zeros((self.m, self.n))
        A[0,:]=self.tb
        A[-1,:]=self.bb
        A[:,0]=self.lb
        A[:,-1]=self.rb
        for i in range(self.ni):
            B=0.25*( A[0:-2,1:-1] + A[1:-1,0:-2] + A[1:-1,2:] + A[2:,1:-1])+pi*self.P[1:-1,1:-1]*self.s**2
            A[1:-1,1:-1]=B

        
        x=np.linspace(0.,self.n*self.s,self.n)
        y=np.linspace(0.,self.m*self.s,self.m)
        plotter.plot(x,y,A)
        return (A)
    
    def GaussZeudel(self):
        A=np.zeros((self.m, self.n))
        A[0,:]=self.tb
        A[-1,:]=self.bb
        A[:,0]=self.lb
        A[:,-1]=self.rb
        for k in range(self.ni):
            Sp1=np.trace(A)
            for i in range(self.m-2):
                for j in range(self.n-2):
                    A[i+1,j+1]=0.25*(A[i,j+1]+A[i+1,j]+A[i+1,j+2]+A[i+2,j+1])+pi*self.P[i+1,j+1]*self.s**2
            Sp2=np.trace(A)
            delta=np.absolute(Sp2-Sp1)
            if (delta<=self.eps):
                break
                
        x=np.linspace(0.,self.n*self.s,self.n)
        y=np.linspace(0.,self.m*self.s,self.m)
        plotter.plot(x,y,A)
        return (A)
    
    def Relax(self,omega):
            A=np.zeros((self.m, self.n))
            R=np.zeros((self.m, self.n))
            A[0,:]=self.tb
            A[-1,:]=self.bb
            A[:,0]=self.lb
            A[:,-1]=self.rb
            for i in range(self.ni):
                Sp1=np.trace(A)
                B=0.25*( A[0:-2,1:-1] + A[1:-1,0:-2] + A[1:-1,2:] + A[2:,1:-1])+pi*self.P[1:-1,1:-1]*self.s**2
                R[1:-1,1:-1]=B-A[1:-1,1:-1]
                A=A+omega*R
                Sp2=np.trace(A)
                delta=np.absolute(Sp2-Sp1)
                k=i
                if (delta<=self.eps):
                    break
            
            x=np.linspace(0.,self.n*self.s,self.n)
            y=np.linspace(0.,self.m*self.s,self.m)
            plotter.plot(x,y,A)
            return (A,k)

class Field:
    def __init__(self,M,N,Potential,Step,plotter):
        self.n=N
        self.m=M
        self.U=Potential
        self.s=Step
        self.plotter=plotter
    def Ex(self):
        ex=(self.U[:,1:]-self.U[:,:-1])/(2*self.s)
        x=np.linspace(0.,self.n*self.s,self.n-1)
        y=np.linspace(0.,self.m*self.s,self.m)
        plotter.plot(x,y,ex)
        return ex
    def Ey(self):
        ey=(self.U[1:,:]-self.U[:-1,:])/(2*self.s)
        x=np.linspace(0.,self.n*self.s,self.n)
        y=np.linspace(0.,self.m*self.s,self.m-1)
        plotter.plot(x,y,ey)
        return ey

TB=np.zeros(b)
BB=np.zeros(b)
LB=np.zeros(a)[:]=100.
RB=np.zeros(a)
pp=np.zeros((a,b))
w=1.0
            
"""plotter = Plotter('LaplasJ', 'x', 'y')
solveL=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z1=solveL.Jakobi()
plotter.show()"""

"""plotter = Plotter('LaplasGZ', 'x', 'y')            
solveGZ=SolvePoisson(a,b,1,pp,5000,LB,RB,TB,BB,0.01,plotter)
Z2=solveGZ.GaussZeudel()
plotter.show()"""
            
"""plotter = Plotter('LaplasR', 'x', 'y')
solveR=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z3,k3=solveR.Relax(w)
plotter.show()"""

plotter = AdvancedPlotter('Analitic', 'x', 'y')
solveL=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z0=solveL.Analitic(100)
plotter.show()
            
plotter = AdvancedPlotter('LaplasJ', 'x', 'y')
solveL=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z1=solveL.Jakobi()
plotter.show()

"""plotter = AdvancedPlotter('LaplasR', 'x', 'y')
solveL=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z3,k3=solveR.Relax(w)
plotter.show()"""

TB=np.zeros(b)
BB=np.zeros(b)
LB=np.zeros(a)[:]=100.
RB=np.zeros(a)[:]=-100.
pp=np.zeros((a,b))

plotter = AdvancedPlotter('Capacity', 'x', 'y')
solveL=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z5=solveL.Jakobi()
plotter.show()

TB=np.zeros(b)
BB=np.zeros(b)
LB=np.zeros(a)
RB=np.zeros(a)
pp=np.zeros((a,b))
pp[15:-15,20]=1.
pp[15:-15,-20]=-1.

plotter = AdvancedPlotter('Poisson', 'x', 'y')
solveL=SolvePoisson(a,b,1,pp,10000,LB,RB,TB,BB,0.01,plotter)
Z4=solveL.Jakobi()
plotter.show()

plotter = AdvancedPlotter('Ex', 'x', 'y')
field=Field(a,b,Z4,1,plotter)
ex=field.Ex()
plotter.show()

plotter = AdvancedPlotter('Ey', 'x', 'y')
field=Field(a,b,Z4,1,plotter)
ey=field.Ey()
plotter.show()