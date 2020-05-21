
# coding: utf-8

# In[ ]:

# This is a little tool box for an ODE analysis
# autor: MahAlg
# date: 21. Mai 2020

import numpy as np
import matplotlib.pyplot as plt

class ODE:
    
    def __init__(self,RHS,x0,xend,y0):
        self.__RHS = RHS
        self.__y0   = y0
        if x0 < xend:
            self.__x0   = x0
            self.__xend = xend
        elif x0 > xend:
            print("Warning: x0 is greater than xend! It was implicit switched.")
            self.__x0   = xend
            self.__xend = x0
        else:
            print("Warning: x0 equal to xend!")
            # TODO: Fehlermeldung!
        self.solution = []
          
    def directionField(self,xmin,xmax,ymin,ymax,nx=20,ny=20):
        if xmin>xmax:
            print("Warning: In method 'directionField' xmin greater than xmax!\n It has been switched.")
            tmp = xmin
            xmin = xmax
            xmax = xmin
        if ymin>ymax:
            print("Warning: In method 'directionField' ymin greater than xyax!\n It has been switched.")
            tmp = ymin
            ymin = ymax
            ymax = ymin    
        
        X,Y = np.meshgrid(np.linspace(xmin,xmax,nx),
                          np.linspace(ymin,ymax,ny))
        X_dir = 0*X+1
        Y_dir = self.__RHS(X,Y)
        L = np.sqrt(X_dir*X_dir+Y_dir*Y_dir)
        plt.quiver(X, Y, X_dir/L, Y_dir/L)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def explicitEuler(self,n):
        xgrid = np.linspace(self.__x0,self.__xend,n)
        ygrid = np.linspace(0,0,n)
        ygrid[0] = self.__y0
        h = (self.__xend-self.__x0)/(n-1)
        for k in range(n-1):
            print("Progress: %6.2f %%" % (100*(k+1)/(n-1)),end='\r')
            ygrid[k+1] = ygrid[k]+h*self.__RHS(xgrid[k],ygrid[k])
        self.solution.append([xgrid,ygrid,'expl.E, n='+str(n)])
            
    def RK4(self,n):
        xgrid = np.linspace(self.__x0,self.__xend,n)
        ygrid = np.linspace(0,0,n)
        ygrid[0] = self.__y0
        h = (self.__xend-self.__x0)/(n-1)
        for k in range(n-1):
            print("Progress: %6.2f %%" % (100*(k+1)/(n-1)),end='\r')
            k1 = self.__RHS(xgrid[k],ygrid[k])
            k2 = self.__RHS(xgrid[k]+0.5*h,ygrid[k]+0.5*h*k1)
            k3 = self.__RHS(xgrid[k]+0.5*h,ygrid[k]+0.5*h*k2)
            k4 = self.__RHS(xgrid[k]+h,ygrid[k]+h*k3)
            ygrid[k+1] = ygrid[k]+h*(k1/6+k2/3+k3/3+k4/6)
        self.solution.append([xgrid,ygrid,'RK4, n='+str(n)])
        
    def solve(self,n,method='explE'):
        if method=='explE':
            self.explicitEuler(n)
        elif method=='RK4':
            self.RK4(n)
        else:
            print("Method not defined!\nYou can choose between:")
            print(" - explicit Euler Method (type: 'explE')")
            print(" - explicit Runge-Kutta Method (type: 'RK4')")
            
    def show(self,directions='off'):
        colors = ['red','blue','green','gold','darkorange','cyan']
        nsol = len(self.solution)
        if nsol==0:
            print("There are no solutions in memory.")
            return
        
        plt.plot(self.solution[0][0],self.solution[0][1],color=colors[0],label=self.solution[0][2])
        ymin = min(self.solution[0][1])
        ymax = max(self.solution[0][1])
        for i in range(1,nsol):
            plt.plot(self.solution[i][0],self.solution[i][1],color=colors[i],label=self.solution[i][2])
            ymin = min(min(self.solution[i][1]),ymin)
            ymax = max(max(self.solution[i][1]),ymax)
        
        plt.legend(loc='best')
        
        if directions != 'off':
            self.directionField(self.__x0,self.__xend,ymin,ymax)
        else:
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
    
    def clear(self):
        self.solution = []

