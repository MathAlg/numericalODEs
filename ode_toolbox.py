
# coding: utf-8

# In[ ]:

# This is a little tool box for an ODE analysis
# autor: MahAlg
# date: 21. Mai 2020

import numpy as np
import matplotlib.pyplot as plt

class ODE:
    eps     = .1e-5 # error tolerance for newton method (can be changed from outside)
    itermax = 100   # maximum iteration number for newton method (can be changed from outside)
    def __init__(self,x0,xend,y0,RHS,DRHS=None):
        self.__RHS  = RHS
        self.__DRHS = DRHS
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
        self.__log = {}
          
    def directionField(self,xmin,xmax,ymin,ymax,nx=25,ny=25):
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
    
    ###############################################################################
    #                               Newton Method
    ###############################################################################
    
    def NewtonMethod(self,func,Dfunc,y_k0,m,method_key):
            if abs(func(y_k0))<self.eps:
                return y_k0
            else:
                y_k = y_k0
                for i in range(self.itermax):
                    y_k = y_k - func(y_k)/Dfunc(y_k)
                    if abs(func(y_k))<self.eps:
                        return y_k
                
                self.__log[method_key].append("Warning: Newton method has reached iteration limit "
                                              +str(self.itermax)+" for point number "+str(m)+".\n"+
                                              "Initial guess: "+str(y_k0)+", last calculated: "+str(y_k)+".\n"+
                                              "Last calculated y was given back to implicit method.\n")
                return y_k
            
    ###############################################################################
    #                           Procedures for solving
    ###############################################################################
    
    def explicitEuler(self,n):
        xgrid = np.linspace(self.__x0,self.__xend,n)
        ygrid = np.linspace(0,0,n)
        ygrid[0] = self.__y0
        h = (self.__xend-self.__x0)/(n-1)
        for k in range(n-1):
            print("Progress: %6.2f %%" % (100*(k+1)/(n-1)),end='\r')
            ygrid[k+1] = ygrid[k]+h*self.__RHS(xgrid[k],ygrid[k])
        self.solution.append([xgrid,ygrid,'expl.E, n='+str(n)])
        
    def implicitEuler(self,n):
        method_key = "implE, n="+str(n)
        self.__log.update({method_key:[]})
        if not self.__DRHS:
            print("There is no derivative (in y) of right hand side 'f'!\nImplicit Method not possible to proceed.")
            return
        xgrid = np.linspace(self.__x0,self.__xend,n)
        ygrid = np.linspace(0,0,n)
        ygrid[0] = self.__y0
        h = (self.__xend-self.__x0)/(n-1)
        for k in range(n-1):
            print("Progress: %6.2f %%" % (100*(k+1)/(n-1)),end='\r')
            F  = lambda y: ygrid[k]-y+h*self.__RHS(xgrid[k+1],y)
            DF = lambda y: h*self.__DRHS(xgrid[k+1],y)-1
            ygrid[k+1] = self.NewtonMethod(F,DF,ygrid[k],k,method_key)
        self.solution.append([xgrid,ygrid,'impl.E, n='+str(n)])
        
    def MidPointRule(self,n):
        method_key = "MPR, n="+str(n)
        self.__log.update({method_key:[]})
        if not self.__DRHS:
            print("There is no derivative (in y) of right hand side 'f'!\nImplicit Method not possible to proceed.")
            return
        xgrid = np.linspace(self.__x0,self.__xend,n)
        ygrid = np.linspace(0,0,n)
        ygrid[0] = self.__y0
        h = (self.__xend-self.__x0)/(n-1)
        for k in range(n-1):
            print("Progress: %6.2f %%" % (100*(k+1)/(n-1)),end='\r')
            F  = lambda y: ygrid[k]-y+h*self.__RHS(xgrid[k+1]+0.5*h,0.5*(y+ygrid[k]))
            DF = lambda y: 0.5*h*self.__DRHS(xgrid[k+1]+0.5*h,0.5*(y+ygrid[k]))-1
            ygrid[k+1] = self.NewtonMethod(F,DF,ygrid[k],k,method_key)
        self.solution.append([xgrid,ygrid,'MPR, n='+str(n)])
        
    def implicitTrapez(self,n):
        method_key = "implT, n="+str(n)
        self.__log.update({method_key:[]})
        if not self.__DRHS:
            print("There is no derivative (in y) of right hand side 'f'!\nImplicit Method not possible to proceed.")
            return
        xgrid = np.linspace(self.__x0,self.__xend,n)
        ygrid = np.linspace(0,0,n)
        ygrid[0] = self.__y0
        h = (self.__xend-self.__x0)/(n-1)
        for k in range(n-1):
            print("Progress: %6.2f %%" % (100*(k+1)/(n-1)),end='\r')
            F  = lambda y: ygrid[k]+0.5*h*self.__RHS(xgrid[k],ygrid[k])-y+0.5*h*self.__RHS(xgrid[k+1],y)
            DF = lambda y: 0.5*h*self.__DRHS(xgrid[k+1],y)-1
            ygrid[k+1] = self.NewtonMethod(F,DF,ygrid[k],k,method_key)
        self.solution.append([xgrid,ygrid,'impl.Trapez, n='+str(n)])
    
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
        elif method=='implE':
            self.implicitEuler(n)
        elif method=='MPR':
            self.MidPointRule(n)
        elif method=='implT':
            self.implicitTrapez(n)
        else:
            print("Method not defined!\nYou can choose between:")
            print(" - explicit Euler Method (enter: 'explE')")
            print(" - explicit Runge-Kutta Method (enter: 'RK4')")
            print(" - implicit Euler Method (enter: 'implE')")
            print(" - mid point rule (enter: 'MPR')")
            print(" - implicit Trapez Method (enter: 'implT')")
            
    ###############################################################################
    #                           Showing, administration
    ###############################################################################
            
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
            
    def log(self):
        if len(self.__log)==0:
            print("There are no logs!")
        elif len(self.__log)==1:
            for key in self.__log:
                print(key)
                if len(self.__log[key])==0:
                    print("No entrys in log!")
                else:
                    for entry in self.__log[key]:
                        print(entry)
        else:
            key_list = []
            print("Choose desired log:")
            i=0
            for key in self.__log:
                i+=1
                key_list.append(key)
                print(str(i)+" "+key)
            user_input = input("")
            while not int(user_input) in [i for i in range(1,i+1)]:
                print("Incorrect input. Please choose your desired log by insert the corresponding number!\n")
                user_input = input("")
            print(key_list[int(user_input)-1])
            if len(self.__log[key_list[int(user_input)-1]])==0:
                print("No entrys in log!")
            else:
                for entry in self.__log[key_list[int(user_input)-1]]:
                    print(entry)
                
    def clear(self):
        self.solution = []
        self.__log = {}
        
    def reset(self):
        self.solution = []
        self.__log = {}
        self.eps     = .1e-5
        self.itermax = 100

