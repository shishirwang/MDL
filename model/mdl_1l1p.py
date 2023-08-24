import sys
sys.path.append('../utils.py')
from utils import *

class mdl_1l1p:
    def __init__(self,dimension,activation1,activation2,d_activation1):
        self.d=dimension
        self.f1=activation1
        self.f2=activation2
        self.d_f1=d_activation1
        
    def train(self,mode,learning_rate,mini_batch_size,epoch,repetition,train_data,train_label,\
              test_data,test_label,test_every_epoch=False):
        model_trained=[]
        for i in range(repetition):
            if test_every_epoch==True:
                self.accuracy=[]
            self.initialize(mode)
            self.adam_xi=adam(self.xi)
            self.adam_sigma=adam(self.sigma)
            for j in range(epoch):
                data=train_data.T
                label=train_label.T
                random_state=np.random.get_state()
                np.random.shuffle(data)
                np.random.set_state(random_state)
                np.random.shuffle(label)
                for k in range(int(train_data.shape[1]/mini_batch_size)):
                    mini_batch_data=data[k*mini_batch_size:(k+1)*mini_batch_size].T
                    mini_batch_label=label[k*mini_batch_size:(k+1)*mini_batch_size].T            
                    self.evaluate(mini_batch_data)
                    self.gradient(mini_batch_label)
                    self.update(learning_rate*(0.5**(int(j/30))))
                if test_every_epoch==True:
                    self.accuracy.append(self.test(test_data,test_label))
            if test_every_epoch==False:
                self.accuracy=self.test(test_data,test_label)
            print(time.asctime(time.localtime(time.time())),':',i,self.accuracy)                    
            model_trained.append(deepcopy(self))
        return model_trained
        
    def initialize(self,mode):
        self.xi=[(1/(self.d[l]*np.log(self.d[l])))**(1/6)\
                 *np.random.normal(0,1,(self.d[l],mode)) for l in range(len(self.d)-1)]
        self.xi.append((1/(self.d[-2]*np.log(self.d[-2])))**(1/6)\
                       *np.random.normal(0,1,(self.d[-1],mode)))
        self.sigma=[np.diag((1/(self.d[l]*np.log(self.d[l])))**(1/6)\
                    *np.random.normal(0,1,mode)) for l in range(len(self.d)-1)]
        self.w=[self.xi[l]@self.sigma[l]@self.xi[l+1].T for l in range(len(self.d)-1)]
        
    def evaluate(self,data):
        self.z=[0]
        
        self.h=[data]
        for l in range(len(self.d)-2):
            self.z.append(self.w[l].T@self.h[l])
            self.h.append(self.f1(self.z[l+1]))
        self.z.append(self.w[-1].T@self.h[-1])
        self.h.append(self.f2(self.z[-1]))   
        
    def gradient(self,label):
        k=[0]
        self.nabla_xi=[]
        self.nabla_sigma=[]
        
        k.append(self.h[-1]-label)
        for l in range(2,len(self.d)):
            k.insert(1,self.w[-l+1]@k[-l+1]*self.d_f1(self.z[-l]))
        self.nabla_xi.append(self.h[0]@k[1].T@self.xi[1]@self.sigma[0].T/label.shape[1])
        self.nabla_sigma.append(np.diag(\
            np.sum((self.xi[0].T@self.h[0])*(self.xi[1].T@k[1]),axis=1)/label.shape[1]))
        for l in range(1,len(self.d)-1):
            self.nabla_xi.append((k[l]@self.h[l-1].T@self.xi[l-1]@self.sigma[l-1].T+\
                                 self.h[l]@k[l+1].T@self.xi[l+1]@self.sigma[l].T)/label.shape[1])
            self.nabla_sigma.append(np.diag(\
                np.sum((self.xi[l].T@self.h[l])*(self.xi[l+1].T@k[l+1]),axis=1)/label.shape[1]))  
        self.nabla_xi.append(k[-1]@self.h[-2].T@self.xi[-2]@self.sigma[-1].T/label.shape[1])
    
    def update(self,learning_rate):
        self.xi=self.adam_xi.update(self.nabla_xi,learning_rate,1e-4)
        self.sigma=self.adam_sigma.update(self.nabla_sigma,learning_rate,1e-4)
        self.w=[self.xi[l]@self.sigma[l]@self.xi[l+1].T for l in range(len(self.d)-1)]

    def test(self,test_data,test_label):
        self.evaluate(test_data)
        y=np.argmax(self.h[-1],axis=0)
        t=np.argmax(test_label,axis=0)
        return np.sum(y==t)/test_data.shape[1]
