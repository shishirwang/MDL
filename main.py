from utils import *
from mnist import load

with open('pickle/model_trained/mdl_1l2p.pickle','rb') as file:
    l0=pickle.load(file)
with open('pickle/model_trained/mdl_1l1p.pickle','rb') as file:
    l1=pickle.load(file)
with open('pickle/model_trained/bp.pickle','rb') as file:
    l2=pickle.load(file)

n=5

#error versus epoch
x=np.arange(1,151)

def error(model):
    return np.log(1-np.asarray(model.accuracy))

y0=np.asarray(list(map(error,l0[0]))).T
y0_mean=np.mean(y0,axis=1)
y0_std=np.std(y0,axis=1)

y1=np.asarray(list(map(error,l1[0]))).T
y1_mean=np.mean(y1,axis=1)
y1_std=np.std(y1,axis=1)

y2=np.asarray(list(map(error,l2[0]))).T
y2_mean=np.mean(y2,axis=1)
y2_std=np.std(y2,axis=1)

with open('pickle/figure_data/figure0.pickle','wb') as file:
    pickle.dump((x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std),file)

#dispersion
x0=np.arange(2,6)
x1=np.arange(1,5)
x2=np.arange(1,6)

def dispersion(xi):
    d=[]
    for l in range(len(xi)):
        center=np.mean(xi[l],axis=1)[:,np.newaxis]
        d.append(np.mean(np.sqrt(np.sum((xi[l]-center)**2,axis=0))))
    return d

xi1=[l0[1][i].xi1[1:5] for i in range(n)]
y0=np.transpose(list(map(dispersion,xi1)))
y0_mean=np.mean(y0,axis=1)
y0_std=np.std(y0,axis=1)

xi2=[l0[1][i].xi2 for i in range(n)]
y1=np.transpose(list(map(dispersion,xi2)))
y1_mean=np.mean(y1,axis=1)
y1_std=np.std(y1,axis=1)

xi=[l1[1][i].xi for i in range(n)]
y2=np.transpose(list(map(dispersion,xi)))
y2_mean=np.mean(y2,axis=1)
y2_std=np.std(y2,axis=1)

with open('pickle/figure_data/figure1.pickle','wb') as file:
    pickle.dump((x0,x1,x2,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std),file)

#principal angle
x=np.arange(1,6)

mnist=np.array(load.load_mnist(one_hot=True),dtype="object")
train_data=mnist[0][0]
train_label=mnist[0][1]
index=[[] for i in range(10)]
for i in range(10):
    for j in range(train_label.shape[0]):
        if train_label[j][i]==1:
            index[i].append(j)
data=[train_data[index[i][:5000]].T for i in range(10)]

def pca(x,percent):
    cov=np.cov(x)
    eigenvalue,eigenvector=np.linalg.eigh(cov)
    eigenvalue_descending,eigenvector_descending=eigenvalue[::-1],eigenvector.T[::-1].T
    for i in range(len(eigenvalue)):
        if np.sum(eigenvalue_descending[slice(i+1)])>percent*np.sum(eigenvalue):
            return eigenvector_descending[:,slice(i+1)]
        
def principal_angle(model):
    a=[]
    for i,j in list(combinations(np.arange(10),2)):
        model.evaluate(data[i])
        h0=deepcopy(model.h)
        model.evaluate(data[j])
        h1=deepcopy(model.h)
        l=[]
        for k in range(1,6):
            q0=pca(h0[k],0.8)
            q1=pca(h1[k],0.8)
            s=np.linalg.svd(q0.T@q1)[1]
            l.append(s[0])
        a.append(l)
    return np.mean(a,axis=0)

y0=np.transpose(list(map(principal_angle,l0[-1])))
y0_mean=np.mean(y0,axis=1)
y0_std=np.std(y0,axis=1)

y1=np.transpose(list(map(principal_angle,l1[-1])))
y1_mean=np.mean(y1,axis=1)
y1_std=np.std(y1,axis=1)

y2=np.transpose(list(map(principal_angle,l2[-1])))
y2_mean=np.mean(y2,axis=1)
y2_std=np.std(y2,axis=1)

with open('pickle/figure_data/figure2.pickle','wb') as file:
    pickle.dump((x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std),file)

#remove modes
x=np.arange(61)

mnist=np.array(load.load_mnist(one_hot=True),dtype="object")
test_data = mnist[1][0].T
test_label = mnist[1][1].T
    
def remove(model,percent,l,i):
    sigma_removed=deepcopy(model.sigma)
    xi1=model.xi[1:4]
    xi2=model.xi[:3]
    
    l2_xi1=np.sqrt(np.sum(xi1[l]**2,axis=0))
    l2_xi2=np.sqrt(np.sum(xi2[l]**2,axis=0))
    abs_sigma=np.abs(np.diag(model.sigma[l]))
    gamma=np.sum(abs_sigma)/np.sum(l2_xi1+l2_xi2)
    tau=gamma*(l2_xi1+l2_xi2)+abs_sigma
    
    if i==0:
        index_tau=np.argsort(tau)
        temp=np.diag(model.sigma[l]).copy()
        temp[index_tau[:int(percent*0.01*70)]]=0
        sigma_removed[l]=np.diag(temp)
    if i==1:
        index_random=np.random.permutation(model.sigma[l].shape[0])
        temp=np.diag(model.sigma[l]).copy()
        temp[index_random[:int(percent*0.01*70)]]=0
        sigma_removed[l]=np.diag(temp)
    return sigma_removed

def accuracy(model):
    a=[]
    for l,i in [[0,0],[0,1],[1,0],[1,1]]:
        b=[]
        for percent in range(61):
            sigma_removed=remove(model,percent,l,i)
            model.w=[model.xi[i]@sigma_removed[i]@model.xi[i+1].T for i in range(3)]
            b.append(model.test(test_data,test_label))
        a.append(b)
    return np.transpose(a)

y0,y1,y2,y3=np.transpose(list(map(accuracy,l1[0])),(2,1,0))

y0_mean=np.mean(y0,axis=1)
y0_std=np.std(y0,axis=1)
y1_mean=np.mean(y1,axis=1)
y1_std=np.std(y1,axis=1)
y2_mean=np.mean(y2,axis=1)
y2_std=np.std(y2,axis=1)
y3_mean=np.mean(y3,axis=1)
y3_std=np.std(y3,axis=1)

with open('pickle/figure_data/figure3.pickle','wb') as file:
    pickle.dump((x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std,y3_mean,y3_std),file)

#measure
x=np.arange(1,71)

def measure(model,l):
    xi1=model.xi[1:4]
    xi2=model.xi[:3]
    
    l2_xi1=np.sqrt(np.sum(xi1[l]**2,axis=0))
    l2_xi2=np.sqrt(np.sum(xi2[l]**2,axis=0))
    abs_sigma=np.abs(np.diag(model.sigma[l]))
    gamma=np.sum(abs_sigma)/np.sum(l2_xi1+l2_xi2)
    tau=gamma*(l2_xi1+l2_xi2)+abs_sigma
    
    return np.sort(tau)[::-1],np.sort(abs_sigma)[::-1],np.sort(gamma*l2_xi1)[::-1],\
           np.sort(gamma*l2_xi2)[::-1]

l=1
y0,y1,y2,y3=np.transpose(list(map(measure,l1[0],[l]*n)),(1,2,0))

y0_mean=np.mean(y0,axis=1)
y0_std=np.std(y0,axis=1)
y1_mean=np.mean(y1,axis=1)
y1_std=np.std(y1,axis=1)
y2_mean=np.mean(y2,axis=1)
y2_std=np.std(y2,axis=1)
y3_mean=np.mean(y3,axis=1)
y3_std=np.std(y3,axis=1)

with open('pickle/figure_data/figure4.pickle','wb') as file:
    pickle.dump((x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std,y3_mean,y3_std),file)

