from utils import *
from model.mdl_1l2p import *
from model.mdl_1l1p import *
from model.bp import *

mnist=np.array(load.load_mnist(one_hot=True),dtype="object")
train_data = mnist[0][0].T
train_label = mnist[0][1].T
test_data = mnist[1][0].T
test_label = mnist[1][1].T

repetition=5
mini_batch_size=100

#1l2p
learning_rate=0.004
file_path='pickle/model_trained/mdl_1l2p.pickle'

dimension=[[784,100,100,10]]+[[784,100,100,100,10]]+[[784,100,100,100,100,100,10]]
mode=[30]*3
epoch=[150]+[100]*2
test_every_epoch=[True]+[False]*2

l=[]
with open(file_path,'wb') as file:
    for i in range(len(dimension)):
        print(i)
        model=mdl_1l2p(dimension[i],relu,softmax,d_relu)
        model_trained=model.train(mode[i],learning_rate,mini_batch_size,epoch[i],repetition,\
                                  train_data,train_label,test_data,test_label,test_every_epoch[i])
        l.append(model_trained)
    pickle.dump(l,file)

#1l1p
file_path='pickle/model_trained/mdl_1l1p.pickle'

dimension=[[784,100,100,10]]+[[784,100,100,100,10]]+[[784,100,100,100,100,100,10]]
mode=[70]+[30]*2
learning_rate=[0.006]+[0.003]*2
epoch=[150]+[100]*2
test_every_epoch=[True]+[False]*2

l=[]
with open(file_path,'wb') as file:
    for i in range(len(dimension)):
        print(i)
        model=mdl_1l1p(dimension[i],relu,softmax,d_relu)
        model_trained=model.train(mode[i],learning_rate[i],mini_batch_size,epoch[i],repetition,\
                                  train_data,train_label,test_data,test_label,test_every_epoch[i])
        l.append(model_trained)
    pickle.dump(l,file)

#bp
learning_rate=0.001
file_path='pickle/model_trained/bp.pickle'

dimension=[[784,100,100,10],[784,100,100,100,100,100,10]]
epoch=[150,100]
test_every_epoch=[True,False]

l=[]
with open(file_path,'wb') as file:
    for i in range(len(dimension)):
        print(i)
        model=bp(dimension[i],relu,softmax,d_relu)
        model_trained=model.train(learning_rate,mini_batch_size,epoch[i],repetition,\
                                  train_data,train_label,test_data,test_label,test_every_epoch[i])
        l.append(model_trained)
    pickle.dump(l,file)
