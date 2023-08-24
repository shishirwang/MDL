from utils import *

#figure0
with open('pickle/figure_data/figure0.pickle','rb') as file:
    l=pickle.load(file)
x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std=l

plt.figure('figure0')
plt.plot(x,y0_mean,label='1L2P (p=30, lr=0.004)')
plt.fill_between(x,y0_mean-y0_std,y0_mean+y0_std,alpha=0.5)
plt.plot(x,y1_mean,label='1L1P (p=70, lr=0.006)')
plt.fill_between(x,y1_mean-y1_std,y1_mean+y1_std,alpha=0.5)
plt.plot(x,y2_mean,label='BP (lr=0.001)')
plt.fill_between(x,y2_mean-y2_std,y2_mean+y2_std,alpha=0.5)
plt.xlabel('epoch')
plt.ylabel('log(error rate)')
plt.legend()
plt.savefig('figure/figure0.pdf')

#figure1
with open('pickle/figure_data/figure1.pickle','rb') as file:
    l=pickle.load(file)
x0,x1,x2,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std=l

plt.figure('figure1')
plt.errorbar(x0,y0_mean,y0_std,fmt='o--',capsize=4,label=r'1L2P $\xi$ (lr=0.004)')
plt.errorbar(x1,y1_mean,y1_std,fmt='o--',capsize=4,label=r'1L2P $\hat{\xi}$')
plt.errorbar(x2,y2_mean,y2_std,fmt='o--',capsize=4,label='1L1P (lr=0.003)')
plt.xlabel('layer index')
plt.ylabel('dispersion')
plt.xticks(x2)
plt.legend()
plt.savefig('figure/figure1.pdf')

#figure2
with open('pickle/figure_data/figure2.pickle','rb') as file:
    l=pickle.load(file)
x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std=l

plt.figure('figure2')
plt.errorbar(x,y0_mean,y0_std,fmt='o--',capsize=4,label='1L2P (lr=0.004)')
plt.errorbar(x,y1_mean,y1_std,fmt='o--',capsize=4,label='1L1P (lr=0.003)')
plt.errorbar(x,y2_mean,y2_std,fmt='o--',capsize=4,label='BP (lr=0.001)')
plt.xlabel('hidden layer index')
plt.ylabel('cos(principal angle)')
plt.xticks(x)
plt.legend()
plt.savefig('figure/figure2.pdf')

#figure3
with open('pickle/figure_data/figure3.pickle','rb') as file:
    l=pickle.load(file)
x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std,y3_mean,y3_std=l

plt.figure('figure3')
plt.plot(x,y0_mean,label=r'layer 1-2, $\tau$')
plt.fill_between(x,y0_mean-y0_std,y0_mean+y0_std,alpha=0.5)
plt.plot(x,y1_mean,linestyle='--',label='layer 1-2, random',)
plt.fill_between(x,y1_mean-y1_std,y1_mean+y1_std,alpha=0.5)
plt.plot(x,y2_mean,label=r'layer 2-3, $\tau$')
plt.fill_between(x,y2_mean-y2_std,y2_mean+y2_std,alpha=0.5)
plt.plot(x,y3_mean,linestyle='--',label='layer 2-3, random')
plt.fill_between(x,y3_mean-y3_std,y3_mean+y3_std,alpha=0.5)
plt.xlabel('removed modes (%)')
plt.ylabel('test accuracy')
plt.legend()
plt.savefig('figure/figure3.pdf')

#figure4
with open('pickle/figure_data/figure4.pickle','rb') as file:
    l=pickle.load(file)
x,y0_mean,y0_std,y1_mean,y1_std,y2_mean,y2_std,y3_mean,y3_std=l

plt.figure('figure4')
plt.plot(x,y0_mean,label=r'$\tau$')
plt.fill_between(x,y0_mean-y0_std,y0_mean+y0_std,alpha=0.5)
plt.plot(x,y1_mean,label=r'$|\Sigma|$')
plt.fill_between(x,y1_mean-y1_std,y1_mean+y1_std,alpha=0.5)
plt.plot(x,y2_mean,label=r'$\gamma||\xi||_2$')
plt.fill_between(x,y2_mean-y2_std,y2_mean+y2_std,alpha=0.5)
plt.plot(x,y3_mean,label=r'$\gamma||\hat{\xi}||_2$')
plt.fill_between(x,y3_mean-y3_std,y3_mean+y3_std,alpha=0.5)
plt.xlabel('rank')
plt.ylabel('measure')
plt.legend()

sub=plt.gca().inset_axes([0.51,0.58,0.49,0.42])
sub.loglog(x,y0_mean)
sub.fill_between(x,np.squeeze(y0_mean-y0_std),np.squeeze(y0_mean+y0_std),alpha=0.5)
sub.set_xlabel('rank')
sub.set_ylabel(r'$\tau$')
sub.set_ylim(0.1,2)
sub.set_xlim(1,500)
sub.axvline(x=(30),linestyle='--',color='coral')

plt.savefig('figure/figure4.pdf')
