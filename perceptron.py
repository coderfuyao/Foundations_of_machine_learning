from autograd import numpy as np
from matplotlib import pyplot as plt
from autograd import grad 
from gradient_descent import gradient_descent_2 as gd2
import argparse

parser = argparse.ArgumentParser(description='Implemention of gradient descent')
parser.add_argument('-i','--iteration', type = int, default = 100)
parser.add_argument('--lr1', type = float, default = 0.1)
parser.add_argument('--lr2', type = float, default = 0.01)
parser.add_argument('--csvname',default='3d_classification_data_v0.csv')
parser.add_argument('--draw',type = int, default = 0)

def main():
    global args, x, y
    args = parser.parse_args()
    datapath = 'datasets/'
    csvname = datapath + args.csvname
    data = np.loadtxt(csvname,delimiter = ',')

    x = data[:-1,:]
    y = data[-1:,:] 
    x = np.concatenate((np.ones((1,x.shape[1]),dtype=float),x))

    weights = np.random.rand(x.shape[0],1)    
    alpha = [0.1,0.01]
    weight_his1, cost_his1, wrong_num1 = gd2(cost_function, args.lr1, args.iteration, weights)
    weight_his2, cost_his2, wrong_num2 = gd2(cost_function, args.lr2, args.iteration, weights)
    update_weight1 = weight_his1[-1]
    update_weight2 = weight_his2[-1]
    res1 = np.sign(np.dot(x.T,update_weight1))
    res2 = np.sign(np.dot(x.T,update_weight2))
    acc1 = accuracy(res1,y)
    acc2 = accuracy(res2,y)
    print('the accuracy when lr=0.1 after 50 iterations is {:.3f}'.format(acc1))
    print('the accuracy when lr=0.01 after 50 iterations is {:.3f}'.format(acc2))

    if args.draw:
        plt.figure(21,figsize=(8,6))
        #draw pic1
        plt.subplot(211)
        plt.tight_layout(5)
        plt.title('cost function history plot')
        plt.plot([i for i in range(args.iteration+1)],cost_his1,'b')
        plt.plot([i for i in range(args.iteration+1)],cost_his2,'r')
        plt.legend(['$lr=0.1$','$lr=0.01$'])

        #draw pic2
        plt.subplot(212)
        plt.title('misclassification history plot')
        plt.scatter([i for i in range(args.iteration+1)],wrong_num1)
        plt.scatter([i for i in range(args.iteration+1)],wrong_num2)
        plt.legend(['$lr=0.1$','$lr=0.01$'])
        plt.show()


def cost_function(w):
    tmp = np.maximum(0,(-y.T*np.dot(x.T,w)))
    return np.mean(tmp)

def softmax(w):
    cost = np.sum(np.log(1 + np.exp(-y.T*np.dot(x.T,w))))
    return cost/float(np.size(y))


def count_misclassification(weights):
    '''
    count misclassification number
    '''
    res = np.sign(np.dot(x.T,weights))
    return np.where(res.T[0]!=y[0])[0].shape[0]

def accuracy(res,y):
    '''
    count accuracy
    '''
    acc = np.where(res.T[0]==y[0])[0].shape[0] / y[0].shape[0]
    return acc    

if __name__ == '__main__':
    main()