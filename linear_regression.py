from autograd import numpy as np
from matplotlib import pyplot as plt
from autograd import grad 
from gradient_descent import gradient_descent as gd
import argparse

parser = argparse.ArgumentParser(description='Implemention of linear regression')
parser.add_argument('-i','--iteration', type = int, default = 500)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--csvname',default = 'kleibers_law_data.csv')
parser.add_argument('--draw',type = int, default = 0)

def main():
    global args , x , y ,logx
    args = parser.parse_args()
    datapath = 'datasets/'
    csvname = datapath + args.csvname
    data = np.loadtxt(csvname,delimiter = ',')
    #load in data
    x = data[:-1,:]
    y = data[-1:,:]
    logx = np.log(x)
    y = np.log(y)
    x = np.concatenate((np.ones((1,1498),dtype=float),logx))

    weights = np.random.rand(x.shape[0],1) #initial weights
    weight_his, cost_his  = gd(square_cost, args.lr, args.iteration, weights)
    if args.draw:
        draw_pic(weight_his,cost_his)

def square_cost(w):
    pre = np.dot(x.T,w)

    return np.mean(np.square(pre-y.T))

def draw_pic(weight_his,cost_his):
    final_w = weight_his[-1]
    _y = np.dot(x.T,final_w)


    plt.figure(figsize = (14,7))
    plt.scatter(logx[0],y[0])
    plt.plot(logx[0],_y,'r',lw = 5)    

    plt.xlabel('log(xp)',fontsize = 30)
    plt.ylabel('log(yp)',fontsize = 30)

    plt.text(5, 10, 'log(yp)={:.3f}*log(xp)+ {:.3f}'.format(final_w[0][0],final_w[1][0]) , fontdict={'size': 20, 'color':  'red'})


    plt.show()


if __name__ == '__main__':
    main()