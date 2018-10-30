def multiclass_perceptron(w,x,y):
    '''
    input: weights, x, target
    output: cost(float)
    '''
    # pre-compute predictions on all points

    all_evals = model(x,w)
    all_evals_tmp = np.exp(all_evals)
    # compute maximum across data points
    a = np.log(np.sum(all_evals_tmp,axis = 0))    

    # compute cost in compact form using numpy broadcasting
    b = all_evals[y.astype(int).flatten(),np.arange(np.size(y))]
    cost = np.sum(a - b)
    
    # add regularizer
    cost = cost + lam*np.linalg.norm(w[1:,:],'fro')**2
    
    # return average
    return cost/float(np.size(y))