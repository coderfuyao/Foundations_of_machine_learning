from autograd import grad

def gradient_descent(g,alpha,max_its,w):
    '''
    input: cost_function, learning_rate, max_iterations, initial_weight
    output: list[weights], list[cost]
    '''
    gradient = grad(g)
    weight_history = [w] # weight history container
    cost_history = [] # cost function history container
    wrong_his = [count_wrong(w)]
    d = 0
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = gradient(w)

        w = w - alpha * grad_eval        
        weight_history.append(w)
        cost_history.append(g(w))

    return weight_history,cost_history