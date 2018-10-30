from autograd import grad
from perceptron import count_misclassification

def gradient_descent(g,alpha,max_its,w):
    '''
    input: cost_function, learning_rate, max_iterations, initial_weight
    output: list[weights], list[cost]
    '''
    gradient = grad(g)
    weight_history = [w] # weight history container
    cost_history = [] # cost function history container
    d = 0
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = gradient(w)

        w = w - alpha * grad_eval        
        weight_history.append(w)
        cost_history.append(g(w))

    return weight_history,cost_history

def gradient_descent_2(g,alpha,max_its,w):
    '''
    count misclassification number each iteration
    input: cost_function, learning_rate, max_iterations, initial_weight
    output: list[weights], list[cost], list[wrong_numbers]
    '''
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost function history container
    wrong_num = [count_misclassification(w)]
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = gradient(w)
        # take gradient descent step
        w = w - alpha*grad_eval
        mis_k = count_misclassification(w)
        wrong_num.append(mis_k)
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history,wrong_num   