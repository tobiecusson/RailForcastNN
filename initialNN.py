# coding: utf-8

# In[4]:


# IMPORT ALL REQUIRED PACKAGES

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
import scipy
import sklearn

# In[5]:


# Read the "X" input data and "y" training data 'result' labels into the program
# ...and reshape the X variables into a vector and normalise them to values between 0-1

#X_excel = pd.read_excel('/Users/tobiecusson/desktop/test.xlsx',sheetname='X_train_partial',header=0,converters={'x1':float,'x2':float, 'x3':float})
X_excel = pd.read_excel('/Users/tobiecusson/desktop/X_train.xlsx',sheetname='X_train',header=0)
X_orig = np.array(X_excel)

#y_train = pd.read_excel('/Users/tobiecusson/desktop/171025_RUDD_X_and_y_data_HardPastedGroups.xlsx',sheetname='y_train_partial',header=0,converters={'y':float })
y_train = pd.read_excel('/Users/tobiecusson/desktop/y_train.xlsx',sheetname='y_train',header=0)
                        
y_train = np.array(y_train)
y_train = y_train.T

#print ("This is X_orig:" + str(X_orig))
#print ("This is the shape of X_orig : " + str(X_orig.shape))

assert (type(X_orig) == np.ndarray)
assert (type(y_train) == np.ndarray)

# Normalise X to mean 0 and variance 1
mu = np.mean((X_orig), axis=0, keepdims=True)
var = np.var((X_orig), axis=0, keepdims=True)
#print("This is mu:" + str(mu))
#print("This is var:" + str(var))

epsilon = 1e-08
X_train = (X_orig-mu)/(var+epsilon)
X_train = X_train.T

print ("This is the shape of X_train: " + str(X_train.shape))
print ("This is the shape of y_train: " + str(y_train.shape))
#print ("This is X_train: " + str(X_train))
#print ("This is y_train: " + str(y_train))


# In[6]:


layer_dims = [(X_train.shape[0]),2, 1]
#print ("There are", len(layer_dims)-1, "layers total (excluding the input layer), and", len(layer_dims)-1, "Weight matrices (W) & bias vectors (b) in the NN")
#print ("There are", layer_dims[0], "'input' nodes,", layer_dims[1], "nodes in the first hidden layer...")
cache = {}
caches= {} 
parameters = {}


# In[7]:


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    
    #print("initialise_parameters_deep function")     
    
    parameters = {}                # define the parameters dictionary with the W's and b's
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        #print("W" + str(l) +" shape is = " + str(parameters['W' + str(l)].shape)) #for testing
        #print("b" + str(l) +" shape is = " + str(parameters['b' + str(l)].shape)) #for testing

    return parameters

#initialize_parameters_deep(layer_dims) #for testing


# In[8]:


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    #print("linear_forward_function")    
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

#parameters = initialize_parameters_deep(layer_dims)
#linear_forward(X_train, parameters['W1'], parameters['b1'])


# In[9]:


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    #print("relu function")
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


# In[10]:


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    #print("linear_activation_forward function")
    
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation == "none":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = no_activation(Z)
    
    else:
        print("activation not correctly passed to this function")
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def test_linear_activation_forward():
    parameters = initialize_parameters_deep(layer_dims)
    #print("Shape of W1:" +str(parameters['W1'].shape))
    #print("Shape of W2:" +str(parameters['W2'].shape))
    A, cache = linear_activation_forward(X_train, parameters['W1'], parameters['b1'], "relu")
    print ("This is A1:" + str(A))
#test_linear_activation_forward()


# In[11]:


def L_model_forward(X, parameters) :
    """
    Implement forward propagation for the [LINEAR->"act_all_but_last"]*(L-1)->LINEAR->"act_last" computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    activation_last -- the activation to be used in final layer, stored as a text string: "sigmoid", "relu", "none", etc
    
    
    Returns:
    AL -- last post-activation value (NB: In the code below, the variable 'AL' denotes y_hat)
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_activation_forward() (there is one, indexed L-1)
    """
    #print("L model forward function")

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> "activation_all_but_last"]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    # Implement final layer activation
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],parameters['b' + str(L)], "relu")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def test_L_model_forward():
    parameters = initialize_parameters_deep(layer_dims)
    AL, caches = L_model_forward(X_train, parameters)
    print ("This is AL:" + str(AL))
#test_L_model_forward()


# In[12]:


def MSE(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the mse loss function
    """  
    m = y.shape[1]
    #print("m:" + str(m))
    loss = np.nansum(np.dot(y-yhat.T, y-yhat.T))
    #print("loss:" + str(loss))
    cost = loss/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def test_MSE():
    parameters = initialize_parameters_deep(layer_dims)
    AL, caches = L_model_forward(X_train, parameters)
    print("MSE = " + str(MSE(AL,y_train)))
#test_MSE()


# In[13]:


#We have now reached the end of the forward propagation 

#The predicted values for Y_hat = AL have been determined


# In[14]:


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    #print("relu_backward function")
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def test_relu_backward():
    parameters = initialize_parameters_deep(layer_dims)
    AL, caches = L_model_forward(X_train, parameters)
    cache = caches[len(layer_dims)-2]
    #this is the cache from the final layer L, since there are len(layer_dims)-1 layers and the first is indexed by 0
    #For the logistic cost function, dAL (which we will call dA) is defined as below
    #print (cache)
    activation_cache = np.reshape(cache[1], (1,330))
    dAL = 2/y_train.shape[1] * (AL-y_train) # derivative of cost with respect to AL
    dZ = relu_backward(dAL, activation_cache)
    print("This is dZ:" + str(dZ))
    
#test_relu_backward() #Switch off this line when not testing


# In[15]:


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    #print("linear_backward function")

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# In[16]:


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid", "relu", "none", etc
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    #print("linear_activation_backwards function")

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)    
    elif activation == "none":
        dZ = no_activation_backward(dA, activation_cache)
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

#add test to linear_activation_backward function here


# In[17]:


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->activation_all_but_last] * (L-1) -> LINEAR -> activation_last
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "activation_all_but_last" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "activation_last" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    #print("L_model_backwards function")
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = 2/m * (AL-Y) # derivative of cost with respect to AL
         
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "relu")

    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#add test to L_model_backward function here


# In[18]:


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward_sigmoid
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    #print("update_parameters function")
    
    L = len(parameters) // 2 # number of layers in the neural network
  
    for l in range(L):
        #print("W" + str(l+1) +" before updating is:" + str(parameters["W" + str(l+1)]))
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db"+ str(l+1)]
        #print("W" + str(l+1) +" after updating is:" + str(parameters["W" + str(l+1)]))

    return parameters

#add test to update_parameters function here


# In[ ]:


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations , print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU].
    
    Arguments:
    X -- data, numpy array of shape (number of examples, number of variables)
    Y -- true "label" vector, of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    #print("L_layer_model function")

    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)    
        
        # Compute cost.
        cost = MSE(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 5 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters




# In[ ]:


resultA = L_layer_model(X_train, y_train, layer_dims, learning_rate = 0.03, num_iterations = 10000, print_cost=True)


# In[ ]:


resultB = L_layer_model(X_train, y_train, layer_dims, learning_rate = 0.02, num_iterations = 10000, print_cost=True)


# In[ ]:


resultC = L_layer_model(X_train, y_train, layer_dims, learning_rate = 0.01, num_iterations = 10000, print_cost=True)


# In[ ]:


resultD = L_layer_model(X_train, y_train, layer_dims, learning_rate = 0.005, num_iterations = 10000, print_cost=True)

