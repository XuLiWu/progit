# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:54:26 2017
#Part 2： Logistic Regression with a Neural Network mindset
@author: Liwu Xu
"""

import numpy as np
import matplotlib.pyplot as plt  
#h5py is a common package to interact with a dataset that is stored on an H5 file.
import h5py 
import scipy
#from PTL import  Image
from scipy import ndimage
from basics import sigmoid #加载sigmoid函数（自写）

# Loading the data (cat/non-cat)
from lr_utils import load_dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


#Common steps for pre-processing a new dataset are:
#（1）Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, …)
#（2）Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
#（3）“Standardize” the data

#Example of apicture
index=10
#plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
#np.squeeze可以压缩维度

m_train = train_set_x_orig.shape[0]#训练样本数
m_test = test_set_x_orig.shape[0]#测试样本数
num_px = train_set_x_orig.shape[1]#由于图片是正方形，宽=高
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b∗c∗d, a) is to use: 
#X_flatten = X.reshape(X.shape[0], -1).T     
 # X.T is the transpose of X
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. 
#But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#The main steps for building a Neural Network are:
#1. Define the model structure (such as number of input features)
#2. Initialize the model’s parameters
#3. Loop:
#- Calculate current loss (forward propagation)
#- Calculate current gradient (backward propagation)
#- Update parameters (gradient descent)

def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))#断言函数
    return w,b
dim = 2
w,b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m=X.shape[1]
    #列数
    # FORWARD PROPAGATION (FROM X TO COST)
    Z=np.dot(w.T,X)+b
    A=sigmoid(Z)
    cost=-(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dZ=A-Y
    db=(1.0/m)*np.sum(dZ)
    dw=(1.0/m)*np.dot(X,dZ.T)
    
    assert(dw.shape==w.shape)
   # assert(db.shape==float)
    assert(db.dtype==float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads={"dw":dw,
           "db":db}
    return grads,cost
#np.array([1,2,1])   size=（3，）
#np.array([[1,2,1]]) size=(1,3)
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print("dw="+str(grads["dw"]))
print("db="+str(grads["db"]))
print("cost="+str(cost))

# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
     Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs=[]
    for i in range(num_iterations): 
        grads, cost = propagate(w, b, X, Y)
        dw=grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
         # Record the costs
        if i % 100 == 0:
            costs.append(cost)#列表
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost)) #%i 整型 %f 浮点型
    params={"w":w,
            "b":b}
    grads={"dw":dw,
           "db":db}
            
    return params,grads,costs
        
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
def predict(w,b,X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0][i]>0.5:
            Y_prediction[0][i]=1
        else:
             Y_prediction[0][i]=0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 200, learning_rate = 0.5, print_cost = False):
    w,b= initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w=parameters["w"]
    b=parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
if __name__=='__main__':
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    # Example of a picture that was wrongly classified.
    index = 1
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    plt.show()
    print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate="+str(d["learning_rate"]))
    plt.show()
    #The learning rate α determines how rapidly we update the parameters. If the learning rate is too large we may “overshoot” the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. 
    #Choose the learning rate that better minimizes the cost function.
    #If your model overfits, use other techniques to reduce overfitting.
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')
    
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    
    plt.ylabel('cost')
    plt.xlabel('iterations')
    
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    ## START CODE HERE ## (PUT YOUR IMAGE NAME) 
    my_image = "dog.jpg"   # change this to the name of your image file 
    ## END CODE HERE ##
    
    # We preprocess the image to fit your algorithm.
    
    image = np.array(ndimage.imread(my_image, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px,3)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)
    
    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")