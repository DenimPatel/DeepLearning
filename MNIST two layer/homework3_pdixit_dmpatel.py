import numpy as np
X_train = np.load("mnist_train_images.npy")
y_train = np.load("mnist_train_labels.npy")
X_test = np.load("mnist_test_images.npy")
y_test = np.load("mnist_test_labels.npy")
X_validation = np.load("mnist_validation_images.npy")
y_validation = np.load("mnist_validation_labels.npy")
w = np.random.random([X_train.shape[1], y_train.shape[1]])
b = np.random.random([1,y_train.shape[1]])
model = {
    "w": w,
    "b": b
}    
def softmax(z):
#     z -= np.max(z)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1))

def forward_pass(x, model):
    w= model["w"]
    b= model["b"]
    z = (x@w+b)
    a = softmax(z).T
#     y = np.argmax(a, axis=1)
    return a

def calculate_CE_loss(a, one_hot_y):
    cost = (-1 / a.shape[0]) * np.sum(one_hot_y * np.log(a))
    return cost

def calculate_regularization(w,alpha):
    REGULARIZATION = 0.5 * alpha * (w.T@w) 
    return REGULARIZATION
def backprop(prediction, x, y, model, alpha=0.01, lr=0.00001):
    w= model["w"]
    b= model["b"]
    grad_w = (-1/(x.shape[0]))*(x.T@(y-prediction)) + alpha * w
    w = w - lr*grad_w
    grad_b = -1*np.mean(y-prediction, axis = 0)
    b = b - lr*grad_b
    model["w"] = w
    model["b"] = b
    return model
def evaluate(x = X_validation, y = y_validation, model=model, alpha=0.1):
    prediction = forward_pass(x, model)
    CE_loss = calculate_CE_loss(prediction, y)
    w= model["w"]
    reg_loss = calculate_regularization(w,alpha)
    return CE_loss+reg_loss
def accuracy(y, y_true):
    return (sum(y == y_true)/len(y_true)) * 100

# Hyperparameters - GRID SEARCH
lrs = [3.3e-2]
alphas = [0.001]
batch_sizes = [400]
epochs = [500]

best_loss = 1e5
best_accuracy = 0
best_params = {}

for lr in lrs:
    for alpha in alphas:
        for batch_size in batch_sizes:
            for epoch in epochs:
                w = np.random.random([X_train.shape[1], y_train.shape[1]])
                b = np.random.random([1,y_train.shape[1]])
                model = {
                    "w": w,
                    "b":b}  
                batches = (int)(len(X_train)/batch_size)
#                 print(batches)
                for _ in range(epoch):
                    for i in range(batches):
    #                     print(i)
                        data_x, data_y = X_train[(i)*batch_size:(i)*batch_size+batch_size,], y_train[(i)*batch_size:(i)*batch_size+batch_size,]
#                         print(len(data_x))
                        a = forward_pass(data_x, model)
    #                     CE_loss = calculate_CE_loss(a, data_y)
    #             #         regularization_loss = calculate_regularization(w,alpha)
    #             #         cost = CE_loss + regularization_loss
    #             #         print(cost)
                        output = np.argmax(a, axis=1)
                        actual_y = np.argmax(data_y, axis=1)
                #         print("train accuracy= ",accuracy(output,actual_y),"train loss = ", calculate_loss(a, data_y))
                        model = backprop(a, data_x, data_y, model, alpha, lr)

                 #                 save best model

            print("Perfomance on validation")
            a = forward_pass(X_validation, model)
            output = np.argmax(a, axis=1)
            actual_y = np.argmax(y_validation, axis=1)
            validation_accuracy = accuracy(output,actual_y)
            valdiation_loss = calculate_CE_loss(a, y_validation)+calculate_regularization(w.ravel(),alpha) 
            print("lr: ", lr, "\t alpha: ", alpha, "\t batch_size: ", batch_size, "\t epoch: ", epoch)
            print("\t val_acc: ",validation_accuracy,"\t val_loss ", valdiation_loss)
            print("")
            if(validation_accuracy>best_accuracy):
                best_accuracy = validation_accuracy
                best_model = model
                best_params = {
                    "lr" : lr,
                    "alpha": alpha,
                    "batch_size": batch_size,
                    "epoch":epoch
                }

                
## for testing the best model
a = forward_pass(X_test, best_model)
output = np.argmax(a, axis=1)
actual_y = np.argmax(y_test, axis=1)
print("test accuracy= ",accuracy(output,actual_y),"test loss = ", calculate_CE_loss(a, y_test))


'''
Hyper parameters that we found
learning rate: 0.033
alpha: 0.001
batch_size: 400
epoch: 500
test_accuray: 91.66
test_loss: 0.2963
'''