import numpy as np
import matplotlib.pyplot as plt

X_train = np.load("mnist_train_images.npy")
y_train = np.load("mnist_train_labels.npy")
X_test = np.load("mnist_test_images.npy")
y_test = np.load("mnist_test_labels.npy")
X_validation = np.load("mnist_validation_images.npy")
y_validation = np.load("mnist_validation_labels.npy")


def softmax(z):
#     z -= np.max(z)
    z = (np.array(z).T - np.max(z, axis = 1)).T
    return (np.exp(z).T / np.sum(np.exp(z), axis=1))

def accuracy(y, y_true):
    return (sum(y == y_true)/len(y_true)) * 100

def relu(x):
    activated = np.maximum(0, x) #just for clarification
    return activated
def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def init_params(input_size = 784, output_size=10, many_hidden_layers=4):
    params = {}
    nodes_in_layers = [75, 50, 25, 45, 30, 25, 20]
    for layer in range(many_hidden_layers+1):
        if(layer==0):
            c = nodes_in_layers[layer+1] ** -0.5
            params["w" + str(layer)]=np.random.uniform(-1*c/2,c/2,(input_size, nodes_in_layers[layer]))
            
        elif(layer==(many_hidden_layers)):
            c = nodes_in_layers[layer+1] ** -0.5
            params["w" + str(layer)]=np.random.uniform(-1*c/2,c/2,(nodes_in_layers[layer-1], output_size))
        else:
            c = nodes_in_layers[layer+1] ** -0.5
            params["w" + str(layer)]=np.random.uniform(-1*c/2,c/2,(nodes_in_layers[layer-1], nodes_in_layers[layer]))

        if(layer==0):
            params["b" + str(layer)]=np.full([1, nodes_in_layers[layer]], 0.01)
        elif(layer==(many_hidden_layers)):
            params["b" + str(layer)]=np.full([1, output_size], 0.01) 
        else:
            params["b" + str(layer)] = np.full([1, nodes_in_layers[layer]], 0.01)
    return params

def forward_pass(data_x, params, many_hidden_layers):
    layer_cache = {}
    a_current = data_x
    for i in range(many_hidden_layers+1):
    #     affine transformation
        z_current = np.dot(a_current,params["w" + str(i)]) + params["b" + str(i)]
    #     activation 
        if(i == many_hidden_layers):
            a_current = softmax(z_current).T
        else:
            a_current = relu(z_current)
#         print("z mean = ", z_current.mean())
#         print("a mean = ",a_current.mean())
        layer_cache["z" + str(i)] = z_current;
        layer_cache["a" + str(i)] = a_current;
    return layer_cache


def calculate_derivatives(data_x, data_y,params, layer_cache, many_hidden_layers, alpha = 0.001):
    derivatives = {}
    batch = len(data_x)
    for i in reversed(range(many_hidden_layers+1)):
        if(i==many_hidden_layers):
            g = (np.subtract(layer_cache["a" + str(i)],data_y))
        else:
            g = np.multiply(np.dot(g,params["w" + str(i+1)].T),reluDerivative(layer_cache["z" + str(i)]))
        derivatives["b" + str(i)] = np.mean(g, axis = 0) 
        if(i==0):
            derivatives["w" + str(i)] = (1/batch)* np.dot(data_x.T,g)
            derivatives["w" + str(i)] += alpha * params["w" + str(i)]
        else:
            derivatives["w" + str(i)] = (1/batch)* np.dot(layer_cache["a" + str(i-1)].T,g)
            derivatives["w" + str(i)] += alpha * params["w" + str(i)]
    return derivatives


def update_weights(params, derivatives, many_hidden_layers,lr = 1e-3):
    for i in range(many_hidden_layers+1):
        params["w" + str(i)] -= lr*derivatives["w" + str(i)]
        params["b" + str(i)] -= lr*derivatives["b" + str(i)]
    return params

def calculate_accuracy(X, y,params, many_hidden_layers):
    layer_cache = forward_pass(X, params, many_hidden_layers)
    y_hat = layer_cache["a"+str(many_hidden_layers)]
    # x_current = softmax(y_hat).T
    output = np.argmax(y_hat, axis=1)
    actual_y = np.argmax(y, axis=1)
    accuracy_val = accuracy(output,actual_y)
    return accuracy_val  


many_hidden_layers = 2
params =init_params(input_size = 784, output_size=10, many_hidden_layers=many_hidden_layers)
train_acc_epoch = []
val_acc_epoch = []


lrs = [1e-2, 1e-3, 7e-3, 3.3e-2, 1e-3]
alphas = [0.06, 0.01, 0.001, 0.05, 0.005]
batch_sizes = [25, 50, 100, 200]
epochs = [15, 25, 50, 75=]
# epoch = 60
best_accuracy = 0
for lr in lrs:
    for alpha in alphas:
        for batch_size in batch_sizes:
            for epoch in epochs:
                batches = (int)(len(X_train)/batch_size)
                for cur_epoch in range(epoch):
                    for i in range(batches):
                        data_x, data_y = X_train[(i)*batch_size:(i)*batch_size+batch_size,], y_train[(i)*batch_size:(i)*batch_size+batch_size,]
                        layer_cache = forward_pass(data_x, params, many_hidden_layers)
                        derivatives = calculate_derivatives(data_x, data_y,params, layer_cache, many_hidden_layers, alpha = 0.001)
                        params = update_weights(params, derivatives, many_hidden_layers,lr = lr)

                    train_acc = calculate_accuracy(X_train, y_train,params, many_hidden_layers)
                    val_acc = calculate_accuracy(X_validation, y_validation,params, many_hidden_layers)
                    train_acc_epoch = np.append(train_acc_epoch, train_acc)
                    val_acc_epoch = np.append(val_acc_epoch, val_acc)
                    print("epoch =", cur_epoch, " training accuracy = ", train_acc,  " validation accuracy = ", val_acc)
test_acc = calculate_accuracy(X_test, y_test, params, many_hidden_layers)
print("test accuracy = ", test_acc)
x = np.linspace(0, len(val_acc_epoch),len(val_acc_epoch))
plt.plot(x, val_acc_epoch,'-', label="validation")
plt.plot(x, train_acc_epoch,'--', label="train")
plt.legend(loc="lower right")
plt.ylim(0, 110)
plt.show()















