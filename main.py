import os

import h5py
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow.python.data.ops.dataset_ops
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

tf.__version__

#Data loading

train_set = h5py.File("train_signs.h5", 'r')
test_set = h5py.File("test_signs.h5", 'r')


x_train = tf.data.Dataset.from_tensor_slices(train_set["train_set_x"])
y_train = tf.data.Dataset.from_tensor_slices(train_set['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_set['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_set['test_set_y'])


type(x_train)

#describes how an image inside tensor flow is , f.e TensorSpec(shape=(64, 64, 3), dtype=tf.uint8, name=None)

print(x_train.element_spec)

#take images via iterators when needed
print(next(iter(x_train)))


#--------------------------------------------------------------------------------------------------#

#create a set , which takes the labels y_train from dataset. Using numpy the labels turn into number and add to set.
unique_labels = set()
for element in y_train:
    unique_labels.add(element.numpy())
print(unique_labels)


images_iter = iter(x_train)
labels_iter = iter(y_train)

plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(next(images_iter).numpy().astype("uint8"))
    plt.title(next(labels_iter).numpy().astype("uint8"))
    plt.axis("off")
    plt.savefig("dataset_sample.png")


def normalize(image):

    image = tf.cast(image,tf.float32) #/255.0
    image = tf.reshape(image,[-1]) #  tensor which contains 12288 numbers
    return image

new_train = x_train.map(normalize)
new_test = x_test.map(normalize)



def Sigmoid(z):
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a


#one dimension matrix with the 6 possible categories
def one_hot_matrix(label, C=6):
    one_hot=tf.reshape(tf.one_hot(label , C ,axis=0),shape=[C,])
    return one_hot


def initialize_parameters():
    # Glorot Normal creates numbers from  normal distribution
    initializer = tf.keras.initializers.GlorotNormal(seed=1)


    W1 = tf.Variable(initializer(shape=(25,12288)))
    b1 = tf.Variable(initializer(shape=(25,1)))
    W2 = tf.Variable(initializer(shape=(12,25)))
    b2 = tf.Variable(initializer(shape=(12,1)))
    W3 = tf.Variable(initializer(shape=(6,12)))
    b3 = tf.Variable(initializer(shape=(6,1)))

    parameters={"W1" : W1,
                "b1" : b1,
                "W2" : W2,
                "b2" : b2,
                "W3" : W3,
                "b3" : b3}
    return parameters

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.linalg.matmul(W1,X),b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.add(tf.linalg.matmul(W2,A1),b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.add(tf.linalg.matmul(W3,A2),b3)
    return Z3

def compute_total_loss(logits,labels):

    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels),tf.transpose(logits),from_logits=True))
    return total_loss

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 800,minibatch_size = 32,print_cost=True):
    # 3-layer tensor flow neural network
    # Linear -> ReLu ->Linear ->ReLu ->Linear ->Softmax
    #X-train = training_set (12288,1080( training_examples number))
    #Y_train = test set, of shape (output size = 6, number of training examples = 1080)
    #X_test =training set, of shape (input size = 12288, number of training examples = 120)
    #Y_test = test set, of shape (output size = 6, number of test examples = 120)

    costs = []
    train_acc=[]
    test_acc=[]
    parameters = initialize_parameters()

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    #automatically measures the percentage of correct model predictions
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    #Combines the images with the labels in each respective category into a .zip file.
    dataset = tf.data.Dataset.zip((X_train,Y_train))
    test_dataset = tf.data.Dataset.zip((X_test,Y_test))


    total_elems = dataset.cardinality().numpy()


    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

    for epoch in range(num_epochs):
        epoch_total_loss = 0
        train_accuracy.reset_state()


        for(minibatch_X , minibatch_Y) in minibatches :
            #A log file is specified that records the network's guesses as well as the final error.
            with tf.GradientTape() as tape:

                #predict
                Z3 = forward_propagation(tf.transpose(minibatch_X),parameters)
                #loss
                minibatch_total_loss = compute_total_loss(Z3,tf.transpose(minibatch_Y))

            # tf.transpose(Z3)-> in order to match with  minibatch_Y(32*6)
            train_accuracy.update_state(minibatch_Y,tf.transpose(Z3))

            trainable_variables = [W1,b1,W2,b2,W3,b3]

            #back prop
            grads = tape.gradient(minibatch_total_loss,trainable_variables)


            optimizer.apply_gradients(zip(grads,trainable_variables))

            epoch_total_loss +=minibatch_total_loss

        epoch_total_loss/=total_elems


        if print_cost == True and epoch %100== 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
            print("Train accuracy:", train_accuracy.result())



            for (minibatch_X,minibatch_Y) in minibatches :

                Z3 = forward_propagation(tf.transpose(minibatch_X),parameters)

                test_accuracy.update_state(minibatch_Y,tf.transpose(Z3))
            print("Test_Accuracy:", test_accuracy.result())



            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())

            test_accuracy.reset_state()


    return parameters,costs,train_acc,test_acc


new_y_train = y_train.map(one_hot_matrix)
new_y_test = y_test.map(one_hot_matrix)
parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=800)

plt.figure()
plt.plot(np.squeeze(costs))
plt.ylabel('Cost')
plt.xlabel('Epochs (per 50)')
plt.title('Learning rate = 0.0001')
plt.savefig("loss_curve.png")
