**General Info about the project :** 

This project was developed during the **Deep Learning Specialization** course by Andrew Ng.It applies the theoretical 
concepts of forward propagation, backpropagation and optimization algorithms tαught during the course, using the TensorFlow.
This Deep Neural Network in python designed for image classification. The model is trained on a dataset of sign language
digits and classifies hand gestures into one of six categories(0 - 5).

**Architecture and Algorithms:**

This model is a Deep neural network with two hidden layers. The input consists of an image dataset,processed using the 
TensorFlow, where each image is flattened and fed into the model as a 1D tensor of 12,288 features.The first hidden 
layer contains 25 neurons and the second has 12. The output layer has 6, each one representing one specific sign language.


Optimization techniques that used , in order to make our model as accurate as possible:

i)**mini batch gradient descent**

ii)**adam optimizer**

**Libraries :**

i)**h5py **         (for loading tha datase files)

ii)**numpy  **      (for numerical and matrix operations)

iii)**tensorflow**  (for building and training the neural network)

iv)**matplotlib**   (for data visualization - loss curve)



**How to run :**

Make sure you clone this repository to your local machine : git clone 
You must have both the dataset files,** train_signs.h5 and test_signs.h5**, into the root directory, in the same folder as main.
    
Install the follow requirements :** pip install tensorflow numpy h5py matplotlib**

Run the script :** python main.py**

**Conclusion**

During training, the loss started at 46.589901 and converged to (). Of course if you change the learnign rate,the starting 
and final loss would be different. We initialize the learning rate = 0.0001.Even if it is relatively low,the Adam 
itselfs  optimizes the learning rate to its best value.

The final loss reached  a satisfactory level, even though at the last 2 epochs remains the same.Training intentionally 
stopped at this point,as adding more epochs would not significantly decrease the loss and could lead to model overfitting
