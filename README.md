# Multi-Layer Perceptron with NumPy
<p align="center">
<img width="425" height="200" alt="Screenshot from 2025-10-23 10-49-36" src="https://github.com/user-attachments/assets/4900c620-3458-4087-9b65-dcd5876a8aa7" />
</p>
The project presents the realization of an MLP neural network with 4 hidden
layers and aims to recognize numbers in images. The database is imported from
tensorflow.keras.datasets.mnist.load_data() and contains 60000 training data and
10000 validation data
<br />.
<p align="center">
<img width="501" height="107" alt="Screenshot from 2025-10-23 10-55-31" src="https://github.com/user-attachments/assets/d740acc4-16d3-41ac-9702-7e4ab448d037" />
</p>

  Project includes:
  * Data per-processing
  * Feedforward
  * Backpropagation
  * Gradient Descent
  * Mean Squared Error
  * Accuracy measure
  * Code implementation
<br />
<br />

## Data per-processing
The structure of the data retrieved from tensorflow.keras.datasets.mnist.load_ data() has the following form:
* The training data consists of 60000 images of size 28X28 bits with a cullet depth of 8 bits (gray scale), illustrating numbers and 60000 labels of size 1 digit (digits from 0 to 9 ).
* The validation data consists of 10000 images of 28X28 size with the same color feature, illustrating numbers and 10000 labels of 1 digit size.
  
The data pre-processing is performed as follows:
* The 28X28 image existing in the arrays 60000X28X28 and 10000X28X28, respectively, is transformed into a vector of 784 elements. Consequently, the two matrices will have the form 60000X784 and 10000X784 respectively.
* Data normalization is performed with the equation <img width="116" height="22" alt="image" src="https://github.com/user-attachments/assets/c5a10d44-8233-4a23-a416-4ff4ff5adbb1" /> esulting in values between 0 and 1. In this case we can divide the two matrices that have the image characteristics, to the value of 255 because the 8-bit gray scale has values between 0 and 255.
* The labels corresponding to the images have the form 60000X1 and 10000X1 respectively. The values of these matrices will be transformed into categorical values. For example the value  0 = |1 0 0 0 0 0 0 0 0 0| ,  1 = |0 1 0 0 0 0 0 0 0 0| , 2 = |0 0 1 0 0 0 0 0 0 0| ... Etc.   This gives the matrices 60000X10 and 10000X10 respectively.
* For a good management of computational resources and a better optimization of the model (neural network), it is recommended to divide the training data into smaller batches. For this project we have chosen the batch size of 100, thus the shape of the feature matrices as well as the labeling data, will be: 600X100X784X784 and 600X100X784X10 respectively
<br />
<br />

## Feedforward


<br />
<br />



<br />
<br />



<br />
<br />



<br />
<br />

<br />
<br />


<br />
<br />


<br />
<br />


<br />
<br />

Bibliography:
* https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
* https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
* https://machinelearning.tobiashill.se/2018/12/04/part-2-gradient-descent-and-backpropagation/
* https://www.3blue1brown.com/lessons/backpropagation-calculus
* http://neuralnetworksanddeeplearning.com/chap2.html
* https://hmkcode.com/ai/backpropagation-step-by-step/
* https://sefiks.com/2017/01/21/the-math-behind-backpropagation/#google_vignette
* https://medium.com/@samuelsena/pengenalan-deep-learning-part-3-backpropagation-algorithm-720be9a5fbb8
* https://pabloinsente.github.io/the-multilayer-perceptron
* https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0/
* https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
