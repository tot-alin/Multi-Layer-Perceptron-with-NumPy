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
Feedforward, the neural network's process of passing data from the input layer to the output layer. This process is carried out by each perceptron of the network, calculating the weighted sum of inputs 1.1a, and passed to the activation function 1.1b. Figure 1 illustrates the shape of a perceptron, where X1..p - represents the input, w1..P - the weighting for each input, b - bias (balancing value), S - the weighted sum 1.1a and A- the activation function
<br />fig. 1
<p align="center">
<br /><img width="450" height="330" alt="Screenshot from 2025-10-23 11-14-51" src="https://github.com/user-attachments/assets/08715390-acfd-4297-8c93-3e58ca101a3d" />
</p>
<br /><img width="112" height="23" alt="image" src="https://github.com/user-attachments/assets/15a17722-d94d-4e87-b91e-599052d965e4" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.1a)

<br /><img width="107" height="41" alt="image" src="https://github.com/user-attachments/assets/41e433af-8eb0-44af-adaa-ffd6b7c6293c" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.1b)

<br />S - weighted sum, w - weighting, x - input values , f() - activation function , A - activation function result

In accordance with the above equations, the relations and their matrix form for a neural network with 4 hidden layers are given below.  The data input, as shown in matrix 1.2, is a matrix 100 x 784 . The size of each line, is a 28X28 pixel image transformed into a vector of 784 . The number of lines is the number of features in the data packet that goes into the processing, 100 in this case.

<br /><img width="233" height="68" alt="image" src="https://github.com/user-attachments/assets/5b87b6b0-01d7-4a94-a8de-b25427fb125d" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.2)

The initialization of the weight matrices is performed by generating a randomized matrix where the number of rows is the feature size and the number of columns is the number of perceptrons in that layer, i.e. the number of outputs. For example matrix 1.3, the matrix of weights on the first hidden layer has the number of rows equal to the size of the resulting vector from the 28X28pixel image i.e. 784. The number of columns, as can be seen in matrix 1.3, is the number of outputs to layer 2.

<br /><img width="308" height="109" alt="image" src="https://github.com/user-attachments/assets/4898d59d-8742-4614-b2d0-c25db2dbe4a5" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.3)

Initializing the bias vector, it is a matrix with a single row and has the number of scalars eqal numbers layers. 

Equation 1.4 relates the weighted sum of the input layer to which the bias vector is added. The shape of the SL1 matrix is 100x64, the shape resulting from multiplying the matrix X (100x784) by the matrix wL1 (784x64). For equations 1.6, 1.8 and 1.10, the approach is identical but the matrix X is replaced by the matrix resulting from the activation function in the previous layer, and the weighted matrix w is the matrix related to the respective layer.

<br /><img width="118" height="21" alt="image" src="https://github.com/user-attachments/assets/cfae6d6d-f1f6-41ee-8b52-d64682942bc9" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.4)

<br /><img width="529" height="68" alt="image" src="https://github.com/user-attachments/assets/d2fa78c6-e2f5-49a0-8f4e-04cd85900a99" />
<br />matrix form of eucation 1.4

The activation functions expressed in Equations 1.5, 1.7, 1.9 and 1.11 are the weighted sum matrix for each layer, to which the sigmoid function (in this case, Equation 1.2) is applied for the element in the matrix.

<br /><img width="82" height="22" alt="image" src="https://github.com/user-attachments/assets/bbe24480-dd95-4b97-824e-59e1352b3a6b" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.5)

<br /><img width="306" height="72" alt="image" src="https://github.com/user-attachments/assets/3180d3a3-b64a-49b3-afc0-e20963e3d78e" />

<br /><img width="130" height="21" alt="image" src="https://github.com/user-attachments/assets/ca8f38a8-482f-4753-baf2-01f29a0d0c8f" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.6)

<br /><img width="543" height="68" alt="image" src="https://github.com/user-attachments/assets/5b18cb44-d9b0-4198-90f9-ed0e3c7be328" />

<br /><img width="82" height="22" alt="image" src="https://github.com/user-attachments/assets/9f4370c4-71ee-4459-a633-252f6c0029f0" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.7)

<br /><img width="298" height="72" alt="image" src="https://github.com/user-attachments/assets/cc4d068c-ee83-40ad-bdc3-1fc8a1ea6f02" />


<br /><img width="130" height="21" alt="image" src="https://github.com/user-attachments/assets/9b2a7749-21b7-4640-b360-74e71b119d81" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.8)

<br /><img width="549" height="68" alt="image" src="https://github.com/user-attachments/assets/f68b07ad-d51b-4380-a256-060009fe2bac" />

<br /><img width="82" height="22" alt="image" src="https://github.com/user-attachments/assets/03d302c8-72ee-4517-bd8e-258a0783dd58" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.9)

<br /><img width="306" height="72" alt="image" src="https://github.com/user-attachments/assets/1bbd91e1-96db-4b44-91dd-8156c835c57d" />

<br /><img width="132" height="21" alt="image" src="https://github.com/user-attachments/assets/0cb74c4b-fdfb-4797-b0bf-ce40bf91123a" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.10)

<br /><img width="551" height="68" alt="image" src="https://github.com/user-attachments/assets/7797ae75-acdf-4cba-95ea-8867d0ea1d01" />

<br /><img width="83" height="22" alt="image" src="https://github.com/user-attachments/assets/f7f86100-f218-4af8-8097-f4d595c2d651" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (1.11)

<br /><img width="307" height="72" alt="image" src="https://github.com/user-attachments/assets/7956e98c-9c92-47c8-b004-ea127ce247ea" />

<br />S Ln – weighted sum matrix of layer Ln ;  w Ln – weight matrix Ln ; X – matrix of input data ; ALn – the matrix resulting from the activation function (sigmoid in this case) for the Ln 
<br />
<br />

## Backpropagation
Backpropagation is the method of training the neural network in order to reduce the difference between the predicted and the actual outcome. As the name suggests it has the reverse direction to Feedforward, resulting in gradients with which the weights (w) are adjusted.

Determining gradients is done using the chain rule, which helps us find the derivative of a compound function.  The case of a function <img width="79" height="20" alt="image" src="https://github.com/user-attachments/assets/707b193d-e490-413e-86b0-f267c4a84082" /> , where g -is the function of x, and f is the function of g, the result is the derivative of y with respect to x is (2.1):

<img width="87" height="39" alt="image" src="https://github.com/user-attachments/assets/8915bc40-779f-418a-a74a-44755fe657fb" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.1)

According to the principle stated above and considering equation 1.1, we can write the partial derivative equation of the correction gradient <img width="26" height="39" alt="image" src="https://github.com/user-attachments/assets/1a62c8db-f75c-45bd-8320-35677d58ba73" /> for a single-layer linear perceptron, as (2.2).

<img width="199" height="47" alt="image" src="https://github.com/user-attachments/assets/c510a5ea-44cf-4650-a438-78273b8bbdf5" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.2)
E – loss, w – coefficients weights, <img width="61" height="23" alt="image" src="https://github.com/user-attachments/assets/6612c2fd-5395-462e-b63d-0975948510b4" /> – weighted sum, X – input values

Using the example of equation 2.2 and the chain rule with multiple consecutive functions, we can determine the partial derivative equations for a model with four hidden layers. Based on Figure 2, which represents a sketch of the model's operation, we can deduce the partial derivatives and their order.

<br />fig. 2
<p align="center">
<br /><img width="1303" height="451" alt="Screenshot from 2025-10-23 12-36-48" src="https://github.com/user-attachments/assets/3473bc0c-71f9-42b4-9fb6-c71c8435c551" />
</p>

This gives equations 2.3 which determine the gradients of the weights on the 4 hidden layers.

<img width="450" height="190" alt="Screenshot from 2025-10-23 12-39-40" src="https://github.com/user-attachments/assets/7a66e85c-12c3-44cd-899c-08f0cef8f9a9" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.3)

E –loss; A Ln – activation function for layer n; S Ln – weighted sum on layer n; w Ln – weight of layer n

Similar to the determination of the gradients of the weights we determine the gradients for the bias, but replacing w L by b L. This is a vector that has the size of the number of perceptrons in the layer.

<img width="450" height="190" alt="Screenshot from 2025-10-23 12-44-25" src="https://github.com/user-attachments/assets/1c028f11-a9a1-4a37-930f-88ad4253d80a" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.4)

b Ln – bias for layer n

Taking into account that the neural network has a repetitive form with respect to the positioning of the perceptrons composing it, we can generalize the relations for the partial derivatives as follows:
* For the output layer, <img width="33" height="41" alt="image" src="https://github.com/user-attachments/assets/c69f152b-70ea-4bac-8fd6-43944db7dffb" /> represents the change in loss (E) as a function of the change in the activation function result (AL). In this situation we have chosen the simplest loss function, i.e. the difference between the model output (AL) and the actual output (Y) used during training

<br /><img width="84" height="41" alt="image" src="https://github.com/user-attachments/assets/3fa14ae7-6519-49e4-b348-524c039224a5" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.5)

* The influence of the result of the activation function (A L = f(SL)) on the weighted sum (S L), is equal to the derivative of the activation function <img width="42" height="23" alt="image" src="https://github.com/user-attachments/assets/bdca3c34-a906-4fe3-9f12-80ad94315088" /> In this case we have chosen the sigmoid function, so the derivative of this function is <img width="156" height="22" alt="image" src="https://github.com/user-attachments/assets/743e1bfa-e1ef-4568-af4c-bda2f5045bca" /> ( taken from the literature)

<img width="282" height="44" alt="image" src="https://github.com/user-attachments/assets/c48c7ae9-5ba6-4bb1-864b-e7f836e04955" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.6) 
  
* Equation 2.7 represents the change in the weighted sum of the upper layer  S L+1 as a function of the inputs assigned to it from the output of the lower layers A L . This is equal to the value of the weights in the upper layer w L+1. We consider b L+1 = 0, since we are interested in the modification of S L+1 as a function of A L .

<img width="347" height="44" alt="image" src="https://github.com/user-attachments/assets/b2d8564c-58e4-4d69-8179-b971b38e0336" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.7) 

* The derivative of the weighted sum S L with respect to the weight w L, is equal to the result of the activation function in the previous layer A L-1 , except for the first layer because A L-1 becomes the input to the model, i.e. X. We consider b L = 0, since we are interested in the evolution of S L as a function of w L.

<img width="320" height="44" alt="image" src="https://github.com/user-attachments/assets/41ce54a0-cb2d-4c7c-9645-2dfbc56289d2" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.8)

* The weighted sum S L with respect to b L, is equal to 1 because the inputs      A L-1 and the weights w are 0, because we are interested in the evolution of    S L as a function of b L.

<img width="273" height="44" alt="image" src="https://github.com/user-attachments/assets/47dfec4d-36da-407a-a5d4-609b9ca6736c" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.9)

Considering equations 2.5, 2.6, 2.7, 2.8 and 2.9, equations 2.3 and 2.4 can be written as 2.10 below.

Index T is the transposed matrix for realizing the junctions between layers. If for forward propagation, the matrix w L realizes the junction between the shape of the input matrices in the L-layer and the output shape, for back propagation the reverse effect must be realized.This is done by rotating the matrix from the bottom left corner to the top right corner.

<br /><img width="211" height="41" alt="image" src="https://github.com/user-attachments/assets/38aa5287-4ee9-447a-bef7-755319c43d29" />

<br /><img width="310" height="41" alt="image" src="https://github.com/user-attachments/assets/672b07fd-60a0-45d2-90db-6c7078f4ae63" />

<br /><img width="408" height="41" alt="image" src="https://github.com/user-attachments/assets/e4f07bd1-6080-4adb-a374-eafa715a570a" />

<br /><img width="501" height="41" alt="image" src="https://github.com/user-attachments/assets/2587fb67-8352-4ca0-9cf3-d75eacd7e7b6" />

<br /><img width="192" height="41" alt="image" src="https://github.com/user-attachments/assets/7896ebe5-fb15-4e3f-931f-15430b62c139" />

<br /><img width="290" height="41" alt="image" src="https://github.com/user-attachments/assets/f6490c08-8415-4081-a6b3-ffb4a5182e55" />

<br /><img width="388" height="41" alt="image" src="https://github.com/user-attachments/assets/0ee24199-d899-4c9f-a6ff-df1f7c4449e3" />

<br /><img width="379" height="41" alt="image" src="https://github.com/user-attachments/assets/b72e9371-4583-4f75-93b1-63dadb4f00c8" />

<br /><img width="335" height="23" alt="image" src="https://github.com/user-attachments/assets/940a1a30-d746-4c24-911a-b10417e0f6b6" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.10)

To simplify the implementation process, we consider
* for the last layer <img width="220" height="44" alt="image" src="https://github.com/user-attachments/assets/53ddc60f-83f6-46c0-884d-b95a87e0fef9" />
* for the rest of the layers <img width="242" height="44" alt="image" src="https://github.com/user-attachments/assets/089d251e-4357-45d4-997f-a2c40adc9e75" />

And we can write equations 2.10 excluding the last term from 2.11

<br /><img width="150" height="25" alt="image" src="https://github.com/user-attachments/assets/3829eec3-0b9f-4625-a196-ee2587135f77" />

<br /><img width="569" height="72" alt="image" src="https://github.com/user-attachments/assets/244e0dcb-03b5-47bf-a3f3-eb5342f10337" />

<br /><img width="141" height="23" alt="image" src="https://github.com/user-attachments/assets/3aa5619b-a853-4de9-a964-3e263804e34a" />

<br /><img width="579" height="72" alt="image" src="https://github.com/user-attachments/assets/298b3581-9181-4cd7-b842-88447f1af487" />

<br /><img width="140" height="23" alt="image" src="https://github.com/user-attachments/assets/73c0d4a1-42b9-4f88-81c0-132b0e37e1b7" />

<br /><img width="570" height="72" alt="image" src="https://github.com/user-attachments/assets/ca23ef8d-856d-4019-ac07-e575bffdb6b7" />

<br /><img width="139" height="23" alt="image" src="https://github.com/user-attachments/assets/f9324153-e5f5-459c-9725-a6e0d85ad973" />

<br /><img width="582" height="72" alt="image" src="https://github.com/user-attachments/assets/d67c8bb8-0f10-44f9-8538-9f43ebecba54" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.11)

Considering equations 2.10 and 2.11 results in the following form of the gradient determinant (2.15 and 2.16).  The form of the gradient matrix is identical to the form of the weighted matrix

<br /><img width="162" height="181" alt="image" src="https://github.com/user-attachments/assets/3dea45fc-71f6-49d5-89a7-57a6086ec8c7" />

<br /><img width="296" height="68" alt="image" src="https://github.com/user-attachments/assets/26491bc0-6142-443a-aa33-d254d238b755" />

<br /><img width="295" height="68" alt="image" src="https://github.com/user-attachments/assets/91dc4678-fa50-46e6-834a-99b7eeda6ee0" />

<br /><img width="308" height="68" alt="image" src="https://github.com/user-attachments/assets/337c5fa6-e4e9-463f-a169-c8647c8988c0" />

<br /><img width="349" height="68" alt="image" src="https://github.com/user-attachments/assets/12009a4e-46f9-4a0f-8a2d-615f05d17b6a" />   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.12)

As it follows from equations 2.10, the gradients of b L are equal to δ L and the composition of the gradients resulting from each input feature (100 in this case) is realized by summing over the vertical

<br /><img width="118" height="181" alt="image" src="https://github.com/user-attachments/assets/f80ec163-cbdf-4f1b-b0a8-9511afefabe5" />

<br /><img width="244" height="72" alt="image" src="https://github.com/user-attachments/assets/a1b12d5f-08c7-44d5-ae96-655f420c4d95" />

<br /><img width="237" height="72" alt="image" src="https://github.com/user-attachments/assets/68ef6615-6caf-47cf-a28c-0294532020da" />

<br /><img width="230" height="72" alt="image" src="https://github.com/user-attachments/assets/57388d8c-5f6f-4199-b591-13e8ce842541" />

<br /><img width="237" height="72" alt="image" src="https://github.com/user-attachments/assets/a89bc371-5cda-4cdb-9b55-86b87e1e3b60" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2.13)
<br />
<br />

## Gradient Descent
Gradient descent is a method of optimizing a model by finding a local minimum of a differential function. In machine learning, it has the role of correcting the weights used in the neural network. The generalized form as well as the way of working is expressed in equation 3.1

<br /><img width="136" height="21" alt="image" src="https://github.com/user-attachments/assets/d6f43d0c-b8a2-4312-8a5c-f6007841f71e" />


<br /><img width="128" height="21" alt="image" src="https://github.com/user-attachments/assets/bd394af6-bd54-4388-87d0-acc8ac756266" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (3.1)

<br /><img width="11" height="19" alt="image" src="https://github.com/user-attachments/assets/7b258c5d-c966-48e2-8dbf-1d054812cb4d" /> – learning rate

Equation 2.3 presents the weight and bias optimization approach for each layer

<br /><img width="569" height="41" alt="image" src="https://github.com/user-attachments/assets/24184904-d865-4869-bc71-768af8688cc9" />

<br /><img width="530" height="41" alt="image" src="https://github.com/user-attachments/assets/dfbb90e5-436c-414b-a2a4-c8373058613a" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (3.2)
<br />
<br />

## Mean Squared Error
The mean squared error is a method of expressing the errors that a model may have. It is realized according to equation 4.1, being the average of the squares of the difference between the predicted and the actual result.

<img width="140" height="39" alt="image" src="https://github.com/user-attachments/assets/05ce1cb0-1fb9-4219-9780-ab67fc4354eb" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (4.1)

<br />
<br />

## Accuracy metric
For classification tasks, this method provides a quick information of the model performance in terms of the correctness of the delivered results. The accuracy expresses the ratio of the number of correct results to the total number of results

<img width="297" height="39" alt="image" src="https://github.com/user-attachments/assets/6641ac34-d8c1-4e27-b788-60be140e879f" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (5.1)

<br />


<br />
<br />


<br />
<br />

## Bibliography:
https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

https://machinelearning.tobiashill.se/2018/12/04/part-2-gradient-descent-and-backpropagation/

https://www.3blue1brown.com/lessons/backpropagation-calculus

http://neuralnetworksanddeeplearning.com/chap2.html

https://hmkcode.com/ai/backpropagation-step-by-step/

https://sefiks.com/2017/01/21/the-math-behind-backpropagation/#google_vignette

https://medium.com/@samuelsena/pengenalan-deep-learning-part-3-backpropagation-algorithm-720be9a5fbb8

https://pabloinsente.github.io/the-multilayer-perceptron

https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0/

https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
