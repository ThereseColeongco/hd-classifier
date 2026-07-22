# Heart Disease Classifier
Decided to use a neural network, specifically MLP, for this, even if UCI says accuracy and precision are highest with XGBoost classification and logistic regression, because I want to learn how to build a neural network using PyTorch.

## How MLPs Work
MLP stands for multilayer perceptron. It's the most basic kind of neural network. Neuron = a number. Each number aka activation corresponds to something... A neural network has more than 1 layer. activation in final layer of neurons represents how much the system thinks a certain feature corresponds with a certain target. There can be hidden layers in the network. Activations in 1 layer determine activations in next layer. 

Each layer should break down the problem into subproblems.

How do activations from 1 layer -> activations in next layer? For 1 neuron:
1. Assign weight to connections between layers (can be positive or negative) then take the sum of activations of previous layer * corresponding weight
2. You can add a bias for inactivity to the sum expression (bias = just a number)
3. Put this weighted sum + the bias into an activation function (e.g. sigmoid [slow learner] or ReLU [more closely mimics neurons in actual brain where it's either on or off; good for deep neural networks] function)
    - Activation functions squish the range of the weighted sum to a range between 0 and 1
    - Therefore, activation = a measure of how positive the weighted sum is (because the range of the weighted sum is always somewhere between 0 and 1 once fed into the activation function)

So the weights tell you what the neuron is picking up on, the bias tells you how high does weighted sum need to be before neuron becomes meaningfully active.

When a neural network learns, you're getting computer to find the right weights and biases to solve the problem.

Knowing what the weights and biases are and what they're doing is important for experimenting with how to change the structure of the neural network to improve. 

All activations are organized into a vector and all weights are organized into a matrix and each number in the vector resulting from the multiplication = 1 of the activations

### How does it learn appropriate weights and biases?


### Backpropagation