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

### How does it learn appropriate weights and biases? Gradient descent

We want an algorithm where we can show the neural network a bunch of training data and it'll adjust the weights and biases to improve performance on training data, and then give it testing data it's never seen before to see how well it perform on those (if model is accurate and generalizable, then we've succeeded)

Gradient descent: find minimum of a function.
Weights = representing the strengths of connections between neurons
Biases = indication of whether neuron tends to be active or inactive

1. Initialize weights and biases randomly
2. Define cost function, a way of telling the computer they did wrong. Add up squares of differences between each of the trash activations and the value you want them to have (e.g. 0 for all neurons except 1 specific one, the correct one that you want to have a value of 1) = cost of single training example. Small cost when correct, large when network doesn't know what it's doing.

Consider average cost of all training data -> measure of how bad the network is

Neural network components:
Input: training data
Output: some prediction based on the training data
Parameters: weights/biases

Cost function is a layer on top of the neural network components:
Input: weights/biases
Output: 1 number (the cost)
Parameters: the way it's defined depends on the network's behavior over all of its many training examples

Wanna tell it how to change weights/biases to make it better as well. What input to the cost function (weights/biases) minimizes the result of the cost function (the cost)? Depending on the function, finding the minimum cannot feasibly be done explicitly. A more flexible tactic is to start at any point, and figure out which direction to step to make the output of the function lower (i.e. if slope is positive, shift to left. if slope is negative, shift to right.) Doing this repeatedly -> approach some local minimum of the function. No guarantee that the local minimum you end up at will be the lowest possible cost (i.e. the absolute minimum). When step size is proportional to slope, when slope gets closer to minimum, steps get smaller and smaller, preventing overshooting.

#### What is gradient descent?
If input space (weights/biases) is an xy-plane and the cost function is a surface above it, you would ask "which direction decreases C(x,y) most quickly (instead of asking about the slope)?" Multivariable calc. tells us the gradient of the function gives us the direction of steepest increase. Negative of the gradient tells us the direction of steepest decrease. Length of gradient vector is an indication for how steep the steepest slope is.

Algorithm for minimizing cost function is to compute the gradient direction. Take a small step in negative gradient direction. Repeat that over and over.

Putting vector of all the weights and biases into negative gradient of the cost function -> vector containing numbers telling you how much to nudge the weights and biases to most rapidly decrase the cost function.

Negative gradient of the cost function is the direction that tells you w/c nudges to weights and biases is gonna cause the most rapid decrease to cost function. 

Takes weights/biases from random -> actual decision

cost function must have smooth output so we can find minimum w/c is why artificial neurons have continuously ranging activations rather than being binarily active/inactive like biological neurons

Gradient descent = repeatedly nudging input of cost function by some multiple of negative gradient

Sign of nudge tells us whether weight/bias should be nudged up (+) or down (-)
Magnitude of nudge tells us whether weight/bias should be nudged a little, somewhat, or a lot (i.e. which changes matter more b/c big changes matter more/have bigger effect than small changes; some connections matter more for the training data; tells how sensitive ost functino is to each weight/bias: cost of function is more sensitive to changes in weight with the larger magnitude nudge b/c that change has a magnitude-times greater effect)

### Limitations of MLP

Each layer is supposed to be breaking down the problem into subproblems such that each layer handles one subproblem, but that's not actually what happens necessarily. The patterns that the weights lead to may not be comprehensible at all, may seem very random, patterns very loose, yet it can recognize input data... because of this, when you input something random, it won't do something smart and be unsure (e.g. activating all neurons in last layer evenly or not at all), but instead, it confidently makes a prediction, even if it's entirely wrong. It can't create, it can only recognize b/c of tightl ocnstrained training setup. (from its POV, its entire universe is the training data. its cost function never gave it any other incentive other than to be completely confident in its decisions.)

Do neural networks just memorize or does it actually learn things about the data like does its result correspond to some aspect of/pattern in the data or is it just memorizing this feature goes with this target/label? They're doing smth a little bit smarter than just memorizing. Random dataset -> struggling to find local minimum. Structured dataset (correct labels) -> drop very fast to find local minimum, so easier and faster to learn structured data

Local minima that networks tend to learn are of roughly equal quality so if dataset is structured, should be able to find minima much more easily

### Backpropagation

Algo for computing gradient efficiently = backpropagation

