# Micrograd-Neural-Network-Implementation-Micrograd-Neural-Network-Implementation
This repository contains an implementation of a neural network framework inspired by the MicroGrad library, designed for educational purposes. The framework includes fundamental components such as Value, Neuron, Layer, and MLP classes, enabling the construction, forward pass, and backpropagation of simple neural networks.
# Micrograd-Neural-Network-Implementation-Micrograd-Neural-Network-Implementation
This repository contains an implementation of a neural network framework inspired by the MicroGrad library, designed for educational purposes. The framework includes fundamental components such as Value, Neuron, Layer, and MLP classes, enabling the construction, forward pass, and backpropagation of simple neural networks.

# Acknowledgment

## MicroGrad Neural Network Implementation

This code is an implementation of a neural network using the principles and components inspired by MicroGrad, a lightweight and educational gradient-based library.

### Overview

The code defines a simple neural network framework, including fundamental classes such as `Value`, `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron). These classes together enable the construction, forward pass, and backward propagation for neural networks. Key functionalities include:

- **Value Class**: Represents a scalar value with support for basic arithmetic operations, gradient tracking, and backpropagation.
- **Neuron Class**: Represents a single neuron in a neural network layer, capable of applying non-linear activation functions.
- **Layer Class**: Represents a layer of neurons.
- **MLP Class**: Represents a multi-layer perceptron, composed of multiple layers.

### Acknowledgments

The design and implementation of this code are heavily inspired by the MicroGrad library, originally created by Andrej Karpathy. The educational nature of MicroGrad provides an excellent foundation for understanding the core concepts of neural networks and automatic differentiation.

### References

- Karpathy, A. MicroGrad: https://github.com/karpathy/micrograd

### Usage

This section demonstrates how to create and train a multi-layer perceptron (MLP) using the provided classes. In this example, we use a uniform learning rate and epoch configuration to train the MLP on a synthetic dataset.

1. **Define Hyperparameters**:
    ```python
    learning_rate = 0.01
    epochs = 300
    ```

2. **Initialize the Network**:
    ```python
    N = MLP(3, [4, 4, 1])
    ```

3. **Prepare the Dataset**:
    ```python
    X = [
        [2.0, 3.0, -1.0], 
        [3.0, -1.0, 0.5], 
        [0.5, 1.0, 1.0], 
        [1.0, 1.0, -1.0],
        [-2.0, -3.0, 1.0],
        [-3.0, 1.0, -0.5],
        [-0.5, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [2.5, 3.5, -0.5],
        [3.5, -0.5, 0.0],
        [0.0, 1.5, 0.5],
        [1.5, 1.5, -1.5],
        [-2.5, -3.5, 1.5],
        [-3.5, 1.5, -0.75],
        [-0.75, -1.5, -1.5],
        [-1.5, -1.5, 1.5],
    ]
    Y = [
        1.0, -1.0, -1.0, 1.0,
        1.0, -1.0, -1.0, 1.0,
        1.0, -1.0, -1.0, 1.0,
        1.0, -1.0, -1.0, 1.0,
    ]
    ```

4. **Define the Mean Squared Error (MSE) Loss Function**:
    ```python
    def mse_loss(pred, target):
        return sum((p - Value(t))**2 for p, t in zip(pred, target)) / len(target)
    ```

5. **Train the Network**:
    ```python
    for epoch in range(epochs):
        y_pred = [N(x) for x in X]
        loss = mse_loss(y_pred, Y)
        N.zero_grad()
        loss.backward()
        for p in N.parameters():
            p.data -= learning_rate * p.grad
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")
    ```

6. **Make Predictions and Evaluate Accuracy**:
    ```python
    def threshold(pred, threshold=0.0):
        return 1.0 if pred.data > threshold else -1.0

    def Accuracy(N, X, Y):
        correct = 0
        for x, y in zip(X, Y):
            pred = N(x)
            pred_label = threshold(pred)
            if pred_label == y:
                correct += 1
        accuracy = correct / len(Y)
        return accuracy

    print("Prediction:")
    for x in X:
        pred = N(x)
        print(f"Input: {x} => Prediction: {pred.data}")

    accuracy = Accuracy(N, X, Y)
    print(f"Accuracy: {accuracy * 100}%")
    ```

### Running the Code

To run the above example, create a new Python script (e.g., `train_mlp.py`) and include the provided code. Execute the script in your terminal:

```bash
python train_mlp.py

