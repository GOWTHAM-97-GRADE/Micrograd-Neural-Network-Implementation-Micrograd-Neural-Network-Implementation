from value import Value
from nn import MLP

learning_rate = 0.01
epochs = 300

# Initialize the model
N = MLP(3, [4, 4, 1])

# Training data
X = [
    [2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0],
    [-2.0, -3.0, 1.0], [-3.0, 1.0, -0.5], [-0.5, -1.0, -1.0], [-1.0, -1.0, 1.0],
    [2.5, 3.5, -0.5], [3.5, -0.5, 0.0], [0.0, 1.5, 0.5], [1.5, 1.5, -1.5],
    [-2.5, -3.5, 1.5], [-3.5, 1.5, -0.75], [-0.75, -1.5, -1.5], [-1.5, -1.5, 1.5]
]
Y = [
    1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, 1.0
]

# Mean Squared Error Loss
def mse_loss(pred, target):
    return sum((p - Value(t)) ** 2 for p, t in zip(pred, target)) / len(target)

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = [N(x) for x in X]
    loss = mse_loss(y_pred, Y)
    
    # Zero gradients
    N.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update parameters using gradient descent
    for p in N.parameters():
        p.data -= learning_rate * p.grad

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# Evaluate accuracy
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
