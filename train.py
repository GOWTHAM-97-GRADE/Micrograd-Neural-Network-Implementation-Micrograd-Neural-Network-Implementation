
learning_rate=0.01
epochs=1000

N=MLP(3,[8,8,1])

X=[
    [2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0],
    [-2.0,-3.0,1.0],[-3.0,1.0,-0.5],[-0.5,-1.0,-1.0],[-1.0,-1.0,1.0],
    [2.5,3.5,-0.5],[3.5,-0.5,0.0],[0.0,1.5,0.5],[1.5,1.5,-1.5],
    [-2.5,-3.5,1.5],[-3.5,1.5,-0.75],[-0.75,-1.5,-1.5],[-1.5,-1.5,1.5]
]

Y=[1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0]


def mse_loss(pred,target):
    return sum((p-Value(t))**2 for p,t in zip(pred,target))/(2*len(Y))

for epoch in range(epochs):
    y_pred=[N(list(x)) for x in X]
    loss=mse_loss(y_pred,Y)
    N.zero_grad()
    loss.backward()
    for p in N.parameters():
        p.data-=learning_rate*p.grad
    if epoch%10==0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

def threshold(pred,threshold=0.0):
    return 1.0 if pred.data>threshold else -1.0

def Accuracy(N,X,Y):
    correct=0
    for x,y in zip(X,Y):
        pred=N(list(x))
        pred_label=threshold(pred)
        if pred_label==y:
            correct+=1
    accuracy=correct/len(Y)
    return accuracy

test_input=[
    [2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]
]

test_output=[N(list(x)) for x in test_input]

print("Test Outputs:")
for i,output in enumerate(test_output):
    print(f"Input {test_input[i]} -> Output {output.data}")

accuracy=Accuracy(N,X,Y)
print(f"Accuracy: {accuracy*100}%")
