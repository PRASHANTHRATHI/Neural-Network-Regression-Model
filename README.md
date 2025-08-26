# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="935" height="678" alt="image" src="https://github.com/user-attachments/assets/9b1986ca-aa16-48c6-a9d5-f71e7277bdd3" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Prashanth K
### Register Number: 212223230152
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x



# Initialize the Model, Loss Function, and Optimizer

prashanth_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(prashanth_brain.parameters(),lr=0.001)


def train_model(prashanth_brain, X_train, y_train, criterion, optimizer, epochs=4000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(prashanth_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        prashanth_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

<img width="224" height="555" alt="image" src="https://github.com/user-attachments/assets/ddfab14a-597b-4343-978d-cf0761a2f144" />


## OUTPUT
### Training Loss Vs Iteration Plot

<img width="1463" height="162" alt="image" src="https://github.com/user-attachments/assets/3dc4dec6-e07d-4363-951c-076e79f2899a" />
<img width="925" height="570" alt="image" src="https://github.com/user-attachments/assets/0c842e77-65a7-4112-acd5-ee2fddabec06" />


### New Sample Data Prediction

<img width="1173" height="167" alt="image" src="https://github.com/user-attachments/assets/b35bf668-422f-4637-820a-e2c016ba217a" />


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
Include your result here
