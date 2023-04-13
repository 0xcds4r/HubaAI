import torch
import torch.nn as nn
import torch.optim as optim
import json

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def train(net, input_data, target_data, num_epochs=1000, lr=0.001, weight_decay=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = net(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

with open('training_data.json') as f:
    training_data = json.load(f)

input_data = []
target_data = []
for example in training_data:
    input_data.append(example['input'])
    target_data.append(example['output'])

max_length = max(len(example) for example in input_data)
input_size = max_length * 26
output_size = max_length * 26
hidden_size = 1000
input_data = torch.zeros((len(input_data), input_size), dtype=torch.float32)
target_data = torch.zeros((len(target_data), output_size), dtype=torch.float32)
for i, example in enumerate(training_data):
    for j, char in enumerate(example['input']):
        index = j * 26 + ord(char.lower()) - ord('a')
        if index < input_size:
            input_data[i][index] = 1.0
    for j, char in enumerate(example['output']):
        index = j * 26 + ord(char.lower()) - ord('a')
        if index < output_size:
            target_data[i][index] = 1.0

net = Net(input_size, output_size, hidden_size)
train(net, input_data, target_data, num_epochs=700, lr=0.0001, weight_decay=0.00001)

while True:
    user_input = input('You: ')
    input_tensor = torch.zeros((1, input_size), dtype=torch.float32)
    for j, char in enumerate(user_input):
        index = j * 26 + ord(char.lower()) - ord('a')
        input_tensor[0][index] = 1.0

    expected_output = input('What is the expected answer? (press enter for skip)')
    
    if len(expected_output) > 0:
        expected_output_tensor = torch.zeros((1, output_size), dtype=torch.float32)
        for j, char in enumerate(expected_output):
            index = j * 26 + ord(char.lower()) - ord('a')
            if index < output_size:
                expected_output_tensor[0][index] = 1.0
        
        # Train the neural network on the new data
        train(net, input_tensor, expected_output_tensor, num_epochs=700, lr=0.0001, weight_decay=0.00001)
    
    # Use the trained neural network to answer the user's input
    output_tensor = net(input_tensor)
    output = ''
    word = ''
    for j in range(max_length):
        for k in range(26):
            index = j * 26 + k
            if output_tensor[0][index] > 0.5:
                word += chr(k + ord('a'))
            if output_tensor[0][index] > 0.5 or (j == max_length-1 and word != ''):
                output += word
                word = ''
    print('HubaAI:', output.strip())
    
    if len(expected_output) > 0:
        # Prompt the user to add the new input and expected output to the training data
        add_data = input('Do you want to add this data to the training set? (y/n) ')
        if add_data.lower() == 'y':
            training_data.append({'input': user_input, 'output': expected_output})
            with open('training_data.json', 'w') as f:
                json.dump(training_data, f)