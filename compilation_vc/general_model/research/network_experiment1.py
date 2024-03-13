import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Model configurations to test
model_configs = [
    [1024, 512],  # Baseline Configuration
    [1024, 512, 256],  # Increased Depth
    [1024, 512, 256, 128],  # Increased Depth with Fewer Nodes
    [1024, 768, 512, 256, 128],  # Even More Layers
    [512, 256, 128],  # Reduced Node Count per Layer
    [512, 256],  # Shallow Layers
    [512],  # Single Layer
    [124],  # Very Small Single Layer
    []  # Empty Configuration for baseline comparison
]

plt.rcParams.update({'font.size': 5})

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation
batch_size = 64
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
class LinearNet(nn.Module):
    def __init__(self, input_size, layer_sizes, num_classes):
        super(LinearNet, self).__init__()
        layers = [nn.Flatten()]
        for i, layer_size in enumerate(layer_sizes):
            layers.append(nn.Linear(input_size if i == 0 else layer_sizes[i-1], layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Initialize plotting
plt.ion()
fig, axs = plt.subplots(3, len(model_configs), figsize=(20, 15))
cumulative_fig, cumulative_ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
cumulative_ax.set_title('Samples Needed to Reach 98.5% Accuracy Across Configurations')
cumulative_ax.set_xlabel('Network Configuration')
cumulative_ax.set_ylabel('Data Samples')
samples_needed_to_converge = []
config_labels = []

# Training and evaluation function
def train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, config_idx):
    model.train()
    training_losses = []
    test_accuracies = []
    samples_processed = 0
    total_samples_list = []
    epoch = 0

    while True:
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            samples_processed += data.size(0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        training_losses.append(running_loss / len(train_loader))
        test_accuracies.append(accuracy)
        total_samples_list.append(samples_processed)

        # Update plots for the current configuration
        axs[0, config_idx].plot(training_losses, label='Training Loss' if epoch == 0 else "")
        axs[1, config_idx].plot(test_accuracies, label='Test Accuracy' if epoch == 0 else "", color='orange')
        axs[2, config_idx].plot(total_samples_list, label='Total Samples Processed' if epoch == 0 else "", color='red')

        if epoch == 0:
            for i in range(3):
                axs[i, config_idx].legend()
                axs[i, config_idx].set_title(f'Config: {model_configs[config_idx]}')
                axs[i, config_idx].set_xlabel('Epoch')
            axs[0, config_idx].set_ylabel('Loss')
            axs[1, config_idx].set_ylabel('Accuracy (%)')
            axs[2, config_idx].set_ylabel('Total Samples Processed')

        fig.canvas.draw()
        fig.canvas.flush_events()

        if accuracy >= 98.5:
            samples_needed_to_converge.append(samples_processed)
            config_labels.append(str(model_configs[config_idx]))
            cumulative_ax.clear()
            cumulative_ax.set_title('Samples Needed to Reach 98.5% Accuracy Across Configurations')
            cumulative_ax.set_xlabel('Network Configuration')
            cumulative_ax.set_ylabel('Data Samples')
            cumulative_ax.plot(config_labels, samples_needed_to_converge, 'o-', color='blue')
            cumulative_ax.set_xticklabels(config_labels, rotation='horizontal')  # Set labels horizontal
            cumulative_fig.canvas.draw()
            cumulative_fig.canvas.flush_events()
            break

        epoch += 1

# Loop to train and evaluate each model configuration
for config_idx, config in enumerate(model_configs):
    print(f'\nTraining with network configuration: {config}')
    model = LinearNet(784, config, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, config_idx)

plt.ioff()
plt.show()
