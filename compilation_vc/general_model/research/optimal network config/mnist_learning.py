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
    [512]  # Single Layer
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
            layers.append(nn.Linear(input_size if i == 0 else layer_sizes[i - 1], layer_size))
            layers.append(nn.ReLU())
        if layer_sizes:
            layers.append(nn.Linear(layer_sizes[-1], num_classes))
        else:
            layers.append(nn.Linear(input_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Initialize plotting
plt.ion()
fig, axs = plt.subplots(4, len(model_configs), figsize=(10, 10))  # Including an extra row for cumulative data samples
cumulative_fig, cumulative_ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
cumulative_ax.set_title('Average Samples Needed to Reach 98% Accuracy Across Configurations')
cumulative_ax.set_xlabel('Network Configuration')
cumulative_ax.set_ylabel('Average Data Samples')

# Training and evaluation function
def train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, config_idx, run, ax_handles, accuracy_lists, samples_needed_lists):
    model.train()
    training_losses = []
    test_accuracies = []
    samples_processed = 0
    samples_processed_history = []  # Track samples processed after each epoch
    epoch = 0

    try:
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
            accuracy_lists[run].append(accuracy)
            samples_processed_history.append(samples_processed)  # Update samples processed history

            # Update plots for the current configuration
            for i, handle in enumerate(ax_handles):
                if i == 0:
                    handle.plot(training_losses, label=f'Run {run + 1}' if epoch == 0 else "", linestyle='--')
                    handle.set_title(f'Loss: {config_to_str(config)}')
                elif i == 1:
                    handle.plot(test_accuracies, label=f'Run {run + 1}' if epoch == 0 else "", linestyle='--', color='orange')
                    handle.set_title(f'Accuracy: {config_to_str(config)}')
                elif i == 2:  # Update to plot a bar graph for samples processed
                    if samples_processed_history:  # Only draw bars if there are samples processed
                        handle.bar(run + 1, samples_processed_history[-1], color='red')
                    handle.set_title(f'Data Samples: {config_to_str(config)}')
                    handle.set_xlabel('Run')
                    handle.set_ylabel('Cumulative Samples Processed')

            fig.canvas.draw()
            fig.canvas.flush_events()

            if accuracy >= 98 or epoch >= 50:
                break

            epoch += 1

        samples_needed_lists.append(samples_processed)
        return samples_processed  # Return samples processed when training completes successfully

    except Exception as e:
        print(f"Error during training: {e}")
        return None  # Return None when training fails

# Helper function to convert config to string
def config_to_str(config):
    return ', '.join(map(str, config)) if config else 'Direct Input to Output'

def calculate_overall_mean_accuracy(accuracy_list):
    max_length = max(len(acc) for acc in accuracy_list)
    padded_accuracies = np.array([acc + [np.nan] * (max_length - len(acc)) for acc in accuracy_list])
    return np.nanmean(padded_accuracies, axis=0)

# Loop to train and evaluate each model configuration
samples_needed_list = [[] for _ in range(len(model_configs))]
accuracy_lists = [[[] for _ in range(10)] for _ in range(len(model_configs))]
config_labels = []

for config_idx, config in enumerate(model_configs):
    for run in range(10):
        print(f'\nTraining with network configuration: {config}, Run: {run + 1}')
        model = LinearNet(784, config, 10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        ax_handles = [axs[i, config_idx] for i in range(4)]
        samples_needed = train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, config_idx, run, ax_handles, accuracy_lists[config_idx], samples_needed_list[config_idx])
        if samples_needed is not None:
            samples_needed_list[config_idx].append(samples_needed)

    # After all runs for a configuration, calculate and draw the average line for samples needed
    if samples_needed_list[config_idx]:  # Ensure there are samples to calculate an average from
        average_samples_needed = np.mean(samples_needed_list[config_idx])
        # Draw a horizontal line on the third plot (index 2) for the current configuration
        axs[2, config_idx].axhline(y=average_samples_needed, color='green', linestyle='-')

    overall_mean_accuracy = calculate_overall_mean_accuracy(accuracy_lists[config_idx])
    axs[3, config_idx].plot(overall_mean_accuracy, linestyle='-', color='blue')
    axs[3, config_idx].set_title(f'Avg. Accuracy: {config_to_str(config)}')

    config_labels.append(config_to_str(config))

    cumulative_ax.clear()
    cumulative_ax.set_title('Average Samples Needed to Reach 98% Accuracy Across Configurations')
    cumulative_ax.set_xlabel('Network Configuration')
    cumulative_ax.set_ylabel('Average Data Samples')

    valid_config_labels = [config_labels[i] for i, samples in enumerate(samples_needed_list) if samples]
    valid_average_samples_needed = [np.mean(samples) for samples in samples_needed_list if samples]

    cumulative_ax.bar(valid_config_labels, valid_average_samples_needed, color='blue')
    cumulative_ax.set_xticks(range(len(valid_config_labels)))
    cumulative_ax.set_xticklabels(valid_config_labels, rotation=45, ha="right")

    cumulative_fig.canvas.draw()
    cumulative_fig.canvas.flush_events()


plt.ioff()
plt.show()
