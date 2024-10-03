import torch
import numpy as np
import random
from torch.utils.data import Subset
from torchvision import transforms
from voc_dataset import VOCDataset
from train_q2 import ResNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def get_device(use_cuda=True):
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, target, _ in dataloader:
            data = data.to(device)
            output = model.resnet.conv1(data)
            output = model.resnet.bn1(output)
            output = model.resnet.relu(output)
            output = model.resnet.maxpool(output)

            output = model.resnet.layer1(output)
            output = model.resnet.layer2(output)
            output = model.resnet.layer3(output)
            output = model.resnet.layer4(output)
            output = model.resnet.avgpool(output)
            output = torch.flatten(output, 1)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())

    features = np.vstack(features)
    labels = np.vstack(labels)
    return features, labels

def compute_mean_colors(labels, class_colors):
    # Compute mean color for images with multiple classes
    mean_colors = []
    for label in labels:
        active_classes = np.where(label == 1)[0]
        if len(active_classes) == 0:
            mean_colors.append((0, 0, 0))  # Black for no classes
        else:
            colors = np.array([class_colors[i][:3] for i in active_classes])  # Exclude alpha
            mean_color = colors.mean(axis=0)
            mean_colors.append(mean_color)
    return np.array(mean_colors)

def plot_tsne(tsne_results, colors, class_names, save_path='tsne_plot.png'):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=colors, s=10, alpha=0.6)

    # Create custom legend
    patches = [mpatches.Patch(color=class_colors[i], label=class_names[i]) for i in range(len(class_names))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('t-SNE Visualization of Feature Representations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")

    # Optionally, show the plot
    plt.show()

if __name__ == "__main__":
    set_seed(0)
    device = get_device(use_cuda=True)

    # Define class colors (Assign a unique color to each class)
    # Example: Using a color map
    cmap = plt.get_cmap('tab20')
    class_colors = [cmap(i) for i in range(len(VOCDataset.CLASS_NAMES))]

    # Load the dataset without passing 'transform'
    test_dataset = VOCDataset(split='test', size=224)  # Removed 'transform=transform'

    # Randomly select 1000 samples
    total_samples = len(test_dataset)
    if total_samples < 1000:
        raise ValueError(f"Test dataset has only {total_samples} samples.")
    random_indices = random.sample(range(total_samples), 1000)
    subset = Subset(test_dataset, random_indices)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model
    num_classes = len(VOCDataset.CLASS_NAMES)
    model = ResNet(num_classes=num_classes).to(device)

    # Load the trained model weights
    MODEL_PATH = './trained_model.pth'  # Updated path
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Extract features and labels
    features, labels = extract_features(model, test_loader, device)

    # Compute t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    print("t-SNE completed.")

    # Compute mean colors for multi-label
    colors = compute_mean_colors(labels, class_colors)

    # Plot and save the t-SNE visualization
    plot_tsne(tsne_results, colors, VOCDataset.CLASS_NAMES, save_path='tsne_plot.png')