import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from src.models.specrnet import FrontendSpecRNet  # Adjust import as necessary
from src.frontends import get_frontend  # Ensure you have the right path

# Constants
SAMPLING_RATE = 16000
VALIDATION_DIR = '../dataset/validate'  # Update this path as needed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_FEATURES = 128  # Set this to the number of LFCC features you expect

# Load your model
def load_model(checkpoint_path):
    model = FrontendSpecRNet(input_channels=1, device=DEVICE, frontend_algorithm=["lfcc"])
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)  # Ensure the model is on the correct device
    model.eval()  # Set to evaluation mode
    return model

# Load validation dataset
def load_validation_data():
    audio_files = []
    for file in os.listdir(VALIDATION_DIR):
        if file.endswith('.wav'):
            audio_files.append(os.path.join(VALIDATION_DIR, file))
    return audio_files

# Compute Grad-CAM and get dominant features
def compute_dominant_features(model, audio_path, target_label):
    audio, sample_rate = torchaudio.load(audio_path)
    audio = audio.to(DEVICE)  # Move audio tensor to the correct device
    audio.requires_grad_()  # Enable gradient tracking on the audio input

    # Forward pass
    frontend_output = model._compute_frontend(audio)
    
    # Switch to training mode for backward pass
    model.train()  # Set model to training mode temporarily
    output = model._compute_embedding(frontend_output)

    # Calculate probabilities
    output_probs = torch.sigmoid(output)

    # Convert target label to a tensor with the same shape
    target_tensor = torch.tensor([[target_label]], dtype=torch.float32).to(DEVICE)  # Shape: [1, 1]

    # Compute the loss
    loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
    loss = loss_fn(output_probs, target_tensor)  # Compute the loss

    # Compute gradients
    loss.backward()  # Backpropagate the loss to compute gradients

    # Get the gradients
    gradients = audio.grad  # Get gradients with respect to input audio

    # Calculate feature importance
    feature_importance = gradients.mean(dim=1).detach().cpu().numpy()  # Mean across the channels
    feature_importance = np.abs(feature_importance)  # Take absolute value for importance

    return feature_importance

# Visualize top 10 features
def visualize_top_10_features(aggregated_importance):
    # Get the indices of the top 10 features
    top_indices = np.argsort(aggregated_importance)[-10:][::-1]  # Get the indices of the top 10 features
    top_values = aggregated_importance[top_indices]  # Get the top 10 values

    plt.figure(figsize=(10, 4))
    plt.bar(range(10), top_values, color='blue')
    plt.xticks(range(10), top_indices, rotation=45)
    plt.title('Top 10 Dominant LFCC Features Across All Audio Files')
    plt.xlabel('Feature Index')
    plt.ylabel('Average Importance')
    plt.grid()
    plt.show()

def main():
    checkpoint_path = './trained_models/specrnet_balanced/ckpt.pth'  # Update with your file path
    model = load_model(checkpoint_path)
    audio_files = load_validation_data()

    labels = {'bonafide': 0, 'spoof': 1}  # Define your labels
    all_importances = []  # List to collect feature importances

    for audio_file in audio_files:
        label_key = 'bonafide' if 'bonafide' in audio_file else 'spoof'
        target_label = labels[label_key]

        feature_importance = compute_dominant_features(model, audio_file, target_label)
        
        # Check the length of the feature importance and print for debugging
        print(f"Feature importance shape for {audio_file}: {feature_importance.shape}")  # Debugging output
        
        all_importances.append(feature_importance)  # Collect the feature importances

    # Convert the list of importances to a NumPy array for aggregation
    # Use np.concatenate to handle different shapes
    all_importances = np.array([np.pad(fi, (0, MAX_FEATURES - len(fi)), 'constant') if len(fi) < MAX_FEATURES else fi for fi in all_importances])

    # Average the feature importances across all audio files
    aggregated_importance = np.mean(all_importances, axis=0)

    # Visualize only the top 10 features
    visualize_top_10_features(aggregated_importance)

if __name__ == "__main__":
    main()
