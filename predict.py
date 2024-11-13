import torch
import torchaudio
from pathlib import Path 
from src.models.specrnet import FrontendSpecRNet
from src.frontends import prepare_lfcc_double_delta, prepare_mfcc_double_delta

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model configuration
input_channels = 1  # Single channel for audio
model_path = './ckpt.pth'  # Path to your trained model
frontend_algorithm = ["lfcc"]  # Or ["mfcc"], depending on what your model was trained with

# Load the model
model = FrontendSpecRNet(input_channels=input_channels, device=device, frontend_algorithm=frontend_algorithm).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"Model loaded from {model_path}")

def preprocess_audio(file_path):
    print(f"Processing file: {file_path}")
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)
        print(f"Resampled waveform shape: {waveform.shape}")

    # Move waveform to device (GPU or CPU)
    waveform = waveform.to(device)

    return waveform

def predict(file_path):
    # Preprocess the audio
    audio = preprocess_audio(file_path)

    print(f"Input to model shape: {audio.shape}")

    # Perform prediction
    with torch.no_grad():
        output = model(audio)
        prediction = torch.sigmoid(output).item()

    print(f"Raw prediction value: {prediction}")

    # Adjust the threshold for spoof classification
    threshold = 0.9  # Experiment with different thresholds, e.g., 0.6, 0.7, 0.8, etc.

    # Return the prediction as 'Spoof' or 'Bonafide' based on the adjusted threshold
    return "Bonafide" if prediction > threshold else "Spoof"

# Main function to predict the class of the audio file
if __name__ == "__main__":
    # Replace with your own audio file path
    # file_path = "./Bonafide_Dataset_OG/train/audio (1001)$$bonafide.wav"
    # file_path="./short-spoof.wav"
    # file_path="Bonafide_Dataset_OG/train/audio (13048)$$spoof.wav"
    # file_path="./bianca fake (2).mp3"
    # file_path="./andrea000.wav"
    file_path="short-bonafide.wav"

    try:
        prediction = predict(file_path)
        print(f"The audio is classified as: {prediction}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()