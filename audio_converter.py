import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def audio_to_melspectrogram(file_path, display=True):
    """
    Converts an audio file to a mel spectrogram.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: The mel spectrogram.
    """

    y, sr = librosa.load(file_path)  # Load audio
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)  # Convert to log scale (dB)

    if display:
        # Optional: Display the mel spectrogram
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.show()

    return log_S

def melspectrogram_to_audio(mel_spectrogram):
    """
    Converts a mel spectrogram back to an audio file.

    Args:
        mel_spectrogram (numpy.ndarray): The mel spectrogram.

    Returns:
        numpy.ndarray: The reconstructed audio signal.
    """

    # Reconstruct waveform from spectrogram (assumes log scale)
    S_db = mel_spectrogram
    S = librosa.db_to_power(S_db)  
    audio = librosa.feature.inverse.mel_to_audio(S, fmax=8000)

    return audio

if __name__ == "__main__":
    # Example Usage
    audio_file = 'pronunciations/repine.mp3'  # Replace with your audio file path

    # Convert audio to mel spectrogram
    mel_spec = audio_to_melspectrogram(audio_file)

    # Convert mel spectrogram back to audio
    reconstructed_audio = melspectrogram_to_audio(mel_spec)

    # Save the reconstructed audio
    #librosa.output.write_wav('reconstructed_audio.wav', reconstructed_audio, sr=22050) 
    sf.write('repine.wav', reconstructed_audio, 48000)
