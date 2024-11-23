from flask import Flask, request, render_template
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('animal_classification_model.h5')

# Average frequencies for each class
average_frequencies = {
    "Dog": 10975.78125,
    "Rooster": 10975.78125,
    "Pig": 10975.78125,
    "Cow": 10975.78125,
    "Frog": 10975.78125
}

# Function to create a spectrogram from a .wav file
def generate_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Save as an RGB image
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, cmap='inferno')
    plt.axis('off')
    plt.tight_layout(pad=0)
    temp_file = "temp_spectrogram.png"
    plt.savefig(temp_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    return temp_file, y, sr


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join("uploads", "temp_audio.wav")
        file.save(file_path)

        # Generate spectrogram
        spectrogram_path, y, sr = generate_spectrogram(file_path)

        # Preprocess the spectrogram for the model
        img = tf.keras.preprocessing.image.load_img(spectrogram_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        
        class_names = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog']  # Class names
        predicted_species = class_names[predicted_class]

        # Calculate the mean frequency of the uploaded file
        mean_frequency = np.mean(np.abs(librosa.stft(y)))

        # Determine if the animal is in pain/distress
        avg_freq = average_frequencies.get(predicted_species, None)
        if avg_freq and (mean_frequency < avg_freq * 0.5 or mean_frequency > avg_freq * 1.5):
            distress_status = "The animal may be in pain/distress."
        else:
            distress_status = "The animal is not in pain/distress."

        return render_template(
            'index.html',
            predicted_species=predicted_species,
            confidence=confidence,
            distress_status=distress_status,
            # mean_frequency=mean_frequency
        )

if __name__ == '__main__':
    app.run(debug=True)

