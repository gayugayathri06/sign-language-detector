import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"c:\Users\61910253\oneDrive\Desktop\sign_\sign language.keras")
# Create the GUI window
window = tk.Tk()
window.title("Sign_")

# Create a label to display the uploaded image
image_label = tk.Label(window, text="Upload an image or video to predict the sign language")
image_label.pack()

# Create a button to upload an image or video
upload_button = tk.Button(window, text="Upload", command=lambda: upload_file())
upload_button.pack()

# Create a label to display the predicted sign language
prediction_label = tk.Label(window, text="Predicted sign language:")
prediction_label.pack()

# Create a text box to display the predicted sign language
prediction_text = tk.Text(window, height=10, width=40)
prediction_text.pack()

def upload_file():
    # Open a file dialog to select an image or video file
    file_path = filedialog.askopenfilename()
    
    # Check if the file is an image or video
    if file_path.endswith('.jpg') or file_path.endswith('.png'):
        # Load the image using OpenCV
        image = cv2.imread(r"c:\Users\61910253\oneDrive\Desktop\sign_\opencv.ipynb")
        # Preprocess the image
        image = cv2.resize(image, (28, 28))
        image = image / 255.0
        
        # Make predictions using the model
        predictions = model.predict(image)
        
        # Get the predicted sign language
        predicted_sign_language = np.argmax(predictions)
        
        # Display the predicted sign language
        prediction_text.delete(1.0, tk.END)
        prediction_text.insert(tk.END, predicted_sign_language)
        
    elif file_path.endswith('.mp4') or file_path.endswith('.avi'):
        # Load the video using OpenCV
        video = cv2.VideoCapture(file_path)
        
        # Extract frames from the video
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        
        # Preprocess the frames
        frames = [cv2.resize(frame, (28, 28)) for frame in frames]
        frames = [frame / 255.0 for frame in frames]
        
        # Make predictions using the model
        predictions = []
        for frame in frames:
            predictions.append(model.predict(frame))
        
        # Get the predicted sign language
        predicted_sign_language = np.argmax(predictions)
        
        # Display the predicted sign language
        prediction_text.delete(1.0, tk.END)
        prediction_text.insert(tk.END, predicted_sign_language)

window.mainloop()