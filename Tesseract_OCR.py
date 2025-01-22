import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import editdistance  # For CER/WER calculation

# Define model input dimensions
height = 128
width = 32
num_classes = 36  # Adjust based on the dataset (e.g., a-z, 0-9, punctuation)

# Define CTC loss function
def ctc_loss(y_true, y_pred):
    input_length = K.cast(K.shape(y_pred)[1], 'int64')
    label_length = K.cast(K.shape(y_true)[1], 'int64')
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Define model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Reshape((63, 15 * 32)),  # Adjust according to your model's output
    LSTM(128, return_sequences=True),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(), loss=ctc_loss)

# Function to calculate CER
def calculate_cer(predicted, actual):
    return sum(editdistance.eval(p, a) for p, a in zip(predicted, actual)) / sum(len(a) for a in actual)

# Function to calculate WER
def calculate_wer(predicted, actual):
    return sum(editdistance.eval(p.split(), a.split()) for p, a in zip(predicted, actual)) / sum(len(a.split()) for a in actual)

# Train the model
def train_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, checkpoint])
    return history

# PDF to image conversion
def convert_pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        output_image_path = os.path.join(output_dir, f"page_{page_num}.png")
        pix.save(output_image_path)
        print(f"Saved: {output_image_path}")
    return output_dir

# OCR using Tesseract
def extract_text_from_images(image_dir, output_txt_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\BrindhaBrahmanandaBa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as text_file:
        for image_file in os.listdir(image_dir):
            if image_file.endswith(".png"):
                image_path = os.path.join(image_dir, image_file)
                image = Image.open(image_path)
                extracted_text = pytesseract.image_to_string(image)
                text_file.write(f"Text from {image_file}:")
                text_file.write(extracted_text + "\n\n")
                print(f"Processed {image_file}.")
    print(f"Text saved to {output_txt_path}.")

# Main pipeline
def main_pipeline(pdf_path, output_image_dir, output_txt_path):
    print("Converting PDF to images...")
    convert_pdf_to_images(pdf_path, output_image_dir)
    print("Extracting text from images...")
    extract_text_from_images(output_image_dir, output_txt_path)
    print("Pipeline complete.")

# Function to evaluate the model with CER and WER
def evaluate_model(model, X_test, y_test, label_decoder):
    """
    Evaluates the model using CER and WER.
    
    Parameters:
        model: The trained OCR model.
        X_test: Test dataset (images).
        y_test: Test labels (actual text).
        label_decoder: Function to decode model predictions into readable text.
    """
    # Get model predictions (ensure the shape matches expected: (batch_size, time_steps, num_classes))
    predictions = model.predict(X_test)
    
    # Decode predictions and ground truth to text
    predicted_texts = label_decoder(predictions)  # Use the decoder on the whole batch
    actual_texts = [label_decoder([label])[0] for label in y_test]  # Decode ground truth one at a time
    
    # Calculate CER and WER
    cer = calculate_cer(predicted_texts, actual_texts)
    wer = calculate_wer(predicted_texts, actual_texts)
    
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Word Error Rate (WER): {wer:.4f}")
    return cer, wer

# Label decoding function (example for CTC decoding)
def decode_labels(output, char_list):
    """
    Decodes the output of the model using CTC decoding.

    Parameters:
        output: The raw predictions from the model (shape: (batch_size, time_steps, num_classes)).
        char_list: List of characters used in training (alphabet, numbers, etc.).
    """
    decoded_texts = []
    for out in output:
        # Ensure the output is a 2D array (time_steps, num_classes)
        out_text = ''.join([char_list[idx] for idx in np.argmax(out, axis=1) if idx != -1])  # Ignore blank labels (-1)
        decoded_texts.append(out_text)
    return decoded_texts

# Paths
pdf_path = r"D:\data_subset_pdf\input.pdf"
output_image_dir = r"D:\data_subset_pdf\output_images2"
output_txt_path = r"D:\data_subset_pdf\output_text\extracted_text.txt"

# Example test data
X_test = np.random.rand(10, height, width, 1)  # 10 random grayscale images
y_test = np.random.randint(0, num_classes, size=(10, 5))  # 10 random labels (e.g., 5 characters per image)

# Ensure the labels are padded or truncated as necessary
y_test = pad_sequences(y_test, maxlen=5, padding='post', value=-1)

# Decode labels function
char_list = "abcdefghijklmnopqrstuvwxyz0123456789"  # Example character set

# Mock label decoder for predictions and ground truth
label_decoder = lambda output: decode_labels(output, char_list)

# Run pipeline
if __name__ == "__main__":
    main_pipeline(pdf_path, output_image_dir, output_txt_path)

    # Example usage for evaluation
    # Evaluate the model
    cer, wer = evaluate_model(model, X_test, y_test, label_decoder)
    print(f"Final Evaluation Results:\nCER: {cer:.4f}, WER: {wer:.4f}")
