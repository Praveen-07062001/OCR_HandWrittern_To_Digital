
OCR System for Handwritten Text Recognition
This repository provides a complete pipeline for performing OCR (Optical Character Recognition) on PDF documents using deep learning. The pipeline includes PDF-to-image conversion, text extraction using Tesseract OCR, and the training of a deep learning model for recognizing handwritten text. The model uses a combination of CNN (Convolutional Neural Networks) and LSTM (Long Short-Term Memory) with CTC (Connectionist Temporal Classification) loss for sequence-based prediction tasks.

Requirements
Before running the code, make sure you have the following dependencies installed:

Python 3.x
TensorFlow: pip install tensorflow
NumPy: pip install numpy
Pillow: pip install pillow
PyMuPDF (fitz): pip install pymupdf
Pytesseract: pip install pytesseract
Editdistance: pip install editdistance
Scikit-learn: pip install scikit-learn
You also need to install Tesseract OCR from here and set the path to the executable in the code.

Overview of the Pipeline
The pipeline consists of the following major steps:

1. PDF to Image Conversion
PDF files are first converted into images using the PyMuPDF library. Each page of the PDF document is saved as a separate PNG file.

2. OCR with Tesseract
Once the PDF is converted into images, Tesseract OCR is used to extract the text from the images. The extracted text is saved in a text file for further analysis.

3. Deep Learning Model for OCR
A custom deep learning model based on a CNN + LSTM architecture is trained to recognize text from images. The model uses CTC loss to handle sequence-based output (e.g., recognizing text in varying lengths).

4. Model Evaluation
The model can be evaluated using Character Error Rate (CER) and Word Error Rate (WER). These metrics help assess the accuracy of the OCR model in terms of text recognition.

File Structure
/OCR_Pipeline/
|-- /data_subset_pdf/
|   |-- input.pdf                 # Your input PDF file
|   |-- output_images2/           # Directory for converted images from the PDF
|   |-- output_text/              # Directory for the extracted text
|-- model.py                      # Contains the model architecture and training functions
|-- ocr_pipeline.py               # Main pipeline code that runs the OCR process
|-- README.md                     # Documentation

Step-by-Step Instructions

1. Convert PDF to Images
The function convert_pdf_to_images() converts each page of the given PDF file into an image file. It takes the PDF path and an output directory as inputs.
convert_pdf_to_images(pdf_path, output_image_dir)
Each page of the PDF will be saved as a PNG image in the specified output directory.

2. Extract Text from Images using Tesseract OCR
The function extract_text_from_images() uses Tesseract OCR to extract text from the images generated in the previous step. It saves the extracted text into a .txt file.
extract_text_from_images(output_image_dir, output_txt_path)

4. Train the OCR Model
The OCR model is built using a combination of CNN, LSTM, and a custom CTC loss function. To train the model, you need labeled data in the form of images and corresponding text. The training process uses early stopping and model checkpointing to avoid overfitting.
train_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=10)
X_train and y_train: Training images and corresponding labels.
X_val and y_val: Validation images and corresponding labels.
batch_size and epochs: Define the training batch size and the number of epochs.

4. Evaluate the Model
The model can be evaluated on test data using CER and WER metrics to assess its performance. The function evaluate_model() provides these evaluations.
evaluate_model(model, X_test, y_test, label_decoder)
X_test and y_test: Test images and corresponding labels.
label_decoder: A function that decodes the model's predictions back into text.

6. Label Decoding (CTC)
The decode_labels() function is used to decode the output of the model (which is in the form of probabilities) into readable text.
decode_labels(output, char_list)

Code Explanation
1. CTC Loss Function

def ctc_loss(y_true, y_pred):
    input_length = K.cast(K.shape(y_pred)[1], 'int64')
    label_length = K.cast(K.shape(y_true)[1], 'int64')
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
The CTC loss function is used to train the model for sequence prediction. It is suitable for OCR tasks where the number of output characters can vary.

2. Model Architecture

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Reshape((63, 15 * 32)),
    LSTM(128, return_sequences=True),
    Dense(num_classes, activation='softmax')
])


The model uses convolutional layers to extract features from images, followed by an LSTM layer to capture temporal dependencies, and ends with a Dense layer to predict the characters.

3. CER and WER Calculation

def calculate_cer(predicted, actual):
    return sum(editdistance.eval(p, a) for p, a in zip(predicted, actual)) / sum(len(a) for a in actual)

def calculate_wer(predicted, actual):
    return sum(editdistance.eval(p.split(), a.split()) for p, a in zip(predicted, actual)) / sum(len(a.split()) for a in actual)
These functions calculate the Character Error Rate (CER) and Word Error Rate (WER) by comparing the predicted text with the actual ground truth.

How to Run the Pipeline

Place your input PDF file in the data_subset_pdf directory.
Run the ocr_pipeline.py script to convert the PDF to images and extract text using Tesseract OCR.
python ocr_pipeline.py
If you have labeled training data, you can train the OCR model using the train_model() function.
Evaluate the model using the evaluate_model() function to calculate CER and WER.

Example Output
After running the pipeline, you will have:
A set of PNG images representing the pages of the PDF.
A text file with the OCR-extracted text from the images.
Evaluation metrics such as CER and WER, printed to the console.
Troubleshooting
Ensure that Tesseract is correctly installed and its path is set in the code (pytesseract.pytesseract.tesseract_cmd).
Make sure the PDF file is not password-protected, as it may cause issues during the conversion process.# OCR_HandWrittern_To_Digital
