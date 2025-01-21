pip install notebook
pip install tensorflow opencv-python pytesseract pdf2image numpy matplotlib scikit-learn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam
import fitz  # PyMuPDF
import os
import pytesseract
from PIL import Image
import os


# Define height, width, and number of output classes
height = 128
width = 32
num_classes = 36  # Assuming you have 36 output classes (a-z and 0-9)

# Define the model
model = Sequential()

# Convolutional layers to extract features
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Get the shape of the output after convolution and pooling
# It will be (batch_size, 63, 15, 32), and we want to reshape this to (batch_size, timesteps, features)
# Flatten the height and width dimensions into one
model.add(Reshape((63, 15 * 32)))  # Reshape to (timesteps, features), where features = height * channels

# Add LSTM layer
model.add(LSTM(128, return_sequences=True))

# Add Dense output layer
model.add(Dense(num_classes, activation='softmax'))  # Assuming num_classes are predefined

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()

# Replace this with the path to your input PDF file
pdf_path = r"D:\data_subset_pdf\input.pdf"  # <-- Modify this path

# Create an output directory to store the converted images
output_dir = r"D:\data_subset_pdf\output_images"  # <-- Modify this path if needed
os.makedirs(output_dir, exist_ok=True)  # This will create the directory if it doesn't exist

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Loop through all pages in the PDF and save each as an image
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)  # Load the current page
    pix = page.get_pixmap()  # Convert the page to a pixmap (image)

    # Define the output image file path
    output_image_path = os.path.join(output_dir, f"page_{page_num}.png")  # <-- Modify the filename as needed

    # Save the image to the output directory
    pix.save(output_image_path)

    print(f"Saved: {output_image_path}")  # Optional: prints the output path of each image



# Specify the path to the Tesseract executable (modify this if Tesseract is installed in a different directory)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Praveen\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Define the path where your images are stored
output_dir = r"D:\data_subset_pdf\output_images"  # Path to the folder containing PNG images

# Define the output path for your text file
text_output_path = r"D:\data_subset_pdf\output_text\extracted_text.txt"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

# Open the output text file for writing
with open(text_output_path, "w") as text_file:
    # Loop through all the images in the output_images directory
    for image_file in os.listdir(output_dir):
        # Only process PNG images
        if image_file.endswith(".png"):
            # Full path to the image
            image_path = os.path.join(output_dir, image_file)

            # Open the image using PIL
            image = Image.open(image_path)

            # Use pytesseract to extract text from the image
            extracted_text = pytesseract.image_to_string(image)

            # Write the extracted text to the text file
            text_file.write(f"Text from {image_file}:\n")
            text_file.write(extracted_text)
            print(extracted_text)
            text_file.write("\n\n")  # Add a newline between the texts from each page

            print(f"Processed {image_file} and saved text.")  # Optional: to show progress

print(f"Text extraction complete. The text has been saved to {text_output_path}.")
