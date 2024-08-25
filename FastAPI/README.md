# Lung Disease Prediction with FastAPI Web Interface

![image](https://github.com/user-attachments/assets/19af4d20-6659-4f33-9737-126bb160e3ce)

<a href="https://drive.google.com/file/d/1QkqWsQzviPr1LoavJmmV0QSgsQCeDNfX/view?usp=sharing">Watch the video</a>


This section provides instructions to set up and run the FastAPI application for lung disease prediction. The FastAPI app features a web-based user interface where users can upload lung images and receive predictions using a trained model.

## Project Structure

- **`FastAPI/`**: This directory contains the FastAPI application and the web interface.
  - **`app.py`**: The main FastAPI application file that serves the prediction API and web UI.
  - **`static/`**: Contains static files such as CSS, JavaScript, The HTML file for the web interface where users can upload images.
  - **`serving_model_dir/`**: Directory to store the trained model used for predictions.
  - **`requirements.txt`**: Contains all the Python dependencies required to run the FastAPI app.



## Steps to Set Up and Run the FastAPI Application

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone [https://github.com/your-repo.git](https://github.com/sawanjr/LUNG_CANCER-DETECTION-TFX.git)
```

### 2. Navigate to the FastAPI Directory

Move to the `FastAPI` directory, which contains the application files:

```bash
cd ./FastAPI
```

### 3. Add the `serving_model_dir`

Ensure that the `serving_model_dir` (which contains the saved TensorFlow model) is placed inside the `./FastAPI` directory. This folder should have the model that will be used by the app for making predictions.

Example:
```bash
./FastAPI/serving_model_dir
```

The FastAPI app will load the model from this directory to serve predictions via the API.

### 4. Create a Virtual Environment

It is recommended to create a virtual environment to manage dependencies:

On Windows:
```bash
python -m venv venv
```

On MacOS/Linux:
```bash
python3 -m venv venv
```

### 5. Activate the Virtual Environment

Activate the virtual environment:

On Windows:
```bash
.\venv\Scripts\activate
```

On MacOS/Linux:
```bash
source venv/bin/activate
```

### 6. Install Required Packages

Once the virtual environment is activated, install the necessary dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 7. Run the FastAPI Application

To start the FastAPI application, run the following command:

```bash
uvicorn app:app --reload
```

The application will start running locally. Open your browser and navigate to `http://127.0.0.1:8000` to access the web interface for uploading images and making predictions.

---

## FastAPI Application Details

### 1. `app.py`

The `app.py` file is the main FastAPI application script. It contains the following components:

- **API Endpoint for Prediction (`/predict/`)**:
  This endpoint accepts an image file (uploaded by the user) and returns a prediction after processing the image with the model loaded from `serving_model_dir`.

- **HTML Rendering**:
  It renders the `index.html` template to serve the frontend for image upload and prediction.

Here’s a brief overview of the structure:

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf

app = FastAPI()

# Loading the model
model = tf.keras.models.load_model('./serving_model_dir')

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Code to handle file upload, preprocessing, and prediction goes here
    ...
```

### 2. `index.html`

The `index.html` file is the main user interface for the application. It allows users to:

- Upload lung images for prediction.
- Display prediction results (whether the lung disease is benign or malignant).
- Reset the interface to upload more images or clear the gallery.

#### Key Features:
- **File Upload Section**: Allows the user to upload an image file.
- **Prediction Button**: Sends the uploaded image to the backend for prediction.
- **Reset Button**: Clears the gallery and allows new images to be uploaded.

The HTML structure includes:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Disease Prediction</title>
    <style>
        /* CSS for layout, buttons, and background goes here */
    </style>
</head>
<body>
    <header>
        <h1>Lung Disease Prediction</h1>
    </header>

    <div class="container">
        <div class="upload-section">
            <label for="imageFile">Choose an Image</label>
            <input type="file" id="imageFile" accept="image/*">
            <button id="upload-btn">Upload and Predict</button>
        </div>
        <div class="gallery" id="gallery"></div>
        <div class="message">Add another image or reset to clean</div>
        <button id="reset-btn" class="reset-btn">Reset</button>
    </div>

    <footer>
        <p>&copy; 2024 Lung Disease Prediction. All Rights Reserved. | <a href="https://github.com/your-repo" target="_blank">GitHub</a> | <a href="https://www.linkedin.com/in/your-profile" target="_blank">LinkedIn</a></p>
    </footer>

    <script>
        // JavaScript for handling file uploads and displaying predictions
    </script>
</body>
</html>
```

---

## Additional Notes

- **Prediction Model**: Ensure that the TensorFlow model placed in `serving_model_dir` is compatible with the FastAPI app.
- **Reset Functionality**: The app allows users to upload multiple images but ensures a limit of 5 images. Users must reset the gallery before uploading more images.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

