
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import os

app = FastAPI()

# Mount the 'static' directory to serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your TensorFlow model
model_dir = "serving_model_dir/1724478962/"
model = tf.saved_model.load(export_dir=model_dir)
predict_fn = model.signatures["serving_default"]

def serialize_example(image_content):
    """Serialize the image as a TFRecord example."""
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_content])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))  # Dummy label
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def make_prediction(image_content):
    """Make a prediction from the serialized image content."""
    tfrecord_data = serialize_example(image_content)
    prediction = predict_fn(tf.constant([tfrecord_data]))  # Call the model for prediction

    # Adjust this based on the actual structure of your model's output
    prediction_output = prediction['outputs'].numpy()  # Use 'outputs' instead of 'output'
    return prediction_output

@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve the index.html file for the UI."""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Log received file
        print(f"Received file: {file.filename}")

        # Read the uploaded file content
        image_bytes = await file.read()

        # Log the size of the image
        print(f"Image size: {len(image_bytes)} bytes")

        # Make a prediction
        prediction = make_prediction(image_bytes)

        # Log the prediction
        print(f"Prediction: {prediction}")

        # Extract the prediction result
        predicted_class = np.argmax(prediction)  # Use np.argmax on the output array

        return JSONResponse(content={"predicted_class": int(predicted_class)})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
