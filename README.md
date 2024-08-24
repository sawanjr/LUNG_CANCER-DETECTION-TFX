# LUNG_CANCER-DETECTION
  ## End-to-End TFX Pipeline for Lung Image Analysis

## Project Overview

This project demonstrates an end-to-end TFX (TensorFlow Extended) pipeline designed for processing and analyzing lung images. The pipeline integrates several components of TFX to facilitate the ingestion, transformation, training, evaluation, and serving of a machine learning model specifically tuned for lung image classification.

![image](https://github.com/user-attachments/assets/7390ec54-4ef7-4d5f-8665-c86b62c3c614)


## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Setup and Installation](#setup-and-installation)
4. [Data Preparation](#data-preparation)
5. [Pipeline Components](#pipeline-components)
6. [Running the Pipeline](#running-the-pipeline)
7. [Model Serving](#model-serving)
8. [Results and Evaluation](#results-and-evaluation)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

This project aims to build a robust TFX pipeline for lung image classification, leveraging TensorFlow and associated tools to streamline the ML workflow from data ingestion to model serving. The pipeline handles end-to-end processing, including data validation, transformation, model training, and deployment.

## Pipeline Architecture

The TFX pipeline includes the following components:

- **ExampleGen**: Ingests raw image data.
- **ExampleValidator**: Performs data validation and quality checks.
- **SchemaGen**: Generates a schema for data validation.
- **Transform**: Transforms and preprocesses the data.
- **Trainer**: Trains a TensorFlow model on the processed data.
- **TensorflowModelAnalysis**: Analysis of the model.
- **Evaluator**: Evaluates the model performance.
- **Resolver**: Resolves the model and data artifacts.
- **Pusher**: Pushes the model to the serving infrastructure.

  ![image](https://github.com/user-attachments/assets/76c724c0-9d06-42ea-8439-6c14bdf62b96)


## Setup and Installation

### Prerequisites

- Python 3.10+
- TensorFlow 2.13.1
- TFX 1.14.0
- Docker (for containerized deployment)
- interactivePipeline (for experimentation)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sawanjr/LUNG_CANCER-DETECTION-TFX.git
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Docker (if applicable)**

   Follow Docker installation instructions for your operating system: https://docs.docker.com/get-docker/

## Data Preparation

The dataset consists of lung images that are used for training and evaluation. The data should be structured in the following format:

- **Raw Images**: Stored in a directory structure by class labels.
- **Labels**: Associated with each image for supervised learning.

### Converting Images to TFRecord

Use the `imageToTFrecord.py` script to convert images and labels into TFRecord format:

```python
python scripts/imageToTFrecord.py --input_dir path/to/images --output_dir path/to/tfrecords
```

## Pipeline Components

### ExampleGen

Ingests the TFRecord files into the pipeline.

### ExampleValidator

Validates the data to ensure quality and consistency.

### SchemaGen

Generates the schema for data validation based on the dataset.

### ExampleTransform

Transforms the raw images and labels into the format required for training.

### Trainer

Trains a TensorFlow model using the processed data. The model architecture is defined in `model.py`.

### Evaluator

Evaluates the model performance on validation data and generates evaluation metrics.

### Pusher

Pushes the model to a serving infrastructure for inference.

### Resolver

Resolves and manages model and data artifacts throughout the pipeline.

## Running the Pipeline

 **Initialize and Run the Pipeline**

   ```bash
   run .\pipelines\apache_beam\pipeline_beam.py
   ```

## Model Serving

To serve the trained model using TensorFlow Serving:

1. **Build and Run TensorFlow Serving Docker Container**

   ```bash
   !docker run -p 8500:8500 \
   -p 8501:8501 \
   --mount type=bind,source=".\serving_model_dir\1724395112",target=/models/my_model/1 \
   -e MODEL_NAME=my_model -t tensorflow/serving
   ```

2. **Send Prediction Requests**

   Use the REST API to send prediction requests:

   ```python
    import tensorflow as tf
    import requests
    import json
    import base64
    
    def serialize_example(image_path):
        # Read and serialize the image
        image = tf.io.read_file(image_path)
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))  # Dummy label
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    # Serialize the image into a TFRecord format
    tfrecord_data = serialize_example("Test/Malignant/047_CT_56-seg_15.png")
    # Base64 encode the serialized TFRecord
    tfrecord_data_base64 = base64.b64encode(tfrecord_data).decode('utf-8')
    
    # Create the payload for the REST API request
    data = {
        "signature_name": "serving_default",  # Ensure this matches your SavedModel's signature
        "instances": [{"examples": {"b64": tfrecord_data_base64}}]  # Send base64-encoded TFRecord data
    }
    
   ```
   Send the request to TensorFlow Serving running in Docker
   ```
    url = "http://localhost:8501/v1/models/my_model:predict"  # Adjust the model name and endpoint as needed
    headers = {"content-type": "application/json"}
    # Send the request
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    # Print the response from TensorFlow Serving
    print(response.json())
   ```


## Results and Evaluation

The model’s performance is evaluated based on metrics such as accuracy, precision, recall, and F1-score. Evaluation results can be found in `evaluation_results.md`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. 
## License

This project is licensed under the [MIT License](LICENSE).
