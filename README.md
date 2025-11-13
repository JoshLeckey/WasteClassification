AI Waste Classification Model Training
======================================

This repository contains the Python script (train.py) for training a hybrid deep learning model (MobileNetV3Large + EfficientNetB0) for household waste classification. This model is designed to be converted to TensorFlow Lite for on-device deployment in a mobile application, as described in the project documentation.

Requirements
------------

*   Python 3.8+
    
*   The Python libraries listed in requirements.txt.
    
*   (Optional but Recommended) An NVIDIA GPU with CUDA support for fast training.
    

Setup & Installation
--------------------

Follow these steps to set up your environment and run the training script.

### 1\. Clone or Download Files

Ensure you have all the following files in the same directory:

*   train.py
    
*   requirements.txt
    
*   labels.json
    
*   labels.txt
    

### 2\. Create a Python Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to avoid conflicts with other projects.

**On macOS/Linux:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m venv venv  source venv/bin/activate   `

**On Windows:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m venv venv  .\venv\Scripts\activate   `

### 3\. Install Dependencies

With your virtual environment active, install all the required libraries using the requirements.txt file:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 4\. (Optional) Training with a GPU (CUDA Setup)

To train on your NVIDIA GPU, you **do not** need to change the Python code. TensorFlow will automatically detect and use a compatible GPU if your system is set up correctly.

This is the modern, recommended setup:

1.  **Install NVIDIA Driver:** Make sure you have the latest drivers for your GPU installed from the [NVIDIA Drivers website](https://www.nvidia.com/Download/index.aspx). This is the only manual driver install you need.
    
2.  **Install TensorFlow:** The pip install -r requirements.txt command will install the tensorflow package, which now includes the necessary CUDA and cuDNN libraries automatically.
    
3.  python -c "import tensorflow as tf; print(tf.config.list\_physical\_devices('GPU'))"If it prints a list with your GPU details, you are ready to train.
    

### 5\. Download the Dataset

This script is designed to work with the "Garbage Classification V2" dataset from Kaggle.

1.  **Download:** Go to the dataset page: [https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
    
2.  **Unzip:** Download and unzip the file.
    
3.  **Move:** You should have a folder named garbage-dataset. Move this folder into your project directory, so its path is ./garbage-dataset.
    

Your final project directory should look like this:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   /your-project-folder  |-- /garbage-dataset   <-- (Folder from Kaggle)  |   |-- /battery  |   |-- /biological  |   |-- /cardboard  |   |-- ... (and all other classes)  |-- train.py  |-- requirements.txt  |-- labels.json  |-- labels.txt  |-- /venv   `

Training the Model
------------------

Once your environment is set up and the data is in place, you can run the training script:

       python train.py   `

The script will:

1.  Print class distributions.
    
2.  Show a sample batch of training images (a plot window may open).
    
3.  Build the hybrid model.
    
4.  Train the model for 8 epochs.
    
5.  Fine-tune the model for 3 epochs.
    
6.  Save all outputs to your project folder.
    

**To confirm your GPU is working:** While the script is training (e.g., "Epoch 1/8..."), open a new terminal and run nvidia-smi (on Linux or Windows). You should see a python process using your GPU's memory and compute.

Outputs
-------

When the script is finished, it will have generated the following files:

*   **model.tflite**: The final, optimized model for use in your Android app.
    
*   **best\_weights.h5**: The Keras-format weights from the best-performing epoch.
    
*   **hybrid\_quick\_accuracy.png**: A plot of the model's training and validation accuracy.
    
*   **hybrid\_quick\_loss.png**: A plot of the model's training and validation loss.
    
*   **hybrid\_quick\_confusion\_matrix.png**: A confusion matrix visualizing model performance.
    
*   **hybrid\_quick\_confusion\_matrix.csv**: The confusion matrix data in CSV format.
    
*   **hybrid\_quick\_per\_class\_accuracy.csv**: A CSV file listing the accuracy for each class.
    
*   **hybrid\_quick\_misclassifications.csv**: A list of misclassified images from the validation set.
    
*   **hybrid\_quick\_model\_summary.txt**: A text file containing the model's architecture summary.
