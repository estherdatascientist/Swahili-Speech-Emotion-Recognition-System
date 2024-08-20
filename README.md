---

# Skiza-App: Speech Emotion Recognition for Kenyan Swahili

Welcome to Skiza-App! This project focuses on developing and deploying a model to recognize emotions from speech, specifically tailored for Kenyan Swahili. The model leverages advanced machine learning techniques and is deployed using Streamlit for the user interface and FastAPI for API integration.

## **Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Model Evaluation](#model-evaluation)
5. [Deployment](#deployment)
6. [Contributing](#contributing)
7. [License](#license)

## **Installation**

To get started with Skiza-App, follow these steps to set up your environment and install the necessary dependencies:

### **1. Clone the Repository**

```bash
git clone https://github.com/estherdatascientist/Swahili-Speech-Emotion-Recognition-System.git
cd Swahili-Speech-Emotion-Recognition-System
```

### **2. Create a Virtual Environment**

It’s recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **3. Install Dependencies**

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### **4. Download and Prepare Data**

Ensure you have the Swahili speech dataset. Update the `data_dir` path in the configuration files to point to your dataset location.

### **5. Install Streamlit and FastAPI**

If not included in `requirements.txt`, you may need to install Streamlit and FastAPI separately:

```bash
pip install streamlit fastapi uvicorn
```

## **Usage**

### **1. Training the Model**

To train the model, run the training script:

```bash
python train_model.py
```

This script will load the dataset, preprocess the audio files, extract features, and train various models. The best-performing model (CatBoost) will be saved in the `models` directory.

### **2. Running the Streamlit App**

To start the Streamlit web application, use:

```bash
streamlit run app.py
```

This will open a new browser window with the Streamlit interface where you can upload audio files and get emotion predictions.

### **3. Running the FastAPI Server**

To start the FastAPI server, use:

```bash
uvicorn api:app --reload
```

This will run the API server, allowing you to send HTTP requests to get emotion predictions.

## **Features**

- **Emotion Recognition**: Detects emotions from audio clips in Kenyan Swahili.
- **Real-time Predictions**: Provides immediate feedback through the Streamlit web app.
- **API Integration**: Offers RESTful API access through FastAPI for integration with other systems.
- **Model Evaluation**: Includes model evaluation metrics such as accuracy, confusion matrix, and ROC curves.

## **Model Evaluation**

The CatBoost model achieved an accuracy of 87%, making it the best performer among the tested models. For details on model performance and evaluation, refer to the `evaluation` module in the codebase.

## **Deployment**

Skiza-App is deployed using:

- **Streamlit**: For creating an interactive user interface where users can upload audio files and see predictions.
- **FastAPI**: For building a RESTful API to enable programmatic access to the SER model.

## **Contributing**

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please make sure to adhere to the coding standards and include tests for new features.

## **License**

This project is licensed under the GNU General Public License (GPL). See the [LICENSE](LICENSE) file for details.

Thank you for using and contributing to Skiza-App!

---