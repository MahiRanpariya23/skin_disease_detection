🩺 Skin Disease Detection System

📌 Project Description

This project is a deep learning-based web application that detects skin diseases from images. It uses a Convolutional Neural Network (CNN) with MobileNetV2 architecture to classify skin lesions into different categories. The application is built using Python and deployed with Streamlit for an interactive user interface.


🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* Pandas
* Pillow (PIL)
* Scikit-learn

▶️ How to Run the Project

Step 1: Clone the repository
       
        git clone <repo-link>
        cd skin_disease_detection

Step 2: Create virtual environment
       
        python -m venv .venv

Step 3: Activate environment
       
        .venv\Scripts\activate

Step 4: Install dependencies
       
        pip install -r requirements.txt

Step 5: Prepare dataset
       
        python prepare_dataset.py

Step 6: Train the model
      
        python train.py

Step 7: Run the app
       
        streamlit run app.py

📂 Dataset

* Dataset used: HAM10000 (Skin Cancer MNIST)
* Source: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

📊 Results

* Model: MobileNetV2 (Transfer Learning)
* Accuracy: ~75% 
* Classes: 7 skin disease categories

📌 Features

* Upload skin image for prediction
* Displays disease name with confidence
* Risk level indication
* Description and recommendation
* Simple and interactive UI
