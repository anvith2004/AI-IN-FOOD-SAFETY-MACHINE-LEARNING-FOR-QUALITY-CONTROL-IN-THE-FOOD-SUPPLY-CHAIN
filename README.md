# AI-IN-FOOD-SAFETY-MACHINE-LEARNING-FOR-QUALITY-CONTROL-IN-THE-FOOD-SUPPLY-CHAIN
A Flask-based Risk Decision Prediction System using a pre-trained XGBoost model. Features a user-friendly web interface with dynamic drop-downs and AJAX for real-time predictions. Includes data preprocessing with TF-IDF and OneHotEncoding. Modular design for easy customization and deployment.

This project is a Risk Decision Prediction System that leverages a pre-trained XGBoost model to classify food safety incidents based on textual and categorical data. The system provides a user-friendly web interface built with Flask for dynamic input and prediction.

Key Features
- Machine Learning:
  - Uses XGBoost for high-accuracy predictions.
  - Preprocessing with TF-IDF (for text) and OneHotEncoding (for categorical data).
- Web Application:
  - Interactive form with drop-down menus for structured input.
  - Real-time predictions displayed dynamically using AJAX.
- Data Consistency:
  - Drop-down options are dynamically populated from the training data for accuracy and usability.
- Modular Design:
  - Clean separation of model training, preprocessing, and web application logic.

Tech Stack
- Frontend: HTML, CSS, JavaScript, Jinja2, AJAX.
- Backend: Flask, Python, Pandas, Scikit-learn, XGBoost.
- Deployment: Flask server (local development).

How It Works
1. Input: Users enter incident details (e.g., category, type, hazards) via the web interface.
2. Processing: The backend preprocesses the input and uses a pre-trained XGBoost model to predict the risk decision.
3. Output: The predicted risk decision (e.g., "Serious" or "Not Serious") is displayed instantly.

Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train_model.py`
3. Start the Flask server: `python app.py`
4. Open in browser: `http://127.0.0.1:5002`
