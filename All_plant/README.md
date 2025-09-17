# Plant Disease Classification API

## Overview
This repository contains a FastAPI-based web application for classifying plant diseases using deep learning models. The application allows users to upload images of plants, and it predicts the disease (if any) affecting the plant. The predictions are made using a pre-trained TensorFlow model.

## Features
- **Image Upload**: Upload images of plants to classify diseases.
- **Batch Prediction**: Predict diseases for multiple images at once.
- **Health Check**: Check the status of the API and the model.
- **Model Reload**: Reload the model dynamically without restarting the server.
- **Static Files**: Serve static files like the frontend interface.

## Requirements
- Python 3.8 or higher
- TensorFlow
- FastAPI
- Uvicorn
- PIL (Pillow)
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PrinceGhevariya-23DCS033/ISL_dataset.git
   cd ISL_dataset
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the FastAPI server:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to:
   - API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Web Interface: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Configuration
The application settings are managed in the `config.py` file. Key settings include:
- `MODEL_PATH`: Path to the TensorFlow model.
- `IMAGE_SIZE`: Target size for image preprocessing.
- `HOST` and `PORT`: Server host and port.

## Endpoints
- `GET /`: Serves the web interface.
- `GET /health`: Returns the health status of the API and model.
- `GET /classes`: Lists all available class names.
- `POST /predict`: Predicts the disease for a single image.
- `POST /batch_predict`: Predicts diseases for multiple images.
- `POST /reload-model`: Reloads the model dynamically.

## Folder Structure
```
ISL_dataset/
├── main.py          # FastAPI application entry point
├── config.py        # Configuration settings
├── static/          # Static files (e.g., HTML, CSS, JS)
├── saved_models/    # Directory for storing TensorFlow models
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Author
Prince Ghevariya (23DCS033)

---

Feel free to reach out for any questions or suggestions!
