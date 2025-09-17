from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Dict, Any
import logging
from config import config
import os

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global variable to store the model
model = None
is_tfsm_layer = False  # Flag to track model type

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, is_tfsm_layer
    logger.info("Starting application...")
    
    # Try to load model, but don't fail if it's not available
    try:
        if load_model():
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning("Model loading failed on startup - API will run without model")
            logger.warning("You can still access health checks and documentation")
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        logger.warning("API will start without model loaded")
    
    yield
    # Shutdown
    logger.info("Application shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

def check_model_path():
    """Check what model files are available at the specified path"""
    import glob
    
    base_path = config.MODEL_PATH
    logger.info(f"Checking model path: {base_path}")
    
    if not os.path.exists(base_path):
        logger.error(f"Model directory does not exist: {base_path}")
        return None
    
    # Check for different model formats
    model_files = {
        'keras_v3': glob.glob(os.path.join(base_path, "*.keras")),
        'h5_files': glob.glob(os.path.join(base_path, "*.h5")),
        'savedmodel_pb': glob.glob(os.path.join(base_path, "saved_model.pb")),
        'savedmodel_pbtxt': glob.glob(os.path.join(base_path, "saved_model.pbtxt")),
        'all_files': glob.glob(os.path.join(base_path, "*"))
    }
    
    logger.info("Found files:")
    for format_type, files in model_files.items():
        if files:
            logger.info(f"  {format_type}: {files}")
        else:
            logger.info(f"  {format_type}: None")
    
    return model_files

def load_model():
    """Load the TensorFlow model"""
    global model, is_tfsm_layer
    
    # First check what's available
    model_files = check_model_path()
    if model_files is None:
        return False
    
    try:
        # Try different loading strategies based on available files
        
        # 1. Try Keras v3 format (.keras files)
        if model_files['keras_v3']:
            keras_file = model_files['keras_v3'][0]
            logger.info(f"Attempting to load Keras v3 file: {keras_file}")
            try:
                model = tf.keras.models.load_model(keras_file)
                is_tfsm_layer = False
                logger.info(f"Successfully loaded Keras v3 model from {keras_file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load Keras v3 file: {e}")
        
        # 2. Try H5 format (.h5 files)
        if model_files['h5_files']:
            h5_file = model_files['h5_files'][0]
            logger.info(f"Attempting to load H5 file: {h5_file}")
            try:
                model = tf.keras.models.load_model(h5_file)
                is_tfsm_layer = False
                logger.info(f"Successfully loaded H5 model from {h5_file}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load H5 file: {e}")
        
        # 3. Try the original path as Keras model
        logger.info(f"Attempting to load original path as Keras model: {config.MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(config.MODEL_PATH)
            is_tfsm_layer = False
            logger.info(f"Successfully loaded Keras model from {config.MODEL_PATH}")
            return True
        except Exception as keras_error:
            logger.info(f"Keras model loading failed: {keras_error}")
        
        # 4. Try SavedModel format with TFSMLayer
        if model_files['savedmodel_pb'] or model_files['savedmodel_pbtxt']:
            logger.info("Found SavedModel files, attempting to load with TFSMLayer...")
            try:
                model = tf.keras.layers.TFSMLayer(config.MODEL_PATH, call_endpoint='serving_default')
                is_tfsm_layer = True
                logger.info(f"Successfully loaded TFSMLayer from {config.MODEL_PATH}")
                return True
            except Exception as e:
                logger.warning(f"TFSMLayer loading failed: {e}")
                
                # Try alternative call endpoints
                for endpoint in ['predict', 'inference', 'serve']:
                    try:
                        logger.info(f"Trying alternative endpoint: {endpoint}")
                        model = tf.keras.layers.TFSMLayer(config.MODEL_PATH, call_endpoint=endpoint)
                        is_tfsm_layer = True
                        logger.info(f"Successfully loaded TFSMLayer with endpoint '{endpoint}'")
                        return True
                    except Exception as alt_e:
                        logger.warning(f"Failed with endpoint '{endpoint}': {alt_e}")
        
        # 5. Last resort: try to load any model-like files found
        for file_path in model_files['all_files']:
            if os.path.isfile(file_path) and (file_path.endswith('.pb') or file_path.endswith('.h5') or file_path.endswith('.keras')):
                logger.info(f"Last resort: trying to load {file_path}")
                try:
                    if file_path.endswith('.pb'):
                        # Try as SavedModel directory
                        model_dir = os.path.dirname(file_path)
                        model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')
                        is_tfsm_layer = True
                    else:
                        model = tf.keras.models.load_model(file_path)
                        is_tfsm_layer = False
                    logger.info(f"Successfully loaded model from {file_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.error("All model loading attempts failed")
        return False
            
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {str(e)}")
        return False

def make_prediction(processed_image):
    """Make prediction using the loaded model"""
    global model, is_tfsm_layer
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Log input characteristics for debugging
        logger.info(f"Input shape: {processed_image.shape}")
        logger.info(f"Input min/max: {processed_image.min():.6f} / {processed_image.max():.6f}")
        logger.info(f"Input mean/std: {processed_image.mean():.6f} / {processed_image.std():.6f}")
        
        if is_tfsm_layer:
            # For TFSMLayer, we need to call it differently
            predictions = model(processed_image)
            # TFSMLayer might return a dict, extract the predictions
            if isinstance(predictions, dict):
                # Common output names in SavedModel
                possible_keys = ['output_0', 'predictions', 'dense', 'logits']
                predictions_array = None
                for key in possible_keys:
                    if key in predictions:
                        predictions_array = predictions[key].numpy()
                        break
                if predictions_array is None:
                    # Take the first available output
                    predictions_array = list(predictions.values())[0].numpy()
            else:
                predictions_array = predictions.numpy()
        else:
            # Standard Keras model - handle batch size mismatch
            try:
                # Try direct prediction first
                predictions_array = model.predict(processed_image)
            except Exception as batch_error:
                logger.warning(f"Batch size mismatch, trying with repeated input: {batch_error}")
                # If model expects batch size 32, repeat the image
                expected_batch_size = model.input_shape[0] if model.input_shape[0] is not None else 32
                if expected_batch_size > 1:
                    repeated_input = np.repeat(processed_image, expected_batch_size, axis=0)
                    predictions_array = model.predict(repeated_input)
                    # Take only the first prediction since all are identical
                    predictions_array = predictions_array[:1]
                else:
                    raise batch_error
        
        # Log prediction characteristics for debugging
        logger.info(f"Raw predictions shape: {predictions_array.shape}")
        logger.info(f"Raw predictions sum: {predictions_array.sum():.6f}")
        logger.info(f"Raw predictions min/max: {predictions_array.min():.6f} / {predictions_array.max():.6f}")
        # Use first prediction from batch
        first_prediction = predictions_array[0]
        logger.info(f"First prediction shape: {first_prediction.shape}")
        logger.info(f"Top 3 prediction values: {sorted(first_prediction, reverse=True)[:3]}")
        logger.info(f"Predicted class index: {first_prediction.argmax()}")
        logger.info(f"Predicted class name: {config.CLASS_NAMES[first_prediction.argmax()]}")
        
        return predictions_array
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def preprocess_image(image: Image.Image, target_size: tuple = None) -> np.ndarray:
    """
    Preprocess the image for model prediction
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE
        
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Keep original pixel values (0-255), no normalization
        image_array = image_array.astype(np.float32)
        
        # Add batch dimension - just single image
        image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 256, 256, 3)
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error preprocessing image")

@app.get("/")
async def root():
    """Root endpoint - serves the web interface"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {"message": "Plant Disease Classification API", "status": "active", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "TFSMLayer" if is_tfsm_layer else "Keras" if model is not None else "None",
        "model_path": config.MODEL_PATH,
        "classes_count": len(config.CLASS_NAMES) if model is not None else 0
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    model_files = check_model_path()
    return {
        "model_path": config.MODEL_PATH,
        "model_loaded": model is not None,
        "model_type": "TFSMLayer" if is_tfsm_layer else "Keras" if model is not None else "None",
        "available_files": model_files,
        "classes_count": len(config.CLASS_NAMES)
    }

@app.post("/reload-model")
async def reload_model():
    """Reload the model manually"""
    global model, is_tfsm_layer
    
    # Reset model state
    model = None
    is_tfsm_layer = False
    
    # Try to load model
    success = load_model()
    
    return {
        "success": success,
        "model_loaded": model is not None,
        "model_type": "TFSMLayer" if is_tfsm_layer else "Keras" if model is not None else "None",
        "message": "Model reloaded successfully" if success else "Model loading failed"
    }

@app.get("/classes")
async def get_classes():
    """Get all available class names"""
    return {
        "classes": config.CLASS_NAMES,
        "total_classes": len(config.CLASS_NAMES)
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Prediction results with class name, confidence, and probabilities
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = make_prediction(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = config.CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                "class": config.CLASS_NAMES[i],
                "confidence": float(predictions[0][i])
            }
            for i in top_5_indices
        ]
        
        # Parse plant and disease from class name
        if "___" in predicted_class:
            plant, disease = predicted_class.split("___", 1)
        else:
            plant = predicted_class
            disease = "Unknown"
        
        # Debug logging to trace text corruption
        logger.info(f"Raw predicted_class: '{predicted_class}'")
        logger.info(f"Parsed plant: '{plant}'")
        logger.info(f"Parsed disease: '{disease}'")
        
        return {
            "prediction": {
                "class": predicted_class,
                "plant": plant,
                "disease": disease,
                "confidence": confidence
            },
            "top_5_predictions": top_5_predictions,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Predict plant diseases for multiple images
    
    Args:
        files: List of uploaded image files
    
    Returns:
        Batch prediction results
    """
    try:
        if len(files) > config.MAX_BATCH_SIZE:  # Limit batch size
            raise HTTPException(
                status_code=400, 
                detail=f"Maximum {config.MAX_BATCH_SIZE} files allowed per batch"
            )
        
        results = []
        
        for file in files:
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "error": "File must be an image"
                    })
                    continue
                
                # Read and process image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                processed_image = preprocess_image(image)
                
                # Make prediction
                predictions = make_prediction(processed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = config.CLASS_NAMES[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index])
                
                # Parse plant and disease
                if "___" in predicted_class:
                    plant, disease = predicted_class.split("___", 1)
                else:
                    plant = predicted_class
                    disease = "Unknown"
                
                results.append({
                    "filename": file.filename,
                    "prediction": {
                        "class": predicted_class,
                        "plant": plant,
                        "disease": disease,
                        "confidence": confidence
                    }
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "batch_results": results,
            "total_files": len(files),
            "successful_predictions": len([r for r in results if "error" not in r])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
