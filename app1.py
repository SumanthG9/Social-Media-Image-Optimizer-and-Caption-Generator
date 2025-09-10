from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import io
import os
import time
import traceback
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from utilities file
from utils import (
    load_models, process_image_complete, 
    SOCIAL_MEDIA_SIZES, OUTPUT_FOLDER, 
    STATIC_FOLDER
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: No Gemini API key found in environment variables. Set GEMINI_API_KEY environment variable.")
    logger.warning("No Gemini API key found")

# Initialize FastAPI app
app = FastAPI(title="Social Media Image Optimizer")
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static file directories
app.mount("/media", StaticFiles(directory="media"), name="media")
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

# Create required directories if they don't exist
for directory in [OUTPUT_FOLDER, "media", STATIC_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Load models at startup
AI_MODELS = {}

@app.on_event("startup")
def startup_event():
    global AI_MODELS
    try:
        AI_MODELS = load_models(api_key=GEMINI_API_KEY)
        logger.info(f"AI models loaded: gemini={'loaded' if AI_MODELS.get('gemini_loaded', False) else 'not loaded'}")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        AI_MODELS = {"gemini_loaded": False}

# API endpoint to get available platforms
@app.get("/platforms/")
async def get_platforms():
    # Format platform data to match React frontend expectations
    return JSONResponse(content={
        "platforms": {k: {"name": k.replace("_", " ").title(), "size": f"({v[0]}x{v[1]})"} 
                     for k, v in SOCIAL_MEDIA_SIZES.items()}
    })

# API endpoint to process image with optimization
@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...), 
    platform: str = Form(...), 
    format: str = Form("JPEG"),
    smart_crop_enabled: bool = Form(False),
    enhance_image_enabled: bool = Form(False),
    confidence: float = Form(0.3),
    timestamp: str = Form(None)  # Added timestamp parameter to prevent caching
):
    # Log incoming request
    logger.info(f"Processing image request: platform={platform}, format={format}, smart_crop={smart_crop_enabled}")
    
    # Validate file type
    valid_types = ["image/jpeg", "image/png", "image/jpg", "image/gif"]
    if file.content_type not in valid_types:
        logger.warning(f"Invalid file type: {file.content_type}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type. Please upload a JPEG, PNG, or GIF image."}
        )
    
    # Validate platform
    if platform not in SOCIAL_MEDIA_SIZES:
        logger.warning(f"Invalid platform: {platform}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid platform. Available platforms: {', '.join(SOCIAL_MEDIA_SIZES.keys())}"}
        )
    
    try:
        # Read image content
        contents = await file.read()
        logger.info(f"Successfully read image content, size: {len(contents)} bytes")
        
        # Process the image using the utility function
        result = process_image_complete(
            image_bytes=contents,
            platform=platform,
            format=format,
            smart_crop_enabled=smart_crop_enabled,
            enhance_image_enabled=enhance_image_enabled,
            confidence=confidence,
            AI_MODELS=AI_MODELS
        )
        
        # Check if the result contains an error
        if "error" in result:
            logger.error(f"Error in process_image_complete: {result['error']}")
            return JSONResponse(
                status_code=400,  # Change to 400 to indicate error
                content=result    # Return the error and fallback content
            )
        
        logger.info(f"Image processed successfully for platform: {platform}")
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(traceback.format_exc())  # Print full traceback for debugging
        
        return JSONResponse(
            status_code=500,  # Use 500 to indicate server error
            content={
                "error": f"An error occurred: {str(e)}",
                "platform": platform.replace("_", " ").title(),
                "size": SOCIAL_MEDIA_SIZES[platform],
            }
        )
# API endpoint to serve optimized images
@app.get("/media/optimized/{filename}")
def serve_optimized_image(filename: str):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type based on file extension
    file_ext = filename.split('.')[-1].lower()
    content_type = f"image/{file_ext}"
    if file_ext == "jpg":
        content_type = "image/jpeg"
        
    return FileResponse(
        file_path,
        media_type=content_type
    )

# Health check endpoint for frontend to verify API is running
@app.get("/health")
async def health_check():
    # Include Gemini status in health check
    gemini_status = "loaded" if AI_MODELS.get("gemini_loaded", False) else "not loaded"
    return {
        "status": "ok", 
        "message": "API is running", 
        "models": {
            "gemini": gemini_status
        }
    }

# API endpoint to manually reload models (useful if API key changes)
@app.post("/reload-models/")
async def reload_models():
    global AI_MODELS
    try:
        AI_MODELS = load_models(api_key=GEMINI_API_KEY)
        logger.info("Models successfully reloaded")
        return {
            "status": "ok",
            "message": "Models reloaded",
            "models": {
                "gemini": "loaded" if AI_MODELS.get("gemini_loaded", False) else "not loaded"
            }
        }
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return {
            "status": "error",
            "message": f"Error reloading models: {str(e)}",
            "models": {
                "gemini": "not loaded"
            }
        }

# For production deployment with React - check if file exists before serving
@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    react_path = os.path.join("frontend/build", full_path)
    if os.path.exists(react_path):
        return FileResponse(react_path)
    
    index_path = "frontend/build/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    # If even index.html doesn't exist, return a basic response
    return HTMLResponse(content="<html><body><h1>Social Media Image Optimizer</h1><p>Frontend not found. Please check installation.</p></body></html>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)