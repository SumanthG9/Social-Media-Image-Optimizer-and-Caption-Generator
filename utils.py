from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import uuid
import os
import cv2
import numpy as np
import time
import random
import base64
import json
import google.generativeai as genai

# Define paths
OUTPUT_FOLDER = "media/optimized/"
CAPTION_FOLDER = "output_captions/"
STATIC_FOLDER = "static/"
TEMP_FOLDER = "temp/"

# Ensure directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CAPTION_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Define supported platforms & image sizes
SOCIAL_MEDIA_SIZES = {
    "instagram": (1080, 1080),
    "instagram_story": (1080, 1920),
    "twitter": (1600, 900),
    "linkedin": (1200, 627),
    "facebook": (1080, 1920),
    "facebook_story": (1080, 1920),
    "pinterest": (1000, 1500),
    "youtube_thumbnail": (1280, 720)
}

# Quality settings for image processing
MIN_RESOLUTION_HEIGHT = 720  # Maintain at least 720p resolution
MIN_QUALITY = 98  # Increased JPEG quality setting

# Load AI models
def load_models(api_key=None):
    """Initialize and load AI models with error handling."""
    models = {}
    try:
        # Configure Gemini
        if api_key:
            genai.configure(api_key=api_key)
            
            # Initialize Gemini model
            models["gemini"] = genai.GenerativeModel("gemini-1.5-pro")
            models["gemini_loaded"] = True
            print("Gemini model loaded successfully")
        else:
            print("No API key provided for Gemini")
            models["gemini_loaded"] = False
            
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return {"gemini_loaded": False}

def sanitize_text(text):
    """Clean text to ensure it can be properly saved to files."""
    if not text:
        return ""
    try:
        # Try to encode/decode to catch any encoding issues
        return text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        # If there's an encoding issue, replace problematic characters
        return ''.join(c if ord(c) < 65536 else '?' for c in text)

# Process image with Gemini for caption, hashtags, and smart crop info
def process_with_gemini(image_path, models, platform):
    """Process image with Gemini for platform-specific caption, hashtags and smart crop info"""
    try:
        # Check if Gemini model is loaded
        if not models.get("gemini_loaded", False):
            return "No caption could be generated. Gemini model not loaded.", [0.0, 0.0, 1.0, 1.0], "#ai #image"
            
        # Read image as bytes
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        
        # Create platform-specific prompt
        platform_prompts = {
            "instagram": "Create an engaging, conversational caption with emojis that would perform well on Instagram. Focus on lifestyle aspects and emotions.",
            "instagram_story": "Create a brief, attention-grabbing caption suitable for Instagram Stories with relevant emojis.",
            "twitter": "Create a concise, witty caption under 280 characters that would work well on Twitter/X.",
            "linkedin": "Create a professional-sounding caption that highlights business value, industry insights or professional development aspects.",
            "facebook": "Create a casual, descriptive caption that encourages engagement and conversation.",
            "facebook_story": "Create a short, engaging caption suitable for Facebook Stories.",
            "pinterest": "Create an inspirational caption that describes DIY, home decor, recipe, or lifestyle aspects with searchable keywords.",
            "youtube_thumbnail": "Create an attention-grabbing title for a YouTube video that drives clicks while accurately describing the content."
        }
        
        # Get platform-specific prompt instruction
        platform_instruction = platform_prompts.get(platform, "Create a general caption for social media")
            
        # Create the content parts properly according to Gemini API requirements
        contents = [
            {
                "text": f"""Analyze this image and provide:
1. {platform_instruction} (5-6 sentences max)
2. Bounding box coordinates of the main subject in format [x1, y1, x2, y2]
   where coordinates are normalized from 0 to 1
3. 8-10 relevant hashtags specific to {platform.replace('_', ' ')}, matching the platform's typical hashtag style and audience preferences

Format your response as JSON:
{{
  "caption": "your detailed caption here",
  "bounding_box": [x1, y1, x2, y2],
  "hashtags": "#hashtag1 #hashtag2 #hashtag3"
}}

Your response must be valid JSON only, with no additional text."""
            },
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_bytes).decode()
            }
        ]
        
        # Process with Gemini
        response = models["gemini"].generate_content(contents)
        
        # Extract JSON from response
        try:
            # Handle potential text wrapper in response
            response_text = response.text
            # Look for JSON object in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
                
            result = json.loads(json_str)
            
            caption = result.get("caption", "No caption generated.")
            bounding_box = result.get("bounding_box", [0.0, 0.0, 1.0, 1.0])
            hashtags = result.get("hashtags", f"#{platform}")
            
            # Validate bounding box format
            if not isinstance(bounding_box, list) or len(bounding_box) != 4:
                bounding_box = [0.0, 0.0, 1.0, 1.0]
            
            # Ensure bounding box values are within 0-1 range
            bounding_box = [
                max(0.0, min(1.0, bounding_box[0])),
                max(0.0, min(1.0, bounding_box[1])),
                max(0.0, min(1.0, bounding_box[2])),
                max(0.0, min(1.0, bounding_box[3]))
            ]
            
            # Ensure second coordinate is greater than first (x2 > x1, y2 > y1)
            if bounding_box[2] <= bounding_box[0] or bounding_box[3] <= bounding_box[1]:
                bounding_box = [0.0, 0.0, 1.0, 1.0]
                
            return caption, bounding_box, hashtags
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Raw response: {response.text}")
            return "Image analysis failed due to processing error.", [0.0, 0.0, 1.0, 1.0], f"#{platform}"
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return "Image analysis failed.", [0.0, 0.0, 1.0, 1.0], f"#{platform}"

# Function to smart crop images using Gemini bounding box
def smart_crop_gemini(image, bounding_box):
    """Intelligently crop image based on Gemini-detected subject"""
    try:
        if not bounding_box or len(bounding_box) != 4:
            return image
            
        # Convert PIL image to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Convert normalized coordinates to pixel values
        x1 = int(bounding_box[0] * width)
        y1 = int(bounding_box[1] * height)
        x2 = int(bounding_box[2] * width)
        y2 = int(bounding_box[3] * height)
        
        # Add padding around the detection (20% on each side for better framing)
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        
        # Check if the crop area is valid
        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
            return image
            
        # Crop image
        cropped = img_array[y1:y2, x1:x2]
        return Image.fromarray(cropped)
    except Exception as e:
        print(f"Error in smart crop: {e}")
        return image

# Function to resize and optimize image for a specific platform
def optimize_for_platform(image, platform):
    """Resize and format image for target platform with improved quality"""
    try:
        if platform.lower() not in SOCIAL_MEDIA_SIZES:
            raise ValueError(f"Unsupported platform: {platform}")
            
        target_size = SOCIAL_MEDIA_SIZES[platform.lower()]
        
        # Convert to RGB if needed
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Calculate aspect ratios
        img_aspect = image.width / image.height
        target_aspect = target_size[0] / target_size[1]
        
        # Ensure minimum resolution
        if image.height < MIN_RESOLUTION_HEIGHT:
            new_width = int(MIN_RESOLUTION_HEIGHT * img_aspect)
            # Use higher quality Lanczos resampling
            image = image.resize((new_width, MIN_RESOLUTION_HEIGHT), Image.Resampling.LANCZOS)
        
        # Adjust image to match target aspect ratio
        if abs(img_aspect - target_aspect) > 0.01:
            if img_aspect > target_aspect:
                # Image is wider - resize to match target height
                new_width = int(target_size[1] * img_aspect)
                image = image.resize((new_width, target_size[1]), Image.Resampling.LANCZOS)
            else:
                # Image is taller - resize to match target width
                new_height = int(target_size[0] / img_aspect)
                image = image.resize((target_size[0], new_height), Image.Resampling.LANCZOS)
            
            # Center crop to target dimensions
            left = (image.width - target_size[0]) // 2
            top = (image.height - target_size[1]) // 2
            right = left + target_size[0]
            bottom = top + target_size[1]
            image = image.crop((left, top, right, bottom))
        else:
            # Direct resize if aspect ratios already match
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error in optimize_for_platform: {e}")
        raise e

def process_image_complete(image_bytes, platform, format="JPEG", smart_crop_enabled=False, 
                           enhance_image_enabled=False, confidence=0.3, AI_MODELS=None):
    """Complete image processing pipeline using Gemini"""
    try:
        # Generate a unique ID for this processing request
        process_id = str(uuid.uuid4())
        timestamp_val = int(time.time() * 1000)
        
        # Parse image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save original image to temp folder for caption generation
        original_filename = f"original_{process_id}.{format.lower()}"
        original_path = os.path.join(TEMP_FOLDER, original_filename)
        image.save(original_path)
        
        # STEP 1: Process with Gemini for caption, hashtags, and bounding box
        if AI_MODELS and AI_MODELS.get("gemini_loaded", False):
            # Implement fallback mechanism to handle API errors
            try:
                caption, bounding_box, hashtags = process_with_gemini(original_path, AI_MODELS, platform)
            except Exception as e:
                print(f"Gemini processing failed: {e}, using fallback")
                caption = "An image showing scenery or objects."
                bounding_box = [0.0, 0.0, 1.0, 1.0]  # Full image
                hashtags = f"#{platform} #photo #image"
        else:
            caption = "Image analysis not available. Gemini model not loaded."
            bounding_box = [0.0, 0.0, 1.0, 1.0]  # Full image
            hashtags = f"#{platform}"
        
        # STEP 2: Apply smart cropping if enabled
        smart_cropped = False
        if smart_crop_enabled and bounding_box:
            try:
                new_image = smart_crop_gemini(image, bounding_box)
                if new_image is not image:  # Check if cropping was applied
                    image = new_image
                    smart_cropped = True
            except Exception as e:
                print(f"Smart cropping failed: {e}")
        
        # STEP 3: Optimize image for platform (resize)
        image = optimize_for_platform(image, platform)
        
        # STEP 4: Image enhancement removed
        # Always set enhanced to False since we removed enhancement
        enhanced = False
        
        # Sanitize caption and hashtags for proper encoding
        caption = sanitize_text(caption)
        hashtags = sanitize_text(hashtags)
        
        # STEP 5: Save the final processed image
        output_filename = f"{platform}_{process_id}_{timestamp_val}.{format.lower()}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Save with high quality settings and advanced options
        if format.upper() == "JPEG":
            image.save(output_path, quality=MIN_QUALITY, optimize=True, subsampling=0)
        else:
            image.save(output_path, quality=MIN_QUALITY, optimize=True)
        
        # Save caption to file
        caption_filename = f"{platform}_{process_id}_{timestamp_val}.txt"
        caption_path = os.path.join(CAPTION_FOLDER, caption_filename)
        
        # Save caption and hashtags to separate lines with UTF-8 encoding
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(f"{caption}\n\n{hashtags}")
        
        # Clean up temporary files
        if os.path.exists(original_path):
            try:
                os.remove(original_path)
            except Exception as cleanup_error:
                print(f"Error removing temporary file: {cleanup_error}")
                
        return {
            "platform": platform.replace("_", " ").title(),
            "size": SOCIAL_MEDIA_SIZES[platform],
            "optimized_image": f"/media/optimized/{output_filename}",
            "caption_text": caption,
            "hashtags": hashtags,
            "enhanced": enhanced,  # Will always be False now
            "smart_cropped": smart_cropped
        }
    except Exception as e:
        print(f"Error in complete processing: {e}")
        # Return an error dictionary instead of raising to prevent API failure
        return {
            "error": f"Processing failed: {str(e)}",
            "platform": platform.replace("_", " ").title(),
            "size": SOCIAL_MEDIA_SIZES.get(platform, (0, 0)),
            "caption_text": "Image processing failed.",
            "hashtags": f"#{platform}",
            "enhanced": False,
            "smart_cropped": False
        }