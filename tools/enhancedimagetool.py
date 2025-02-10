from tools.base import BaseTool
import os
import base64
import mimetypes
from PIL import Image
import io

class EnhancedImageTool(BaseTool):
    name = "enhancedimagetool"
    description = '''
    Enhanced image processing tool that handles various image formats.
    Takes an image file path as input and returns a formatted list of content blocks.
    Supports PNG, JPG, and GIF formats.
    Returns base64 encoded image data in Claude's expected format.
    '''
    input_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image file"
            }
        },
        "required": ["image_path"]
    }

    def _validate_image_format(self, file_path: str) -> str:
        supported_formats = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif'
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in supported_formats:
            raise ValueError(f"Unsupported image format: {ext}")
        
        return supported_formats[ext]

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        try:
            with Image.open(image_path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format=img.format)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                mime_type = self._validate_image_format(image_path)
                return image_data, mime_type
        except Exception as e:
            raise RuntimeError(f"Error processing image: {str(e)}")

    def _validate_output_format(self, output: list) -> bool:
        if not isinstance(output, list) or len(output) != 1:
            return False
        
        block = output[0]
        required_keys = {"type", "source"}
        source_keys = {"type", "media_type", "data"}
        
        if not all(key in block for key in required_keys):
            return False
        
        if not all(key in block["source"] for key in source_keys):
            return False
            
        return True

    def execute(self, **kwargs) -> list:
        image_path = kwargs.get("image_path")
        
        if not image_path:
            raise ValueError("Image path is required")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image_data, mime_type = self._encode_image(image_path)
            
            output = [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_data
                }
            }]
            
            if not self._validate_output_format(output):
                raise ValueError("Invalid output format")
                
            return output
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")