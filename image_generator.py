from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

class ImageGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        
    def generate_image(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                )
            )
            
            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                return base64_image
            
            return None
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None
