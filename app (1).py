import os
import uuid
import random
import requests
import json
import re
from PIL import Image
from dotenv import load_dotenv
import gradio as gr
from gradio_client import Client

# Load environment variables
load_dotenv()
# Load multiple API keys for OpenRouter
OPENROUTER_API_KEY_1 = os.getenv("OPENROUTER_API_KEY_1")
OPENROUTER_API_KEY_2 = os.getenv("OPENROUTER_API_KEY_2")
OPENROUTER_API_KEY_3 = os.getenv("OPENROUTER_API_KEY_3")
OPENROUTER_API_KEY_4 = os.getenv("OPENROUTER_API_KEY_4")
OPENROUTER_API_KEYS = [OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2, OPENROUTER_API_KEY_3, OPENROUTER_API_KEY_4]
# Filter out any None values
OPENROUTER_API_KEYS = [key for key in OPENROUTER_API_KEYS if key]

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TTS_PASSWORD = os.getenv("TTS_PASSWORD")
HF_TOKEN = os.getenv("HF_TOKEN")  # Add Hugging Face token for authentication

if not DEEPSEEK_API_KEY:
    raise EnvironmentError("Missing DEEPSEEK_API_KEY in environment.")
if not OPENROUTER_API_KEYS:
    raise EnvironmentError("No OPENROUTER_API_KEYS found in environment. Please provide at least one key.")
if not TTS_PASSWORD:
    raise EnvironmentError("Missing TTS_PASSWORD in environment.")
if not HF_TOKEN:
    raise EnvironmentError("Missing HF_TOKEN in environment. Required to access private TTS API.")

# Create audio directory if it doesn't exist
if not os.path.exists("audio"):
    os.makedirs("audio")

# Language mapping for gTTS
LANGUAGE_TO_GTTS = {
    'Afrikaans': 'af', 'Amharic': 'am', 'Arabic': 'ar', 'Bulgarian': 'bg', 'Bengali': 'bn', 'Bosnian': 'bs',
    'Catalan': 'ca', 'Czech': 'cs', 'Welsh': 'cy', 'Danish': 'da', 'German': 'de', 'Greek': 'el', 'English': 'en',
    'Spanish': 'es', 'Estonian': 'et', 'Basque': 'eu', 'Finnish': 'fi', 'French': 'fr', 'French (Canada)': 'fr-CA',
    'Galician': 'gl', 'Gujarati': 'gu', 'Hausa': 'ha', 'Hindi': 'hi', 'Croatian': 'hr', 'Hungarian': 'hu',
    'Indonesian': 'id', 'Icelandic': 'is', 'Italian': 'it', 'Hebrew': 'iw', 'Japanese': 'ja', 'Javanese': 'jw',
    'Khmer': 'km', 'Kannada': 'kn', 'Korean': 'ko', 'Latin': 'la', 'Lithuanian': 'lt', 'Latvian': 'lv',
    'Malayalam': 'ml', 'Marathi': 'mr', 'Malay': 'ms', 'Myanmar (Burmese)': 'my', 'Nepali': 'ne', 'Dutch': 'nl',
    'Norwegian': 'no', 'Punjabi (Gurmukhi)': 'pa', 'Polish': 'pl', 'Portuguese (Brazil)': 'pt',
    'Portuguese (Portugal)': 'pt-PT', 'Romanian': 'ro', 'Russian': 'ru', 'Sinhala': 'si', 'Slovak': 'sk',
    'Albanian': 'sq', 'Serbian': 'sr', 'Sundanese': 'su', 'Swedish': 'sv', 'Swahili': 'sw', 'Tamil': 'ta',
    'Telugu': 'te', 'Thai': 'th', 'Filipino': 'tl', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur',
    'Vietnamese': 'vi', 'Cantonese': 'yue', 'Chinese (Simplified)': 'zh-CN', 'Chinese (Mandarin/Taiwan)': 'zh-TW',
    'Chinese (Mandarin)': 'zh'
}

# Default VLM model to use
DEFAULT_VLM_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"

# TTS client
tts_client = Client("KindSynapse/Youssef-Ahmed-Private-Text-To-Speech-Unlimited", hf_token=HF_TOKEN)

# Default TTS emotion
DEFAULT_TTS_EMOTION = "Voice Affect: Energetic and animated; dynamic with variations in pitch and tone. Tone: Excited and enthusiastic, conveying an upbeat and thrilling atmosphere. Pacing: Rapid delivery when describing the game or the key moments (e.g., \"an overtime thriller,\" \"pull off an unbelievable win\") to convey the intensity and build excitement. Slightly slower during dramatic pauses to let key points sink in. Emotion: Intensely focused, and excited. Giving off positive energy. Personality: Relatable and engaging. Pauses: Short, purposeful pauses after key moments in the game."

# Helper function to get language code for language selection
def get_gtts_lang_code(language_name: str) -> str:
    return LANGUAGE_TO_GTTS.get(language_name, "en")

# Function to detect product in image using VLM with multiple API keys for fallback
def img_detector(model, image_url):
    errors = []
    
    # Try each API key until one works
    for api_key in OPENROUTER_API_KEYS:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What is the product in this image? Please provide a detailed description."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ]
                }),
                timeout=30  # Set a reasonable timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            
            # If we get here, the request failed but didn't raise an exception
            errors.append(f"API Key {OPENROUTER_API_KEYS.index(api_key) + 1} failed with status code: {response.status_code}")
        
        except Exception as e:
            errors.append(f"API Key {OPENROUTER_API_KEYS.index(api_key) + 1} error: {str(e)}")
            continue  # Try the next API key
    
    # If all API keys failed
    error_message = "\n".join(errors)
    return f"All VLM API requests failed:\n{error_message}"

# Function to extract product info using LLM
def extract_product_info(vlm_description, lang):
    prompt = f"""
    Based on the following VLM description of a product image, please extract and provide:
    1. Product Name: A concise name for the product.
    2. Product Category: A single category that best describes the product.
    3. Product Description: A professional marketing description in {lang} (30-50 words).
    VLM Description:
    {vlm_description}
    Format your response as JSON with the following structure:
    {{
        "product_name": "Name of the product" Always give it to me in english,
        "category": "Product category" Always give it to me in english,
        "description": "Professional marketing description"
    }}
    The description should be professionally written, focus on features and benefits, and avoid any introductory phrases like "Here is" or "This is".
    """
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional product analyst and copywriter. Extract structured information from visual descriptions and create professional marketing copy."
                    },
                    {
                        "role": "user",
                        "content": prompt.strip()
                    }
                ],
                "temperature": random.uniform(0.9, 1),
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            },
            timeout=30  # Set a reasonable timeout
        )
        
        result = response.json()["choices"][0]["message"]["content"]
        return json.loads(result)
        
    except Exception as e:
        return {
            "product_name": "Error extracting information",
            "category": "Unknown",
            "description": f"Error: {str(e)}"
        }

# Function to check if text contains Arabic
def contains_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return bool(arabic_pattern.search(text))

# Function to generate audio from text
def text_to_speech(message: str, language: str) -> str:
    clean_text = re.sub(r'<[^>]+>', '', message)
    clean_text = clean_text.lstrip().replace("\n", " ")
    
    if len(clean_text) > 500:
        clean_text = clean_text[:500] + "..."
    
    filename = f"audio/audio_{uuid.uuid4().hex}.mp3"
    
    # Determine if text contains Arabic
    is_arabic = contains_arabic(clean_text)
    
    # Adjust emotion for Arabic text
    emotion = DEFAULT_TTS_EMOTION
    if is_arabic:
        emotion = emotion + " Speaking in Egyptian Arabic dialect."
    
    try:
        # Call the TTS API
        result = tts_client.predict(
            password=TTS_PASSWORD,
            prompt=clean_text,
            voice="nova",
            emotion=emotion,
            use_random_seed=True,
            specific_seed=random.randint(1, 100000),
            api_name="/text_to_speech_app"
        )
        
        # Handle different response types
        if isinstance(result, tuple):
            # Check if any item in the tuple is a URL or file path
            for item in result:
                if isinstance(item, str):
                    if item.startswith('http'):
                        # It's a URL, download it
                        response = requests.get(item)
                        if response.status_code == 200:
                            with open(filename, 'wb') as f:
                                f.write(response.content)
                            return filename
                    elif os.path.exists(item) and os.path.isfile(item):
                        # It's a file path, copy it
                        import shutil
                        shutil.copy(item, filename)
                        return filename
            
            # If we got here, we couldn't find a usable audio file in the tuple
            raise Exception(f"No usable audio found in API response tuple: {result}")
            
        elif isinstance(result, str):
            # Handle string result (URL or file path)
            if os.path.exists(result):
                # If result is a file path, copy it to our directory
                import shutil
                shutil.copy(result, filename)
            else:
                # If result is a URL, download it
                response = requests.get(result)
                if response.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download audio from URL: {response.status_code}")
            
            return filename
        else:
            # Unknown result type
            raise Exception(f"Unexpected result type from TTS API: {type(result).__name__}")
            
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return f"Text-to-speech error: {str(e)}"

# Function to upload image and get base64 URL
def upload_image_and_get_url(image_path):
    # For temporary URL for local testing, just return the file path
    # In production, you might want to create a temporary URL or host the image somewhere
    return image_path

# Process image file
def process_image(image_path, model_name, language):
    try:
        # Get a URL for the image
        image_url = upload_image_and_get_url(image_path)
        
        # Use VLM to detect and describe the product
        vlm_description = img_detector(model_name, image_url)
        
        # Check if VLM processing failed
        if vlm_description.startswith("All VLM API requests failed"):
            return "API Error", "Error", "All OpenRouter API keys failed. Please check your API keys and try again.", None, vlm_description
        
        # Extract product info using LLM
        product_info = extract_product_info(vlm_description, language)
        
        # Generate audio for the description
        try:
            audio_path = text_to_speech(product_info["description"], language)
            if audio_path.startswith("Text-to-speech error"):
                print(f"TTS Error: {audio_path}")
                # Return error but continue with other outputs
                return (
                    product_info["product_name"],
                    product_info["category"],
                    product_info["description"],
                    None,  # No audio
                    f"{vlm_description}\n\nTTS Error: {audio_path}"
                )
        except Exception as tts_error:
            print(f"TTS Exception: {str(tts_error)}")
            # Return error but continue with other outputs
            return (
                product_info["product_name"],
                product_info["category"],
                product_info["description"],
                None,  # No audio
                f"{vlm_description}\n\nTTS Exception: {str(tts_error)}"
            )
        
        return (
            product_info["product_name"],
            product_info["category"],
            product_info["description"],
            audio_path,
            vlm_description  # Return the raw VLM description for debugging
        )
    except Exception as e:
        print(f"Process Image Error: {str(e)}")
        return f"Error: {str(e)}", "Error", "Error processing image", None, str(e)

# Process image from URL
def process_image_url(image_url, model_name, language):
    try:
        # Use VLM to detect and describe the product
        vlm_description = img_detector(model_name, image_url)
        
        # Check if VLM processing failed
        if vlm_description.startswith("All VLM API requests failed"):
            return "API Error", "Error", "All OpenRouter API keys failed. Please check your API keys and try again.", None, vlm_description
        
        # Extract product info using LLM
        product_info = extract_product_info(vlm_description, language)
        
        # Generate audio for the description
        try:
            audio_path = text_to_speech(product_info["description"], language)
            if audio_path.startswith("Text-to-speech error"):
                print(f"TTS Error: {audio_path}")
                # Return error but continue with other outputs
                return (
                    product_info["product_name"],
                    product_info["category"],
                    product_info["description"],
                    None,  # No audio
                    f"{vlm_description}\n\nTTS Error: {audio_path}"
                )
        except Exception as tts_error:
            print(f"TTS Exception: {str(tts_error)}")
            # Return error but continue with other outputs
            return (
                product_info["product_name"],
                product_info["category"],
                product_info["description"],
                None,  # No audio
                f"{vlm_description}\n\nTTS Exception: {str(tts_error)}"
            )
        
        return (
            product_info["product_name"],
            product_info["category"],
            product_info["description"],
            audio_path,
            vlm_description  # Return the raw VLM description for debugging
        )
    except Exception as e:
        print(f"Process Image URL Error: {str(e)}")
        return f"Error: {str(e)}", "Error", "Error processing image URL", None, str(e)

# Test TTS API directly
def test_tts_api():
    try:
        sample_text = "This is a test of the text to speech API."
        result = tts_client.predict(
            password=TTS_PASSWORD,
            prompt=sample_text,
            voice="nova",
            emotion=DEFAULT_TTS_EMOTION,
            use_random_seed=True,
            specific_seed=random.randint(1, 100000),
            api_name="/text_to_speech_app"
        )
        
        # Print detailed information about the result
        result_type = type(result).__name__
        result_info = f"Result type: {result_type}"
        
        if isinstance(result, tuple):
            result_info += f"\nTuple length: {len(result)}"
            for i, item in enumerate(result):
                result_info += f"\n\nItem {i} type: {type(item).__name__}"
                if isinstance(item, str):
                    result_info += f"\nItem {i} string value: {item[:500]}..."
                    # Check if it's a file path
                    if os.path.exists(item):
                        result_info += f"\nItem {i} is an existing file path, size: {os.path.getsize(item)} bytes"
                else:
                    result_info += f"\nItem {i} value: {str(item)[:500]}..."
        elif isinstance(result, str):
            result_info += f"\nResult string length: {len(result)}"
            result_info += f"\nResult starts with: {result[:100]}..."
            
            # Check if it's a file path
            if os.path.exists(result):
                result_info += f"\nResult is an existing file path, size: {os.path.getsize(result)} bytes"
        
        return f"TTS API Test Successful\n{result_info}"
    except Exception as e:
        return f"TTS API Test Failed: {str(e)}"

# Show API status in the interface
def get_api_status():
    status_text = f"OpenRouter API Keys: {len(OPENROUTER_API_KEYS)} configured\n"
    status_text += f"DeepSeek API: {'Available' if DEEPSEEK_API_KEY else 'Not configured'}\n"
    status_text += f"TTS API: {'Available' if TTS_PASSWORD else 'Not configured'}\n"
    status_text += f"HF Token: {'Available' if HF_TOKEN else 'Not configured'}"
    return status_text

# Available VLM models
VLM_MODELS = [
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "google/gemini-2.0-flash-exp:free"
]

# Create Gradio interface
languages = list(LANGUAGE_TO_GTTS.keys())

with gr.Blocks(title="AI Product Description Generator") as demo:
    gr.Markdown("# AI Product Description Generator")
    gr.Markdown("Upload a product image or provide a URL, and get an AI-generated product name, category, description, and audio narration.")
    
    # API Status
    api_status_text = gr.Markdown(get_api_status())
    
    with gr.Tabs():
        with gr.TabItem("Upload Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Upload Product Image", type="filepath")
                    model_dropdown = gr.Dropdown(choices=VLM_MODELS, value=DEFAULT_VLM_MODEL, label="Vision Model")
                    language = gr.Dropdown(choices=languages, value="English", label="Language")
                    upload_button = gr.Button("Generate Product Info")
                
                with gr.Column(scale=2):
                    name_output = gr.Textbox(label="Product Name")
                    category_output = gr.Textbox(label="Product Category")
                    description_output = gr.Textbox(label="Product Description")
                    audio_output = gr.Audio(label="Audio Description")
                    vlm_raw_output = gr.Textbox(label="Raw VLM Output (Debug)", visible=False)
            
            debug_checkbox = gr.Checkbox(label="Show Raw VLM Output", value=False)
            
            def toggle_debug(show_debug):
                return gr.update(visible=show_debug)
            
            debug_checkbox.change(fn=toggle_debug, inputs=[debug_checkbox], outputs=[vlm_raw_output])
            
            upload_button.click(
                fn=process_image,
                inputs=[image_input, model_dropdown, language],
                outputs=[name_output, category_output, description_output, audio_output, vlm_raw_output]
            )
        
        with gr.TabItem("Image URL"):
            with gr.Row():
                with gr.Column(scale=1):
                    url_input = gr.Textbox(label="Product Image URL")
                    url_model_dropdown = gr.Dropdown(choices=VLM_MODELS, value=DEFAULT_VLM_MODEL, label="Vision Model")
                    url_language = gr.Dropdown(choices=languages, value="English", label="Language")
                    url_button = gr.Button("Generate Product Info from URL")
                
                with gr.Column(scale=2):
                    url_name_output = gr.Textbox(label="Product Name")
                    url_category_output = gr.Textbox(label="Product Category")
                    url_description_output = gr.Textbox(label="Product Description")
                    url_audio_output = gr.Audio(label="Audio Description")
                    url_vlm_raw_output = gr.Textbox(label="Raw VLM Output (Debug)", visible=False)
            
            url_debug_checkbox = gr.Checkbox(label="Show Raw VLM Output", value=False)
            
            url_debug_checkbox.change(fn=toggle_debug, inputs=[url_debug_checkbox], outputs=[url_vlm_raw_output])
            
            url_button.click(
                fn=process_image_url,
                inputs=[url_input, url_model_dropdown, url_language],
                outputs=[url_name_output, url_category_output, url_description_output, url_audio_output, url_vlm_raw_output]
            )
        
        with gr.TabItem("Debug Tools"):
            gr.Markdown("## Debug Tools")
            gr.Markdown("Use these tools to test the API connections and diagnose issues.")
            
            test_tts_button = gr.Button("Test TTS API")
            tts_test_output = gr.Textbox(label="TTS API Test Results", lines=10)
            
            test_tts_button.click(
                fn=test_tts_api,
                inputs=[],
                outputs=[tts_test_output]
            )

# Launch the application
if __name__ == "__main__":
    demo.launch()