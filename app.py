from flask import Flask, render_template, request, jsonify, url_for
from ce3 import Assistant
import os
from werkzeug.utils import secure_filename
import base64
from config import Config
import mimetypes
from tools.enhancedimagetool import EnhancedImageTool

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the assistant
assistant = Assistant()
image_tool = EnhancedImageTool()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    image_data = data.get('image')  # Get the base64 image data
    
    # Prepare the message content
    if image_data:
        # Parse the image data and media type
        if isinstance(image_data, dict):
            base64_data = image_data.get('data', '')
            media_type = image_data.get('media_type')
        else:
            base64_data = image_data.split(',')[1] if ',' in image_data else image_data
            media_type = 'image/jpeg'  # fallback
        
        # Create a message with both text and image
        message_content = [
            {
                "type": "text",
                "text": message
            } if message.strip() else None,
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data
                }
            }
        ]
        # Remove None entries
        message_content = [m for m in message_content if m is not None]
    else:
        # Text-only message
        message_content = message
    
    try:
        # Handle the chat message with the appropriate content
        response = assistant.chat(message_content)
        
        # Get token usage from assistant
        token_usage = {
            'total_tokens': assistant.total_tokens_used,
            'max_tokens': Config.MAX_CONVERSATION_TOKENS
        }
        
        # Get the last used tool from the conversation history
        tool_name = None
        if assistant.conversation_history:
            for msg in reversed(assistant.conversation_history):
                if msg.get('role') == 'assistant' and msg.get('content'):
                    content = msg['content']
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'tool_use':
                                tool_name = block.get('name')
                                break
                    if tool_name:
                        break
        
        return jsonify({
            'response': response,
            'thinking': False,
            'tool_name': tool_name,
            'token_usage': token_usage
        })
        
    except Exception as e:
        return jsonify({
            'response': f"Error: {str(e)}",
            'thinking': False,
            'tool_name': None,
            'token_usage': None
        }), 200  # Return 200 even for errors to handle them gracefully in frontend

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Use EnhancedImageTool to process the image
            result = image_tool.execute(image_path=filepath)
            
            # The result is already in the correct format with proper media type
            image_data = result[0]['source']['data']
            media_type = result[0]['source']['media_type']
            
            # Clean up the file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'image_data': image_data,
                'media_type': media_type
            })
            
        except Exception as e:
            # Clean up the file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/reset', methods=['POST'])
def reset():
    # Reset the assistant's conversation history
    assistant.reset()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=False)