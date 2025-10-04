#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Simple Flask server to handle NVIDIA API calls securely
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
from conversation_memory import get_memory_manager, cleanup_old_sessions

app = Flask(__name__)

# Configure CORS to allow requests from GitHub Pages
CORS(app, origins=[
    "https://antonjijo.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "https://Nvidia.pythonanywhere.com"  # Add PythonAnywhere domain for cross-origin requests
])

# API configuration - Get from environment variables
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Define allowed models set
ALLOWED_MODELS = {
    'meta/llama-4-maverick-17b-128e-instruct',
    'deepseek-ai/deepseek-r1',
    'qwen/qwen2.5-coder-32b-instruct',
    'qwen/qwen3-coder-480b-a35b-instruct',
    'deepseek-ai/deepseek-v3.1',
    'openai/gpt-oss-120b',
    'qwen/qwen3-235b-a22b:free',
    'google/gemma-3-27b-it:free',
}

# Validate API keys on startup
if not NVIDIA_API_KEY:
    print("WARNING: NVIDIA_API_KEY environment variable not set!")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set!")

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Check if API keys are available
        if not NVIDIA_API_KEY and not OPENROUTER_API_KEY:
            return jsonify({'error': 'API keys not configured. Please set environment variables.'}), 500
        
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')  # Get session ID for conversation persistence
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get model from request
        selected_model = data.get('model', 'meta/llama-4-maverick-17b-128e-instruct')
        if selected_model not in ALLOWED_MODELS:
            return jsonify({'error': 'Unsupported model', 'allowed': sorted(list(ALLOWED_MODELS))}), 400
        
        # Get conversation memory manager for this session
        memory_manager = get_memory_manager(session_id)
        memory_manager.set_model(selected_model)
        
        # Add user message to conversation history
        memory_manager.add_message('user', user_message)
        
        # Get conversation buffer for API call
        conversation_messages = memory_manager.get_conversation_buffer()

        # Determine API provider and prepare request
        if selected_model in ['qwen/qwen3-235b-a22b:free', 'google/gemma-3-27b-it:free']:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "NVIDIA Chatbot"
            }
            payload = {
                "model": selected_model,
                "messages": conversation_messages,
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        else:
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            payload = {
                "model": selected_model,
                "messages": conversation_messages,
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            response = requests.post(NVIDIA_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            api_response = response.json()
            bot_message = api_response['choices'][0]['message']['content']
            
            # Add assistant response to conversation memory
            memory_manager.add_message('assistant', bot_message)
            
            # Debug: Print the actual response from API
            print(f"API Response: {bot_message[:200]}...")
            
            # Check if the response is the welcome message (this should not happen)
            if "Hello! I'm your NVIDIA-powered chatbot with advanced capabilities" in bot_message:
                return jsonify({'error': 'NVIDIA API returned unexpected response. Please check API configuration.'}), 500
            
            reasoning_content = api_response['choices'][0]['message'].get('reasoning_content', None)
            response_data = {
                'response': bot_message,
                'model': selected_model,
                'conversation_stats': memory_manager.get_conversation_stats()
            }
            if reasoning_content and selected_model == 'deepseek-ai/deepseek-r1':
                response_data['reasoning'] = reasoning_content
            
            # Cleanup old sessions periodically
            cleanup_old_sessions()
            
            return jsonify(response_data)
        else:
            return jsonify({'error': f"API error {response.status_code}: {response.text}"}), 500
            
    except Exception as e:
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/conversation/stats', methods=['GET'])
def get_conversation_stats():
    """Get conversation statistics for a session."""
    session_id = request.args.get('session_id', 'default')
    memory_manager = get_memory_manager(session_id)
    return jsonify(memory_manager.get_conversation_stats())

@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    keep_system_prompt = data.get('keep_system_prompt', True)
    
    memory_manager = get_memory_manager(session_id)
    memory_manager.clear_conversation(keep_system_prompt)
    
    return jsonify({
        'success': True,
        'message': 'Conversation cleared',
        'stats': memory_manager.get_conversation_stats()
    })

@app.route('/api/conversation/export', methods=['GET'])
def export_conversation():
    """Export conversation data for debugging/persistence."""
    session_id = request.args.get('session_id', 'default')
    memory_manager = get_memory_manager(session_id)
    return jsonify(memory_manager.export_conversation())

if __name__ == '__main__':
    print("Starting NVIDIA Chatbot Server...")
    port = int(os.getenv('PORT', 5000))  # Use PORT from environment or default to 5000
    print(f"Backend API: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug in production