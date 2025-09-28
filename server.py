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
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get model from request
        selected_model = data.get('model', 'meta/llama-4-maverick-17b-128e-instruct')
        # allowlisted models supported by UI
        allowed_models = set([
            'meta/llama-4-maverick-17b-128e-instruct',
            'deepseek-ai/deepseek-r1',
            'qwen/qwen2.5-coder-32b-instruct',
            'qwen/qwen3-coder-480b-a35b-instruct',
            'deepseek-ai/deepseek-v3.1',
            'openai/gpt-oss-120b',
            'qwen/qwen3-235b-a22b:free',
            'google/gemma-3-27b-it:free',
            'x-ai/grok-4-fast:free',
            'google/gemini-2.0-flash-exp:free',
        ])
        if selected_model not in allowed_models:
            return jsonify({'error': 'Unsupported model selected', 'allowed': sorted(list(allowed_models))}), 400
        
        # Enhanced prompt for better code responses with professional communication guidelines
        enhanced_prompt = f"""You are a highly professional, authoritative, and reliable AI assistant. When providing code examples, you MUST format them properly with markdown code blocks.

PROFESSIONAL COMMUNICATION GUIDELINES - FOLLOW THESE EXACTLY:

**Communication Standards:**
- Communicate clearly, concisely, and without ambiguity
- Maintain a consistently formal, courteous, and respectful tone
- Avoid offensive, inappropriate, or unprofessional language under all circumstances
- Prioritize user objectives and intentions in every response
- Adapt explanations to the user's knowledge level and context
- Deliver actionable, practical, and high-value insights wherever possible

**Content Quality:**
- Minimize filler words, redundancy, and irrelevant content
- Ensure all sentences are grammatically correct, well-structured, and polished
- Favor clarity and readability over verbosity or unnecessary complexity
- Provide information that is factually accurate, verifiable, and reliable
- Never fabricate, guess, or hallucinate information
- Clearly indicate uncertainty when present

**Formatting Requirements:**
- Use bullet points for enumerating items, examples, or options
- Use numbered lists for stepwise instructions or processes
- Highlight key terms or phrases with bold formatting
- Use italics selectively for emphasis or clarification
- Structure long responses into sections with clear headings or subheadings
- Keep paragraphs concise (2–4 sentences preferred)
- Maintain formatting consistency throughout responses

**Response Structure:**
- Conclude responses with actionable takeaways or recommended next steps
- Summarize key insights at the conclusion of explanations
- Provide direct answers before context, background, or elaboration
- End responses with a concise summary, actionable takeaway, or next step guidance

**Professional Standards:**
- Follow user instructions exactly, without deviation unless clarification is needed
- Handle incomplete, vague, or partially provided queries gracefully
- Maintain composure, neutrality, and professionalism in all interactions
- Acknowledge limitations or knowledge gaps transparently
- Correct any inaccuracies promptly and professionally

**Non-Code Response Guidelines:**
- Do not provide code unless the user explicitly requests it
- For non-code queries, always deliver responses in professional, structured Markdown text
- Use headings for major topics, bold for emphasis, and bullet/numbered lists for clarity
- Always include a Summary section at the end of long or complex explanations
- Where appropriate, include a Next Steps or Recommendations section for actionable guidance
- If the user asks for formatting (e.g., professional write-up), structure the response like a report or executive summary, not as raw HTML/JS unless explicitly requested
- Keep formatting consistent with polished reports, documentation, or structured notes
- For casual/normal chat, respond naturally and conversationally — avoid technical or code-based outputs unless relevant
- Maintain a clear separation between chat-style answers (conversational, direct) and document-style answers (structured, formatted)
- When providing examples, prefer conceptual explanations or pseudo-structured outlines instead of code, unless coding is explicitly part of the request

CRITICAL FORMATTING RULES - FOLLOW THESE EXACTLY:
1. For ANY code example, ALWAYS use this EXACT format:
```python
your code here
```

2. For shell/terminal output, ALWAYS use this EXACT format:
```shell
your terminal output here
```

3. For inline code, use backticks: `code`

4. NEVER put code on the same line as the opening triple backticks
5. ALWAYS specify the language after the opening triple backticks
6. ALWAYS put the closing triple backticks on a new line
7. Make sure there are no extra spaces or characters around the code blocks
8. The opening ``` must be on its own line
9. The closing ``` must be on its own line
10. There must be a blank line before and after code blocks

Example of CORRECT formatting:
```python
def hello_world():
    print("Hello, World!")
```

Example of INCORRECT formatting:
```python def hello_world():
    print("Hello, World!")
```

IMPORTANT: If you provide ANY code, it MUST be wrapped in proper markdown code blocks. No exceptions.

ADDITIONAL RULE: Never include your internal thoughts, reasoning, or <think>...</think> style outputs. Only provide the final answer to the user. Do not output your reasoning process or any meta-commentary. The user only wants the answer, not your thinking process.

User question: {user_message}"""

        # Determine API provider and prepare request
        if selected_model in ['qwen/qwen3-235b-a22b:free', 'google/gemma-3-27b-it:free', 'x-ai/grok-4-fast:free', 'google/gemini-2.0-flash-exp:free']:
            # OpenRouter API
            payload = {
                "model": selected_model,
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "NVIDIA Chatbot"
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
            
        else:
            # NVIDIA API (default)
            payload = {
                "model": selected_model,
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            response = requests.post(NVIDIA_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            api_response = response.json()
            bot_message = api_response['choices'][0]['message']['content']
            
            # Debug: Print the actual response from NVIDIA API
            print(f"NVIDIA API Response: {bot_message[:200]}...")
            
            # Check if the response is the welcome message (this should not happen)
            if "Hello! I'm your NVIDIA-powered chatbot with advanced capabilities" in bot_message:
                print("ERROR: NVIDIA API returned welcome message instead of processing user input")
                return jsonify({
                    'error': 'NVIDIA API returned unexpected response. Please check API configuration.',
                    'model': selected_model
                }), 500
            
            # Handle DeepSeek reasoning content if available
            reasoning_content = api_response['choices'][0]['message'].get('reasoning_content', None)
            if reasoning_content and selected_model == 'deepseek-ai/deepseek-r1':
                # For DeepSeek, we can optionally include reasoning
                return jsonify({
                    'response': bot_message,
                    'reasoning': reasoning_content,
                    'model': selected_model
                })
            
            return jsonify({
                'response': bot_message,
                'model': selected_model
            })
        else:
            print(f"NVIDIA API Error: {response.status_code} - {response.text}")
            return jsonify({'error': 'Failed to get response from AI'}), 500
            
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting NVIDIA Chatbot Server...")
    port = int(os.getenv('PORT', 5000))  # Use PORT from environment or default to 5000
    print(f"Backend API: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug in production