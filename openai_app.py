from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import openai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
if not openai.api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OpenAI API key is required")

def set_openai_params(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
):
    """Set OpenAI parameters"""
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }

def get_completion(params, prompt):
    """Get completion from OpenAI API"""
    try:
        logger.info(f"Making API call for prompt: {prompt[:50]}...")
        
        response = openai.chat.completions.create(
            model=params['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"]
        )
        
        logger.info("API call successful")
        return response.choices[0].message.content
        
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise Exception(f"OpenAI API error: {str(e)}")
    except openai.APIConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise Exception(f"Connection error: {str(e)}")
    except openai.RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise Exception(f"Rate limit exceeded: {str(e)}")
    except openai.AuthenticationError as e:
        logger.error(f"Authentication error: {e}")
        raise Exception(f"Authentication error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise Exception(f"Unexpected error: {str(e)}")

@app.route('/')
def index():
    """API information endpoint"""
    return jsonify({
        'name': 'OpenAI Chat API',
        'version': '1.0.0',
        'endpoints': {
            'chat': {
                'url': '/api/chat',
                'method': 'POST',
                'description': 'Send a prompt to OpenAI and get a response'
            },
            'models': {
                'url': '/api/models',
                'method': 'GET',
                'description': 'Get available models'
            },
            'health': {
                'url': '/api/health',
                'method': 'GET',
                'description': 'Check API health status'
            }
        }
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        if 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt cannot be empty'
            }), 400
        
        # Extract and validate parameters
        try:
            params = set_openai_params(
                model=data.get('model', 'gpt-3.5-turbo'),
                temperature=float(data.get('temperature', 0.7)),
                max_tokens=int(data.get('max_tokens', 256)),
                top_p=float(data.get('top_p', 1)),
                frequency_penalty=float(data.get('frequency_penalty', 0)),
                presence_penalty=float(data.get('presence_penalty', 0))
            )
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid parameter value: {str(e)}'
            }), 400
        
        # Validate parameter ranges
        if not (0 <= params['temperature'] <= 2):
            return jsonify({
                'success': False,
                'error': 'Temperature must be between 0 and 2'
            }), 400
        
        if not (1 <= params['max_tokens'] <= 4000):
            return jsonify({
                'success': False,
                'error': 'Max tokens must be between 1 and 4000'
            }), 400
        
        if not (0 <= params['top_p'] <= 1):
            return jsonify({
                'success': False,
                'error': 'Top_p must be between 0 and 1'
            }), 400
        
        # Get completion from OpenAI
        response_content = get_completion(params, prompt)
        
        return jsonify({
            'success': True,
            'response': response_content,
            'parameters_used': params,
            'prompt_length': len(prompt)
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available OpenAI models"""
    models = [
        {
            'id': 'gpt-3.5-turbo',
            'name': 'GPT-3.5 Turbo',
            'description': 'Fast and efficient model for most tasks'
        },
        {
            'id': 'gpt-4',
            'name': 'GPT-4',
            'description': 'Most capable model for complex tasks'
        },
        {
            'id': 'gpt-4-turbo',
            'name': 'GPT-4 Turbo',
            'description': 'Latest GPT-4 model with improved performance'
        }
    ]
    
    return jsonify({
        'success': True,
        'models': models
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test if we can make a minimal API call
        test_params = set_openai_params(model="gpt-3.5-turbo", max_tokens=1)
        test_response = openai.chat.completions.create(
            model=test_params['model'],
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1
        )
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'api_key_configured': True,
            'openai_connection': 'working',
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'api_key_configured': bool(openai.api_key),
            'openai_connection': 'failed',
            'error': str(e)
        }), 503

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/chat', '/api/models', '/api/health']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Check if API key is configured
    if not openai.api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    print("Starting OpenAI Chat API Server...")
    print(f"API Key configured: {openai.api_key[:8]}..." if openai.api_key else "No API key")
    print("API Base URL: http://localhost:5000")
    print("API Documentation: http://localhost:5000")
    print("Health Check: http://localhost:5000/api/health")

    gunicorn openai_app:app --bind=0.0.0.0:$PORT
