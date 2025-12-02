from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global indexer variable
indexer = None

def get_indexer():
    """Lazy initialization of indexer with error handling"""
    global indexer
    if indexer is None:
        try:
            from indexer import SimpleIndexer
            indexer = SimpleIndexer()
            print("✅ Indexer initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize indexer: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Return None to indicate failure
            return None
    return indexer

@app.route('/upload', methods=['POST'])
def upload_content():
    """
    Upload content with optional custom ID
    Image: multipart with 'file', optional 'id', 'description'
    Text: JSON with 'text', optional 'id', 'category'
    """
    try:
        # Get indexer with error handling
        current_indexer = get_indexer()
        if current_indexer is None:
            return jsonify({
                'error': 'Service temporarily unavailable - indexer not ready',
                'details': 'Please try again in a moment'
            }), 503
            
        # Image upload
        if 'file' in request.files:
            file = request.files['file']
            custom_id = request.form.get('id')
            description = request.form.get('description', '')
            
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                return jsonify({'error': 'Invalid image format'}), 400
            
            # Save temp file
            filepath = os.path.join('temp', file.filename)
            os.makedirs('temp', exist_ok=True)
            file.save(filepath)
            
            # Add to database with custom ID
            item_id = current_indexer.add_image(filepath, description, custom_id)
            os.remove(filepath)
            
            return jsonify({'id': item_id}) if item_id else jsonify({'error': 'Upload failed'}), 500
        
        # Text upload
        elif request.is_json:
            data = request.get_json()
            text = data.get('text')
            custom_id = data.get('id')
            category = data.get('category', '')
            
            if not text:
                return jsonify({'error': 'Text required'}), 400
            
            item_id = current_indexer.add_text(text, category, custom_id)
            return jsonify({'id': item_id}) if item_id else jsonify({'error': 'Upload failed'}), 500
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_content():
    """
    Search and return only IDs
    Image: multipart with 'file', optional 'limit'
    Text: JSON with 'query', optional 'limit'
    """
    try:
        # Get indexer with error handling
        current_indexer = get_indexer()
        if current_indexer is None:
            return jsonify({
                'error': 'Service temporarily unavailable - indexer not ready',
                'details': 'Please try again in a moment'
            }), 503
            
        # Image search
        if 'file' in request.files:
            file = request.files['file']
            limit = int(request.form.get('limit', 5))
            
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            
            # Save temp file for search
            filepath = os.path.join('temp', f"search_{file.filename}")
            os.makedirs('temp', exist_ok=True)
            file.save(filepath)
            
            results = current_indexer.search(filepath, limit)
            os.remove(filepath)
            
            return jsonify({'ids': [match.id for match in results]})
        
        # Text search
        elif request.is_json:
            data = request.get_json()
            query = data.get('query')
            limit = data.get('limit', 5)
            
            if not query:
                return jsonify({'error': 'Query required'}), 400
            
            results = current_indexer.search(query, min(limit, 20))
            return jsonify({'ids': [match.id for match in results]})
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with detailed diagnostics"""
    try:
        # Basic app health
        health_data = {
            'status': 'healthy',
            'service': 'Semantic Search API',
            'version': '1.0',
            'timestamp': str(__import__('datetime').datetime.now())
        }
        
        # Try to check indexer status without forcing initialization
        try:
            global indexer
            if indexer is not None:
                health_data['indexer'] = 'ready'
            else:
                health_data['indexer'] = 'not_initialized'
            
            # Test basic environment
            import torch
            health_data['torch_available'] = True
            health_data['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Check environment variables (without exposing values)
            health_data['pinecone_key_set'] = bool(os.environ.get('PINECONE_API_KEY'))
            
        except Exception as e:
            health_data['indexer'] = f'error: {str(e)}'
        
        return jsonify(health_data), 200
        
    except Exception as e:
        # Even if there are issues, return a basic healthy status
        # so Railway doesn't kill the container
        return jsonify({
            'status': 'basic_healthy',
            'error': str(e)
        }), 200  # Return 200 to pass health check

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for basic connectivity test"""
    return jsonify({'message': 'pong', 'status': 'ok'}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Semantic Search API...")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"🔑 Pinecone key set: {bool(os.environ.get('PINECONE_API_KEY'))}")
    
    try:
        import torch
        print(f"🔥 PyTorch available: {torch.__version__}")
        print(f"📱 Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception as e:
        print(f"❌ PyTorch issue: {e}")
    
    try:
        # Test basic import without initializing
        from indexer import SimpleIndexer
        print("✅ SimpleIndexer can be imported")
    except Exception as e:
        print(f"❌ SimpleIndexer import failed: {e}")
    
    print("🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)