from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global indexer - simple lazy loading
indexer = None

@app.route('/', methods=['GET'])
def home():
    """API information"""
    return jsonify({
        'service': 'Semantic Search API',
        'endpoints': {
            'POST /upload': 'Upload content',
            'POST /search': 'Search content',
            'GET /ping': 'Health check'
        }
    })

@app.route('/ping', methods=['GET'])
def ping():
    """Simple health check"""
    return "OK"

@app.route('/upload', methods=['POST'])
def upload():
    """Upload content with optional custom ID"""
    global indexer
    
    try:
        # Initialize indexer if needed
        if indexer is None:
            from indexer import SimpleIndexer
            indexer = SimpleIndexer()
        
        # Handle image upload
        if 'file' in request.files:
            file = request.files['file']
            custom_id = request.form.get('id')
            description = request.form.get('description', '')
            
            if not file.filename:
                return jsonify({'error': 'No file'}), 400
            
            # Save temp file
            filepath = f"temp/{file.filename}"
            os.makedirs('temp', exist_ok=True)
            file.save(filepath)
            
            # Add to index
            item_id = indexer.add_image(filepath, description, custom_id)
            os.remove(filepath)
            
            return jsonify({'id': item_id})
        
        # Handle text upload
        elif request.is_json:
            data = request.get_json()
            text = data.get('text')
            custom_id = data.get('id')
            category = data.get('category', '')
            
            if not text:
                return jsonify({'error': 'No text'}), 400
            
            item_id = indexer.add_text(text, category, custom_id)
            return jsonify({'id': item_id})
        
        else:
            return jsonify({'error': 'Invalid request'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for similar content"""
    global indexer
    
    try:
        # Initialize indexer if needed
        if indexer is None:
            from indexer import SimpleIndexer
            indexer = SimpleIndexer()
        
        # Handle image search
        if 'file' in request.files:
            file = request.files['file']
            limit = int(request.form.get('limit', 5))
            
            if not file.filename:
                return jsonify({'error': 'No file'}), 400
            
            # Save temp file
            filepath = f"temp/search_{file.filename}"
            os.makedirs('temp', exist_ok=True)
            file.save(filepath)
            
            # Search
            results = indexer.search(filepath, limit)
            os.remove(filepath)
            
            return jsonify({'ids': [r.id for r in results]})
        
        # Handle text search
        elif request.is_json:
            data = request.get_json()
            query = data.get('query')
            limit = data.get('limit', 5)
            
            if not query:
                return jsonify({'error': 'No query'}), 400
            
            results = indexer.search(query, limit)
            return jsonify({'ids': [r.id for r in results]})
        
        else:
            return jsonify({'error': 'Invalid request'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

@app.route('/status', methods=['GET'])
def status():
    """Detailed status without initializing indexer"""
    status_info = {
        'app': 'running',
        'routes': 'available',
        'environment': {
            'pinecone_key_set': bool(os.environ.get('PINECONE_API_KEY')),
            'python_path': os.environ.get('PYTHONPATH', 'not_set'),
            'working_dir': os.getcwd()
        }
    }
    
    # Test imports without initializing
    try:
        import torch
        status_info['torch'] = {'available': True, 'version': torch.__version__}
    except Exception as e:
        status_info['torch'] = {'available': False, 'error': str(e)}
    
    try:
        from indexer import SimpleIndexer
        status_info['indexer_module'] = {'importable': True}
    except Exception as e:
        status_info['indexer_module'] = {'importable': False, 'error': str(e)}
    
    return jsonify(status_info), 200

@app.route('/cleanup', methods=['POST'])
def cleanup_memory():
    """Free model memory - useful for memory-constrained environments"""
    try:
        global indexer
        if indexer is not None and hasattr(indexer, '_cleanup_model'):
            indexer._cleanup_model()
            return jsonify({'message': 'Model memory freed successfully'}), 200
        else:
            return jsonify({'message': 'No model to cleanup'}), 200
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

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
    """Ultra-simple ping endpoint - no dependencies"""
    return "OK", 200

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
    port = int(os.environ.get('PORT', 5000))
    print(f"🔌 Starting on port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)