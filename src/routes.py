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
            
            if not file or not file.filename:
                return jsonify({'error': 'No file provided'}), 400
            
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
                return jsonify({'error': 'No text provided'}), 400
            
            item_id = indexer.add_text(text, category, custom_id)
            return jsonify({'id': item_id})
        
        else:
            return jsonify({'error': 'Invalid request format'}), 400
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
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
        
        # Debug info
        print(f"Content-Type: {request.content_type}")
        print(f"Files: {list(request.files.keys())}")
        print(f"Form: {dict(request.form)}")
        
        # Handle image search
        if 'file' in request.files:
            file = request.files['file']
            limit = int(request.form.get('limit', 5))
            
            if not file or not file.filename:
                return jsonify({'error': 'No file provided'}), 400
            
            print(f"Searching with file: {file.filename}")
            
            # Save temp file
            filepath = f"temp/search_{file.filename}"
            os.makedirs('temp', exist_ok=True)
            file.save(filepath)
            
            try:
                # Search
                results = indexer.search(filepath, limit)
                return jsonify({'ids': [r.id for r in results]})
            finally:
                # Always cleanup temp file
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        # Handle text search
        elif request.is_json:
            data = request.get_json()
            query = data.get('query')
            limit = data.get('limit', 5)
            
            if not query:
                return jsonify({'error': 'No query provided'}), 400
            
            results = indexer.search(query, limit)
            return jsonify({'ids': [r.id for r in results]})
        
        else:
            return jsonify({
                'error': 'Invalid request format',
                'expected': 'multipart/form-data with file or application/json with query'
            }), 400
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)