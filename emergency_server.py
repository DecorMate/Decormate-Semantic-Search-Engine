#!/usr/bin/env python3
"""
Emergency startup script for Railway
Tests basic functionality before full deployment
"""
import os
import sys
sys.path.append('src')

def test_basic_imports():
    """Test if basic imports work"""
    try:
        import torch
        import numpy
        import flask
        from flask import Flask
        import pinecone
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can start"""
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def home():
            return {'status': 'emergency_ok'}
            
        print("✅ Flask app creation successful")
        return app
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return None

def emergency_server():
    """Start emergency minimal server"""
    print("🚨 Starting EMERGENCY minimal server...")
    
    if not test_basic_imports():
        print("❌ Cannot start - basic imports failed")
        sys.exit(1)
    
    app = test_flask_app()
    if not app:
        print("❌ Cannot start - Flask app failed")
        sys.exit(1)
    
    # Add emergency routes
    @app.route('/ping')
    def ping():
        return "EMERGENCY_OK"
    
    @app.route('/status')
    def status():
        return {
            'status': 'emergency_mode',
            'message': 'Running minimal server due to memory constraints',
            'memory_limited': True
        }
    
    # Try to start the full app, fallback to emergency
    try:
        print("🔄 Attempting to import full routes...")
        from routes import app as full_app
        print("✅ Full app imported successfully")
        return full_app
    except Exception as e:
        print(f"⚠️ Full app failed: {e}")
        print("🚨 Using emergency minimal app")
        return app

if __name__ == '__main__':
    app = emergency_server()
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🚀 Starting emergency server on port {port}")
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)