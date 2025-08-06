""
WSGI entry point for the Video Summarization Web Application.

This module provides a WSGI callable for production servers like Gunicorn or uWSGI.
"""
import os
from video_summarizer import create_app

# Create the application instance
app = create_app(os.getenv('FLASK_ENV') or 'production')

if __name__ == "__main__":
    # This block is only used when running with `python wsgi.py`
    # For production, use a production WSGI server like Gunicorn or uWSGI
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
