#!/usr/bin/env python3
"""
Script to run the application with reduced terminal output.
This will significantly speed up processing by reducing logging overhead.
"""

import os
import sys
import logging

def setup_quiet_logging():
    """Configure logging for minimal terminal output"""
    # Set logging to WARNING level to reduce output
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
    )
    
    # Reduce verbosity for external libraries
    external_loggers = [
        "moviepy", "PIL", "urllib3", "requests", "whisper", 
        "transformers", "torch", "numpy", "cv2", "ffmpeg",
        "pydub", "sklearn", "matplotlib", "seaborn", "flask"
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    print("🚀 Starting application with reduced logging...")
    print("📝 Detailed logs will be saved to app.log")
    print("🔇 Terminal output minimized for faster processing")

if __name__ == "__main__":
    setup_quiet_logging()
    
    # Import and run the main application
    try:
        from app import app, CONFIG
        host = CONFIG.get("HOST", "0.0.0.0")
        port = CONFIG.get("PORT", 5000)
        debug = CONFIG.get("DEBUG", False)
        
        print(f"🌐 Server starting on http://{host}:{port}")
        print("⏱️  Processing will be much faster with reduced logging")
        
        app.run(host=host, port=port, debug=False, use_reloader=False)
        
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
