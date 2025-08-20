#!/usr/bin/env python3
"""
Logging configuration utility to control verbosity levels.
Use this to reduce terminal output during processing.
"""

import logging
import os

def configure_logging(level="WARNING", log_to_file=True):
    """
    Configure logging levels to reduce terminal output.
    
    Args:
        level (str): Logging level - "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        log_to_file (bool): Whether to log to file (app.log)
    """
    # Convert string to logging level
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    
    # Configure root logger
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler("app.log"))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    # Reduce verbosity for external libraries
    external_loggers = [
        "moviepy", "PIL", "urllib3", "requests", "whisper", 
        "transformers", "torch", "numpy", "cv2", "ffmpeg",
        "pydub", "sklearn", "matplotlib", "seaborn"
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    print(f"Logging configured with level: {level}")
    print(f"External library logging reduced to WARNING level")
    print(f"Detailed logs will be saved to app.log" if log_to_file else "No file logging")

def set_quiet_mode():
    """Set logging to quiet mode (ERROR level only)"""
    configure_logging("ERROR")

def set_normal_mode():
    """Set logging to normal mode (WARNING level)"""
    configure_logging("WARNING")

def set_verbose_mode():
    """Set logging to verbose mode (INFO level)"""
    configure_logging("INFO")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        level = sys.argv[1].upper()
        configure_logging(level)
    else:
        print("Usage: python logging_config.py [DEBUG|INFO|WARNING|ERROR|CRITICAL]")
        print("Default: WARNING level")
        configure_logging("WARNING")
