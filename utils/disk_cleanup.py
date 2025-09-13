import os
import logging
import shutil
import json
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class DiskCleanup:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.cleanup_dirs = [
            'processed',
            'uploads',
            'summarization_cache',
            'translation_cache',
            'ocr_cache',
            'model_cache'
        ]
        
    def get_disk_usage(self) -> float:
        """Get disk usage percentage for the partition containing base_path"""
        usage = shutil.disk_usage(self.base_path)
        return (usage.used / usage.total) * 100
        
    def cleanup_old_files(self, days_threshold: int = 7) -> int:
        """Clean up files older than threshold days"""
        files_removed = 0
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        for cleanup_dir in self.cleanup_dirs:
            dir_path = self.base_path / cleanup_dir
            if not dir_path.exists():
                continue
                
            try:
                for file_path in dir_path.glob('**/*'):
                    if not file_path.is_file():
                        continue
                    
                    # Skip if file is too new
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime > threshold_date:
                        continue
                        
                    try:
                        file_path.unlink()
                        files_removed += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error cleaning directory {cleanup_dir}: {e}")
                
        return files_removed
        
    def emergency_cleanup(self, target_usage: float = 85.0) -> bool:
        """Perform emergency cleanup when disk usage is critical"""
        current_usage = self.get_disk_usage()
        if current_usage <= target_usage:
            return True
            
        # Try increasingly aggressive cleanup
        for days in [3, 1, 0]:
            files_removed = self.cleanup_old_files(days)
            logger.info(f"Emergency cleanup removed {files_removed} files older than {days} days")
            
            current_usage = self.get_disk_usage()
            if current_usage <= target_usage:
                return True
                
        return False
        
    def should_cleanup(self, threshold: float = 90.0) -> bool:
        """Check if cleanup is needed"""
        return self.get_disk_usage() > threshold
        
    def maintain_disk_space(self) -> None:
        """Main method to maintain disk space"""
        try:
            current_usage = self.get_disk_usage()
            logger.info(f"Current disk usage: {current_usage:.1f}%")
            
            if current_usage > 90.0:
                logger.warning(f"High disk usage detected: {current_usage:.1f}%")
                success = self.emergency_cleanup()
                if success:
                    logger.info("Emergency cleanup successful")
                else:
                    logger.error("Emergency cleanup failed to free enough space")
            elif current_usage > 80.0:
                # Regular cleanup
                files_removed = self.cleanup_old_files()
                logger.info(f"Regular cleanup removed {files_removed} files")
                
        except Exception as e:
            logger.error(f"Error in disk maintenance: {e}")