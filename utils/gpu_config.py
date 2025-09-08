"""
GPU Configuration and Management Utility

This module provides GPU detection, configuration, and optimization utilities
for the video summarizer project. It handles CUDA availability, memory management,
and device selection for various ML models.

Author: Video Summarizer Team
Created: 2024
"""

import os
import logging
import torch
import gc
from typing import Dict, List, Optional, Tuple, Any
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUConfig:
    """GPU configuration and management class."""
    
    def __init__(self):
        self.device = self._detect_device()
        self.gpu_info = self._get_gpu_info()
        self.memory_info = self._get_memory_info()
        self._configure_torch()
        
    def _detect_device(self) -> str:
        """Detect the best available device (GPU or CPU)."""
        try:
            if torch.cuda.is_available():
                # Test CUDA capability
                test_tensor = torch.tensor([1.0], device='cuda')
                del test_tensor
                device = 'cuda'
                torch.cuda.empty_cache()
                logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                # Set environment variable to enable GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                return device
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {str(e)}")
        
        logger.warning("CUDA is not available or failed to initialize. Using CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return 'cpu'
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get detailed GPU information."""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'memory_total': None,
            'memory_allocated': None,
            'memory_cached': None
        }
        
        if torch.cuda.is_available():
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
            gpu_info['memory_allocated'] = torch.cuda.memory_allocated(0)
            gpu_info['memory_cached'] = torch.cuda.memory_reserved(0)
        
        return gpu_info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent
        }
    
    def _configure_torch(self):
        """Configure PyTorch for optimal GPU usage."""
        if self.device == 'cuda':
            # Enable cuDNN benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable cuDNN deterministic mode for reproducibility
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction to avoid OOM errors
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            logger.info("PyTorch configured for GPU optimization")
    
    def get_device(self) -> str:
        """Get the current device."""
        return self.device
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.device == 'cuda'
    
    def get_optimal_batch_size(self, model_name: str = None) -> int:
        """Get optimal batch size based on available memory."""
        if not self.is_gpu_available():
            return 1
            
    def get_memory_usage(self) -> Dict[str, Dict[str, float]]:
        """Get GPU and CPU memory usage.
        
        Returns:
            Dict containing GPU and CPU memory usage in GB
        """
        memory_info = {}
        
        # Get CPU memory usage
        cpu = psutil.Process()
        cpu_memory = cpu.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
        total_cpu_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB
        memory_info['cpu'] = {
            'used': round(cpu_memory, 2),
            'total': round(total_cpu_memory, 2)
        }
        
        # Get GPU memory usage if available
        if self.is_gpu_available():
            gpu_memory = torch.cuda.memory_allocated(0) / (1024 * 1024 * 1024)  # Convert to GB
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
            memory_info['gpu'] = {
                'used': round(gpu_memory, 2),
                'total': round(total_gpu_memory, 2)
            }
        else:
            memory_info['gpu'] = {'used': 0, 'total': 0}
        
        return memory_info
        # Get available GPU memory
        memory_available = self.gpu_info['memory_total'] - self.gpu_info['memory_allocated']
        
        # Estimate batch size based on model and available memory
        if model_name and 'whisper' in model_name.lower():
            if 'large' in model_name.lower():
                return min(4, max(1, memory_available // (2 * 1024**3)))  # 2GB per batch
            elif 'medium' in model_name.lower():
                return min(8, max(1, memory_available // (1 * 1024**3)))  # 1GB per batch
            else:
                return min(16, max(1, memory_available // (512 * 1024**2)))  # 512MB per batch
        else:
            # Default batch size for other models
            return min(8, max(1, memory_available // (1 * 1024**3)))
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.is_gpu_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        if not self.is_gpu_available():
            return {'gpu_allocated': 0, 'gpu_cached': 0, 'cpu_used': self.memory_info['used'] / 1024**2}
        
        return {
            'gpu_allocated': torch.cuda.memory_allocated(0) / 1024**2,
            'gpu_cached': torch.cuda.memory_reserved(0) / 1024**2,
            'cpu_used': self.memory_info['used'] / 1024**2
        }
    
    def optimize_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get optimization settings for specific models."""
        optimizations = {
            'device': self.device,
            'batch_size': self.get_optimal_batch_size(model_name),
            'fp16': self.is_gpu_available(),  # Use mixed precision on GPU
            'torch_compile': False,  # Disable for now due to compatibility issues
        }
        
        # Model-specific optimizations
        if 'whisper' in model_name.lower():
            optimizations.update({
                'beam_size': 5 if self.is_gpu_available() else 1,
                'best_of': 5 if self.is_gpu_available() else 1,
                'temperature': 0.0,
                'compression_ratio_threshold': 2.4,
                'log_prob_threshold': -1.0,
                'no_speech_threshold': 0.6,
                'condition_on_previous_text': True,
            })
        
        elif 'bart' in model_name.lower() or 't5' in model_name.lower():
            optimizations.update({
                'num_beams': 4 if self.is_gpu_available() else 1,
                'early_stopping': True,
                'max_length': 512,
                'min_length': 50,
            })
        
        elif 'm2m100' in model_name.lower() or 'marian' in model_name.lower():
            optimizations.update({
                'num_beams': 5 if self.is_gpu_available() else 1,
                'early_stopping': True,
                'max_length': 512,
            })
        
        return optimizations
    
    def log_status(self):
        """Log current GPU and memory status."""
        logger.info(f"Device: {self.device}")
        if self.is_gpu_available():
            logger.info(f"GPU: {self.gpu_info['device_name']}")
            logger.info(f"GPU Memory: {self.gpu_info['memory_allocated'] / 1024**3:.2f}GB / {self.gpu_info['memory_total'] / 1024**3:.2f}GB")
        logger.info(f"CPU Memory: {self.memory_info['used'] / 1024**3:.2f}GB / {self.memory_info['total'] / 1024**3:.2f}GB")

# Global GPU configuration instance
gpu_config = GPUConfig()

def get_device() -> str:
    """Get the current device (cuda or cpu)."""
    return gpu_config.get_device()

def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return gpu_config.is_gpu_available()

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gpu_config.clear_gpu_memory()

def get_optimal_batch_size(model_name: str = None) -> int:
    """Get optimal batch size for the given model."""
    return gpu_config.get_optimal_batch_size(model_name)

def get_memory_usage() -> Dict[str, Dict[str, float]]:
    """Get GPU and CPU memory usage."""
    return gpu_config.get_memory_usage()
    return gpu_config.get_optimal_batch_size(model_name)

def optimize_for_model(model_name: str) -> Dict[str, Any]:
    """Get optimization settings for specific models."""
    return gpu_config.optimize_for_model(model_name)

def log_gpu_status():
    """Log current GPU and memory status."""
    gpu_config.log_status()

# Environment variable overrides
def force_gpu():
    """Force GPU usage (will raise error if not available)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot force GPU usage.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.info("Forced GPU usage")

def force_cpu():
    """Force CPU usage."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.info("Forced CPU usage")

def set_memory_fraction(fraction: float):
    """Set GPU memory fraction."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"Set GPU memory fraction to {fraction}")
