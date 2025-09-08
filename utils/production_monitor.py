"""
Production-Ready Monitoring and Error Handling System

This module provides comprehensive monitoring, error handling, and performance tracking
for the video summarization pipeline in production environments.

Features:
- Real-time performance monitoring
- Error tracking and alerting
- Resource usage monitoring
- Pipeline health checks
- Automatic recovery mechanisms
- Performance metrics and dashboards
- Production-ready logging and debugging

Author: Video Summarizer Team
Created: 2024
"""

import os
import time
import logging
import json
import psutil
import traceback
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
import threading
from contextlib import contextmanager
from functools import wraps
import signal
import sys

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .gpu_config import get_device, is_gpu_available, get_memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    disk_usage_percent: float
    active_processes: int
    pipeline_status: str
    error_count: int
    success_count: int
    average_processing_time: float

@dataclass
class ErrorInfo:
    """Error information for tracking."""
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    severity: str
    resolved: bool = False

@dataclass
class PipelineHealth:
    """Pipeline health status."""
    overall_health: str
    component_status: Dict[str, str]
    last_error: Optional[ErrorInfo]
    uptime: float
    performance_score: float

class ProductionMonitor:
    """Production monitoring and error handling system."""
    
    def __init__(self, config_dir: str = "monitoring"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.metrics_file = self.config_dir / "metrics.json"
        self.errors_file = self.config_dir / "errors.json"
        self.health_file = self.config_dir / "health.json"
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.error_history: List[ErrorInfo] = []
        self.component_status: Dict[str, str] = {}
        
        # Threading
        self.monitor_thread = None
        self.monitor_active = False
        self.lock = Lock()
        
        # Counters
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Prometheus metrics (if available)
        self.prometheus_metrics = {}
        self._initialize_prometheus()
        
        # Error thresholds
        self.error_thresholds = {
            'critical': 5,      # Critical errors per hour
            'warning': 20,      # Warning errors per hour
            'info': 100        # Info errors per hour
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'disk_usage_percent': 90.0,
            'processing_time': 300.0  # 5 minutes
        }
        
        # Initialize component status
        self._initialize_component_status()
        
        logger.info("ProductionMonitor initialized")
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics if available."""
        if PROMETHEUS_AVAILABLE:
            try:
                self.prometheus_metrics = {
                    'requests_total': Counter('video_summarizer_requests_total', 'Total requests'),
                    'request_duration': Histogram('video_summarizer_request_duration_seconds', 'Request duration'),
                    'errors_total': Counter('video_summarizer_errors_total', 'Total errors', ['error_type']),
                    'cpu_usage': Gauge('video_summarizer_cpu_usage_percent', 'CPU usage percentage'),
                    'memory_usage': Gauge('video_summarizer_memory_usage_percent', 'Memory usage percentage'),
                    'gpu_memory_usage': Gauge('video_summarizer_gpu_memory_usage_percent', 'GPU memory usage percentage'),
                    'pipeline_health': Gauge('video_summarizer_pipeline_health_score', 'Pipeline health score')
                }
                logger.info("Prometheus metrics initialized")
            except Exception as e:
                logger.warning(f"Prometheus initialization failed: {e}")
    
    def _initialize_component_status(self):
        """Initialize component status tracking."""
        components = [
            'transcription', 'keyframe_extraction', 'topic_analysis',
            'summarization', 'translation', 'tts', 'video_generation',
            'ocr', 'audio_visual_sync'
        ]
        
        for component in components:
            self.component_status[component] = 'unknown'
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring."""
        if self.monitor_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitor_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitor_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop."""
        while self.monitor_active:
            try:
                self._collect_metrics()
                self._check_health()
                self._cleanup_old_data()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self):
        """Collect system and application metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics
            gpu_memory_percent = 0.0
            if is_gpu_available():
                try:
                    gpu_memory = get_memory_usage()
                    gpu_memory_percent = gpu_memory.get('used_percent', 0.0)
                except Exception:
                    pass
            
            # Process metrics
            process = psutil.Process()
            active_processes = len(psutil.pids())
            
            # Pipeline metrics
            uptime = time.time() - self.start_time
            avg_processing_time = self._calculate_average_processing_time()
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_usage_percent=disk.percent,
                active_processes=active_processes,
                pipeline_status=self._get_pipeline_status(),
                error_count=self.error_count,
                success_count=self.success_count,
                average_processing_time=avg_processing_time
            )
            
            # Store metrics
            with self.lock:
                self.performance_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Check thresholds
            self._check_performance_thresholds(metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics."""
        if not self.prometheus_metrics:
            return
        
        try:
            self.prometheus_metrics['cpu_usage'].set(metrics.cpu_percent)
            self.prometheus_metrics['memory_usage'].set(metrics.memory_percent)
            self.prometheus_metrics['gpu_memory_usage'].set(metrics.gpu_memory_percent)
            self.prometheus_metrics['pipeline_health'].set(self._calculate_health_score())
        except Exception as e:
            logger.warning(f"Prometheus metrics update failed: {e}")
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and alert if needed."""
        alerts = []
        
        if metrics.cpu_percent > self.performance_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.performance_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_memory_percent > self.performance_thresholds['gpu_memory_percent']:
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > self.performance_thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if metrics.average_processing_time > self.performance_thresholds['processing_time']:
            alerts.append(f"Slow processing: {metrics.average_processing_time:.1f}s")
        
        for alert in alerts:
            self.log_error('performance_warning', alert, severity='warning')
    
    def _get_pipeline_status(self) -> str:
        """Get overall pipeline status."""
        if self.error_count > self.success_count * 0.1:  # More than 10% error rate
            return 'degraded'
        elif self.error_count > 0:
            return 'warning'
        else:
            return 'healthy'
    
    def _calculate_average_processing_time(self) -> float:
        """Calculate average processing time from recent metrics."""
        if not self.performance_history:
            return 0.0
        
        # Use last 10 metrics for average
        recent_metrics = self.performance_history[-10:]
        total_time = sum(m.average_processing_time for m in recent_metrics)
        return total_time / len(recent_metrics)
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        try:
            score = 100.0
            
            # Deduct points for errors
            error_rate = self.error_count / max(1, self.success_count + self.error_count)
            score -= error_rate * 50
            
            # Deduct points for performance issues
            if self.performance_history:
                latest = self.performance_history[-1]
                if latest.cpu_percent > 80:
                    score -= 10
                if latest.memory_percent > 85:
                    score -= 10
                if latest.gpu_memory_percent > 90:
                    score -= 10
            
            # Deduct points for component failures
            failed_components = sum(1 for status in self.component_status.values() if status == 'failed')
            score -= failed_components * 5
            
            return max(0.0, min(100.0, score))
            
        except Exception:
            return 50.0  # Default score if calculation fails
    
    def _check_health(self):
        """Check overall system health."""
        try:
            health_score = self._calculate_health_score()
            
            if health_score >= 90:
                overall_health = 'excellent'
            elif health_score >= 70:
                overall_health = 'good'
            elif health_score >= 50:
                overall_health = 'fair'
            else:
                overall_health = 'poor'
            
            health = PipelineHealth(
                overall_health=overall_health,
                component_status=self.component_status.copy(),
                last_error=self.error_history[-1] if self.error_history else None,
                uptime=time.time() - self.start_time,
                performance_score=health_score
            )
            
            # Save health status
            self._save_health_status(health)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _save_health_status(self, health: PipelineHealth):
        """Save health status to file."""
        try:
            health_data = {
                'timestamp': health.uptime,
                'overall_health': health.overall_health,
                'component_status': health.component_status,
                'performance_score': health.performance_score,
                'uptime': health.uptime,
                'last_error': asdict(health.last_error) if health.last_error else None
            }
            
            with open(self.health_file, 'w') as f:
                json.dump(health_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Health status save failed: {e}")
    
    def log_error(self, error_type: str, error_message: str, 
                  context: Dict[str, Any] = None, severity: str = 'error'):
        """Log an error with context."""
        try:
            error_info = ErrorInfo(
                timestamp=datetime.now(),
                error_type=error_type,
                error_message=error_message,
                stack_trace=traceback.format_exc(),
                context=context or {},
                severity=severity
            )
            
            with self.lock:
                self.error_history.append(error_info)
                self.error_count += 1
                
                # Keep only last 1000 errors
                if len(self.error_history) > 1000:
                    self.error_history = self.error_history[-1000:]
            
            # Update Prometheus metrics
            if self.prometheus_metrics:
                self.prometheus_metrics['errors_total'].labels(error_type=error_type).inc()
            
            # Log based on severity
            if severity == 'critical':
                logger.critical(f"[{error_type}] {error_message}")
            elif severity == 'warning':
                logger.warning(f"[{error_type}] {error_message}")
            else:
                logger.error(f"[{error_type}] {error_message}")
            
            # Save errors to file
            self._save_error_history()
            
        except Exception as e:
            logger.error(f"Error logging failed: {e}")
    
    def log_success(self, operation: str, duration: float = 0.0):
        """Log a successful operation."""
        try:
            with self.lock:
                self.success_count += 1
            
            # Update Prometheus metrics
            if self.prometheus_metrics:
                self.prometheus_metrics['requests_total'].inc()
                if 'request_duration' in self.prometheus_metrics:
                    self.prometheus_metrics['request_duration'].observe(duration)
            
            logger.info(f"Success: {operation} completed in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Success logging failed: {e}")
    
    def update_component_status(self, component: str, status: str):
        """Update status of a pipeline component."""
        try:
            valid_statuses = ['healthy', 'warning', 'failed', 'unknown']
            if status not in valid_statuses:
                logger.warning(f"Invalid component status: {status}")
                return
            
            with self.lock:
                self.component_status[component] = status
            
            logger.info(f"Component {component} status updated to {status}")
            
        except Exception as e:
            logger.error(f"Component status update failed: {e}")
    
    def _save_error_history(self):
        """Save error history to file."""
        try:
            error_data = [asdict(error) for error in self.error_history[-100:]]  # Last 100 errors
            
            # Convert datetime objects to strings
            for error in error_data:
                error['timestamp'] = error['timestamp'].isoformat()
            
            with open(self.errors_file, 'w') as f:
                json.dump(error_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error history save failed: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # Clean up old metrics
            with self.lock:
                self.performance_history = [
                    m for m in self.performance_history
                    if m.timestamp > cutoff_time
                ]
                
                self.error_history = [
                    e for e in self.error_history
                    if e.timestamp > cutoff_time
                ]
            
            logger.debug("Old monitoring data cleaned up")
            
        except Exception as e:
            logger.warning(f"Data cleanup failed: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        try:
            if not self.performance_history:
                return {}
            
            latest = self.performance_history[-1]
            
            return {
                'timestamp': latest.timestamp.isoformat(),
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'gpu_memory_percent': latest.gpu_memory_percent,
                'disk_usage_percent': latest.disk_usage_percent,
                'pipeline_status': latest.pipeline_status,
                'error_count': latest.error_count,
                'success_count': latest.success_count,
                'error_rate': latest.error_count / max(1, latest.success_count + latest.error_count),
                'uptime': time.time() - self.start_time,
                'health_score': self._calculate_health_score(),
                'component_status': self.component_status
            }
            
        except Exception as e:
            logger.error(f"Metrics summary failed: {e}")
            return {}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        try:
            if not self.error_history:
                return {'total_errors': 0, 'recent_errors': []}
            
            # Get errors from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_errors = [
                e for e in self.error_history
                if e.timestamp > cutoff_time
            ]
            
            # Group by error type
            error_types = {}
            for error in recent_errors:
                error_type = error.error_type
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors_24h': len(recent_errors),
                'error_types': error_types,
                'last_error': asdict(self.error_history[-1]) if self.error_history else None
            }
            
        except Exception as e:
            logger.error(f"Error summary failed: {e}")
            return {}
    
    def start_prometheus_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available")
            return
        
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Prometheus server start failed: {e}")

# Global instance
production_monitor = ProductionMonitor()

# Decorators for automatic monitoring
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                production_monitor.log_success(f"{operation_name}", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                production_monitor.log_error(
                    f"{operation_name}_error",
                    str(e),
                    {'duration': duration, 'args': str(args)[:200]},
                    'error'
                )
                raise
        return wrapper
    return decorator

@contextmanager
def monitor_operation(operation_name: str):
    """Context manager for monitoring operations."""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        production_monitor.log_success(f"{operation_name}", duration)
    except Exception as e:
        duration = time.time() - start_time
        production_monitor.log_error(
            f"{operation_name}_error",
            str(e),
            {'duration': duration},
            'error'
        )
        raise

def get_monitoring_stats() -> Dict[str, Any]:
    """Get comprehensive monitoring statistics."""
    return {
        'metrics': production_monitor.get_metrics_summary(),
        'errors': production_monitor.get_error_summary(),
        'health_score': production_monitor._calculate_health_score(),
        'uptime': time.time() - production_monitor.start_time
    }
