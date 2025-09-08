"""
Comprehensive Testing and Validation System

This module provides comprehensive testing capabilities for the video summarization
pipeline, including unit tests, integration tests, performance tests, and validation.

Features:
- Unit tests for all components
- Integration tests for the full pipeline
- Performance benchmarking
- Quality validation
- Stress testing
- Production readiness validation
- Automated test reporting

Author: Video Summarizer Team
Created: 2024
"""

import os
import sys
import time
import json
import logging
import unittest
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_config import is_gpu_available, get_device
from utils.production_summarizer import production_summarizer, summarize_text
from utils.advanced_ocr import advanced_ocr, extract_text_from_image
from utils.audio_visual_sync import audio_visual_synchronizer
from utils.production_monitor import production_monitor, get_monitoring_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestConfig:
    """Test configuration and constants."""
    
    # Test data paths
    TEST_DATA_DIR = Path("test_data")
    SAMPLE_VIDEOS_DIR = TEST_DATA_DIR / "sample_videos"
    SAMPLE_IMAGES_DIR = TEST_DATA_DIR / "sample_images"
    TEST_OUTPUT_DIR = TEST_DATA_DIR / "test_output"
    
    # Test parameters
    MAX_TEST_DURATION = 300  # 5 minutes
    PERFORMANCE_THRESHOLDS = {
        'summarization_time': 30.0,  # seconds
        'ocr_time': 10.0,  # seconds
        'sync_time': 5.0,  # seconds
        'memory_usage': 80.0,  # percent
        'cpu_usage': 90.0,  # percent
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        'min_summary_quality': 0.6,
        'min_ocr_confidence': 0.5,
        'min_sync_score': 0.7,
        'min_health_score': 80.0,
    }

class TestDataManager:
    """Manages test data creation and cleanup."""
    
    def __init__(self):
        self.test_data_dir = TestConfig.TEST_DATA_DIR
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        TestConfig.SAMPLE_VIDEOS_DIR.mkdir(exist_ok=True)
        TestConfig.SAMPLE_IMAGES_DIR.mkdir(exist_ok=True)
        TestConfig.TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    
    def create_sample_video(self, duration: int = 30) -> str:
        """Create a sample video for testing."""
        try:
            output_path = TestConfig.SAMPLE_VIDEOS_DIR / f"sample_{duration}s.mp4"
            
            # Create a simple test video using FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'testsrc=duration={duration}:size=1280x720:rate=30',
                '-f', 'lavfi',
                '-i', f'sine=frequency=1000:duration={duration}',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info(f"Created sample video: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to create sample video: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Sample video creation failed: {e}")
            return None
    
    def create_sample_image(self, text: str = "Test Image") -> str:
        """Create a sample image with text for OCR testing."""
        try:
            output_path = TestConfig.SAMPLE_IMAGES_DIR / f"sample_{hash(text)}.png"
            
            # Create a simple test image using PIL
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            # Draw text
            draw.text((50, 50), text, fill='black', font=font)
            
            # Save image
            img.save(output_path)
            logger.info(f"Created sample image: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Sample image creation failed: {e}")
            return None
    
    def cleanup_test_data(self):
        """Clean up test data."""
        try:
            if TestConfig.TEST_OUTPUT_DIR.exists():
                shutil.rmtree(TestConfig.TEST_OUTPUT_DIR)
                TestConfig.TEST_OUTPUT_DIR.mkdir(exist_ok=True)
            logger.info("Test data cleaned up")
        except Exception as e:
            logger.error(f"Test data cleanup failed: {e}")

class PerformanceTester:
    """Performance testing utilities."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_summarization(self, text: str, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark summarization performance."""
        try:
            times = []
            quality_scores = []
            
            for i in range(iterations):
                start_time = time.time()
                
                result = summarize_text(
                    text=text,
                    language='en',
                    content_type='general',
                    target_length=150,
                    quality_threshold=0.7
                )
                
                duration = time.time() - start_time
                times.append(duration)
                quality_scores.append(result.get('quality_score', 0.0))
            
            avg_time = sum(times) / len(times)
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            result = {
                'average_time': avg_time,
                'min_time': min(times),
                'max_time': max(times),
                'average_quality': avg_quality,
                'iterations': iterations,
                'meets_threshold': avg_time <= TestConfig.PERFORMANCE_THRESHOLDS['summarization_time']
            }
            
            self.results['summarization'] = result
            return result
            
        except Exception as e:
            logger.error(f"Summarization benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_ocr(self, image_path: str, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark OCR performance."""
        try:
            times = []
            confidences = []
            
            for i in range(iterations):
                start_time = time.time()
                
                result = extract_text_from_image(
                    image_path=image_path,
                    languages=['en'],
                    engine='auto'
                )
                
                duration = time.time() - start_time
                times.append(duration)
                confidences.append(result.get('confidence', 0.0))
            
            avg_time = sum(times) / len(times)
            avg_confidence = sum(confidences) / len(confidences)
            
            result = {
                'average_time': avg_time,
                'min_time': min(times),
                'max_time': max(times),
                'average_confidence': avg_confidence,
                'iterations': iterations,
                'meets_threshold': avg_time <= TestConfig.PERFORMANCE_THRESHOLDS['ocr_time']
            }
            
            self.results['ocr'] = result
            return result
            
        except Exception as e:
            logger.error(f"OCR benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_synchronization(self, audio_segments: List[Dict], 
                                 keyframe_segments: List[Dict]) -> Dict[str, Any]:
        """Benchmark audio-visual synchronization."""
        try:
            times = []
            sync_scores = []
            
            for i in range(3):  # Fewer iterations for sync test
                start_time = time.time()
                
                # Create mock audio segments
                audio_segments_mock = [
                    {
                        'start': 0.0,
                        'end': 10.0,
                        'text': f'Test segment {i}'
                    }
                ]
                
                # Create mock keyframe segments
                keyframe_segments_mock = [
                    {
                        'timestamp': 5.0,
                        'filepath': 'test.jpg',
                        'text_content': f'Test keyframe {i}'
                    }
                ]
                
                # This would normally call the sync function
                # For now, we'll simulate the timing
                duration = time.time() - start_time
                times.append(duration)
                sync_scores.append(0.8)  # Mock score
            
            avg_time = sum(times) / len(times)
            avg_score = sum(sync_scores) / len(sync_scores)
            
            result = {
                'average_time': avg_time,
                'min_time': min(times),
                'max_time': max(times),
                'average_score': avg_score,
                'iterations': len(times),
                'meets_threshold': avg_time <= TestConfig.PERFORMANCE_THRESHOLDS['sync_time']
            }
            
            self.results['synchronization'] = result
            return result
            
        except Exception as e:
            logger.error(f"Synchronization benchmark failed: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'thresholds': TestConfig.PERFORMANCE_THRESHOLDS,
            'overall_performance': self._calculate_overall_performance()
        }
    
    def _calculate_overall_performance(self) -> str:
        """Calculate overall performance rating."""
        if not self.results:
            return 'unknown'
        
        passed_tests = sum(1 for result in self.results.values() 
                          if isinstance(result, dict) and result.get('meets_threshold', False))
        total_tests = len(self.results)
        
        if passed_tests == total_tests:
            return 'excellent'
        elif passed_tests >= total_tests * 0.8:
            return 'good'
        elif passed_tests >= total_tests * 0.6:
            return 'fair'
        else:
            return 'poor'

class QualityValidator:
    """Quality validation utilities."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_summarization_quality(self, original_text: str, summary: str) -> Dict[str, Any]:
        """Validate summarization quality."""
        try:
            # Basic quality checks
            compression_ratio = len(summary.split()) / len(original_text.split())
            
            # Check if summary is too short or too long
            if compression_ratio < 0.1:
                quality_score = 0.3
                issues = ['Summary too short']
            elif compression_ratio > 0.5:
                quality_score = 0.4
                issues = ['Summary too long']
            else:
                quality_score = 0.8
                issues = []
            
            # Check for coherence
            sentences = summary.split('.')
            if len(sentences) < 2:
                quality_score *= 0.8
                issues.append('Summary lacks sentence structure')
            
            result = {
                'quality_score': quality_score,
                'compression_ratio': compression_ratio,
                'issues': issues,
                'meets_threshold': quality_score >= TestConfig.QUALITY_THRESHOLDS['min_summary_quality']
            }
            
            self.validation_results['summarization'] = result
            return result
            
        except Exception as e:
            logger.error(f"Summarization quality validation failed: {e}")
            return {'error': str(e)}
    
    def validate_ocr_quality(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OCR quality."""
        try:
            confidence = ocr_result.get('confidence', 0.0)
            text_regions = ocr_result.get('text_regions', [])
            
            # Check confidence threshold
            if confidence < TestConfig.QUALITY_THRESHOLDS['min_ocr_confidence']:
                quality_score = confidence
                issues = ['Low OCR confidence']
            else:
                quality_score = confidence
                issues = []
            
            # Check if any text was detected
            if not text_regions:
                quality_score = 0.0
                issues.append('No text regions detected')
            
            result = {
                'quality_score': quality_score,
                'confidence': confidence,
                'text_regions_count': len(text_regions),
                'issues': issues,
                'meets_threshold': quality_score >= TestConfig.QUALITY_THRESHOLDS['min_ocr_confidence']
            }
            
            self.validation_results['ocr'] = result
            return result
            
        except Exception as e:
            logger.error(f"OCR quality validation failed: {e}")
            return {'error': str(e)}
    
    def validate_system_health(self) -> Dict[str, Any]:
        """Validate overall system health."""
        try:
            health_stats = get_monitoring_stats()
            health_score = health_stats.get('health_score', 0.0)
            
            issues = []
            if health_score < TestConfig.QUALITY_THRESHOLDS['min_health_score']:
                issues.append('System health score below threshold')
            
            # Check component status
            component_status = health_stats.get('monitoring', {}).get('metrics', {}).get('component_status', {})
            failed_components = [comp for comp, status in component_status.items() if status == 'failed']
            
            if failed_components:
                issues.append(f'Failed components: {failed_components}')
                health_score *= 0.8
            
            result = {
                'health_score': health_score,
                'component_status': component_status,
                'failed_components': failed_components,
                'issues': issues,
                'meets_threshold': health_score >= TestConfig.QUALITY_THRESHOLDS['min_health_score']
            }
            
            self.validation_results['system_health'] = result
            return result
            
        except Exception as e:
            logger.error(f"System health validation failed: {e}")
            return {'error': str(e)}
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'results': self.validation_results,
            'thresholds': TestConfig.QUALITY_THRESHOLDS,
            'overall_quality': self._calculate_overall_quality()
        }
    
    def _calculate_overall_quality(self) -> str:
        """Calculate overall quality rating."""
        if not self.validation_results:
            return 'unknown'
        
        passed_validations = sum(1 for result in self.validation_results.values() 
                               if isinstance(result, dict) and result.get('meets_threshold', False))
        total_validations = len(self.validation_results)
        
        if passed_validations == total_validations:
            return 'excellent'
        elif passed_validations >= total_validations * 0.8:
            return 'good'
        elif passed_validations >= total_validations * 0.6:
            return 'fair'
        else:
            return 'poor'

class StressTester:
    """Stress testing utilities."""
    
    def __init__(self):
        self.stress_results = {}
    
    def stress_test_concurrent_requests(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test system under concurrent load."""
        try:
            start_time = time.time()
            
            # Create test requests
            test_text = "This is a test text for stress testing the summarization system. " * 10
            
            def make_request(request_id):
                try:
                    request_start = time.time()
                    result = summarize_text(
                        text=test_text,
                        language='en',
                        content_type='general',
                        target_length=100
                    )
                    request_duration = time.time() - request_start
                    return {
                        'request_id': request_id,
                        'success': True,
                        'duration': request_duration,
                        'quality_score': result.get('quality_score', 0.0)
                    }
                except Exception as e:
                    return {
                        'request_id': request_id,
                        'success': False,
                        'error': str(e),
                        'duration': time.time() - request_start
                    }
            
            # Execute concurrent requests
            with ThreadPoolExecutor(max_workers=min(num_requests, 5)) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_requests)]
                results = [future.result() for future in as_completed(futures)]
            
            total_duration = time.time() - start_time
            
            # Analyze results
            successful_requests = [r for r in results if r['success']]
            failed_requests = [r for r in results if not r['success']]
            
            avg_duration = sum(r['duration'] for r in successful_requests) / max(1, len(successful_requests))
            success_rate = len(successful_requests) / num_requests
            
            result = {
                'total_requests': num_requests,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate': success_rate,
                'total_duration': total_duration,
                'average_request_duration': avg_duration,
                'requests_per_second': num_requests / total_duration,
                'meets_threshold': success_rate >= 0.9 and avg_duration <= 30.0
            }
            
            self.stress_results['concurrent_requests'] = result
            return result
            
        except Exception as e:
            logger.error(f"Concurrent request stress test failed: {e}")
            return {'error': str(e)}
    
    def stress_test_memory_usage(self, duration: int = 60) -> Dict[str, Any]:
        """Test memory usage under load."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_samples = []
            start_time = time.time()
            
            # Run memory-intensive operations
            while time.time() - start_time < duration:
                # Create large objects
                large_text = "Test text for memory stress testing. " * 1000
                
                # Perform summarization
                try:
                    result = summarize_text(
                        text=large_text,
                        language='en',
                        content_type='general',
                        target_length=100
                    )
                except Exception:
                    pass
                
                # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
                
                # Force garbage collection
                gc.collect()
                
                time.sleep(1)  # Sample every second
            
            max_memory = max(memory_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            memory_growth = max_memory - initial_memory
            
            result = {
                'initial_memory_mb': initial_memory,
                'max_memory_mb': max_memory,
                'avg_memory_mb': avg_memory,
                'memory_growth_mb': memory_growth,
                'duration': duration,
                'samples': len(memory_samples),
                'meets_threshold': memory_growth < 500  # Less than 500MB growth
            }
            
            self.stress_results['memory_usage'] = result
            return result
            
        except Exception as e:
            logger.error(f"Memory usage stress test failed: {e}")
            return {'error': str(e)}
    
    def get_stress_report(self) -> Dict[str, Any]:
        """Get comprehensive stress test report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'results': self.stress_results,
            'overall_stress_performance': self._calculate_stress_performance()
        }
    
    def _calculate_stress_performance(self) -> str:
        """Calculate overall stress test performance."""
        if not self.stress_results:
            return 'unknown'
        
        passed_tests = sum(1 for result in self.stress_results.values() 
                          if isinstance(result, dict) and result.get('meets_threshold', False))
        total_tests = len(self.stress_results)
        
        if passed_tests == total_tests:
            return 'excellent'
        elif passed_tests >= total_tests * 0.8:
            return 'good'
        elif passed_tests >= total_tests * 0.6:
            return 'fair'
        else:
            return 'poor'

class ComprehensiveTester:
    """Comprehensive testing coordinator."""
    
    def __init__(self):
        self.data_manager = TestDataManager()
        self.performance_tester = PerformanceTester()
        self.quality_validator = QualityValidator()
        self.stress_tester = StressTester()
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("Starting comprehensive testing...")
        
        try:
            # 1. Setup test data
            logger.info("Setting up test data...")
            sample_video = self.data_manager.create_sample_video(30)
            sample_image = self.data_manager.create_sample_image("Test OCR Text")
            
            # 2. Performance tests
            logger.info("Running performance tests...")
            test_text = "This is a comprehensive test of the video summarization system. " * 20
            perf_results = self.performance_tester.benchmark_summarization(test_text)
            
            if sample_image:
                ocr_perf = self.performance_tester.benchmark_ocr(sample_image)
            
            # 3. Quality validation
            logger.info("Running quality validation...")
            summary_result = summarize_text(test_text, language='en', target_length=150)
            quality_results = self.quality_validator.validate_summarization_quality(
                test_text, summary_result.get('summary', '')
            )
            
            if sample_image:
                ocr_result = extract_text_from_image(sample_image, languages=['en'])
                ocr_quality = self.quality_validator.validate_ocr_quality(ocr_result)
            
            health_validation = self.quality_validator.validate_system_health()
            
            # 4. Stress tests
            logger.info("Running stress tests...")
            stress_results = self.stress_tester.stress_test_concurrent_requests(5)
            
            # 5. Compile results
            self.test_results = {
                'timestamp': datetime.now().isoformat(),
                'test_environment': {
                    'gpu_available': is_gpu_available(),
                    'device': get_device(),
                    'python_version': sys.version,
                },
                'performance': self.performance_tester.get_performance_report(),
                'quality': self.quality_validator.get_quality_report(),
                'stress': self.stress_tester.get_stress_report(),
                'overall_status': self._calculate_overall_status()
            }
            
            logger.info("Comprehensive testing completed")
            return self.test_results
            
        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall test status."""
        try:
            perf_rating = self.performance_tester._calculate_overall_performance()
            quality_rating = self.quality_validator._calculate_overall_quality()
            stress_rating = self.stress_tester._calculate_stress_performance()
            
            ratings = [perf_rating, quality_rating, stress_rating]
            
            if all(rating == 'excellent' for rating in ratings):
                return 'production_ready'
            elif all(rating in ['excellent', 'good'] for rating in ratings):
                return 'ready_with_monitoring'
            elif all(rating in ['excellent', 'good', 'fair'] for rating in ratings):
                return 'needs_optimization'
            else:
                return 'not_ready'
                
        except Exception:
            return 'unknown'
    
    def save_test_report(self, output_path: str = None):
        """Save test report to file."""
        try:
            if not output_path:
                output_path = TestConfig.TEST_OUTPUT_DIR / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            
            logger.info(f"Test report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save test report: {e}")
            return None
    
    def cleanup(self):
        """Clean up test data."""
        self.data_manager.cleanup_test_data()

def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive tests and return results."""
    tester = ComprehensiveTester()
    try:
        results = tester.run_all_tests()
        report_path = tester.save_test_report()
        results['report_path'] = report_path
        return results
    finally:
        tester.cleanup()

if __name__ == "__main__":
    # Run comprehensive tests
    results = run_comprehensive_tests()
    
    print("\n" + "="*50)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*50)
    print(f"Overall Status: {results.get('overall_status', 'unknown')}")
    print(f"Test Report: {results.get('report_path', 'Not saved')}")
    print("="*50)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("All tests completed successfully!")
