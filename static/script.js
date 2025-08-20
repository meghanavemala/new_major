/**
 * Enhanced Video Summarizer Frontend JavaScript
 * 
 * This script handles the complete frontend functionality for the AI Video Summarizer,
 * including form submission, progress tracking, topic selection, and result display.
 * 
 * Features:
 * - Real-time progress tracking with step indicators
 * - Dynamic topic selection interface
 * - Video player for topic summaries
 * - Language-aware interface updates
 * - Download functionality
 * - Error handling and user feedback
 * 
 * Author: Video Summarizer Team
 * Created: 2024
 */

// Global variables
let currentVideoId = null;
let progressInterval = null;
let currentTopicIndex = 0;

// DOM elements
const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const videoForm = document.getElementById('videoForm');
const submitBtn = document.getElementById('submitBtn');
const progressPercentage = document.getElementById('progressPercentage');
const progressMessage = document.getElementById('progressMessage');
const progressFill = document.getElementById('progressFill');
const topicsGrid = document.getElementById('topicsGrid');
const videoPlayerContainer = document.getElementById('videoPlayerContainer');
const videoPlayer = document.getElementById('videoPlayer');
const videoTitle = document.getElementById('videoTitle');
const videoDescription = document.getElementById('videoDescription');
const downloadButtons = document.getElementById('downloadButtons');
const errorMessage = document.getElementById('errorMessage');
const successMessage = document.getElementById('successMessage');
// Final summary elements
const finalSummarySection = document.getElementById('finalSummarySection');
const finalSummaryPlayer = document.getElementById('finalSummaryPlayer');
const finalSummaryDownload = document.getElementById('finalSummaryDownload');

// Step elements
const steps = {
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),
    step4: document.getElementById('step4'),
    step5: document.getElementById('step5')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Refresh hint with backend-provided limit (if needed)
    try {
        const hint = document.querySelector('#fileUploadArea .upload-hint');
        const maxMB = Math.floor(getMaxFileSizeBytes() / (1024 * 1024));
        if (hint) hint.textContent = `MP4, AVI, MOV, MKV, WEBM (Max ${maxMB}MB)`;
    } catch (_) {}
    initializeFileUpload();
    initializeForm();
});

// File upload functionality
function initializeFileUpload() {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('video_file');

    fileUploadArea.addEventListener('click', () => fileInput.click());
    
    fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
        fileUploadArea.classList.add('dragover');
    });
    
    fileUploadArea.addEventListener('dragleave', () => {
        fileUploadArea.classList.remove('dragover');
    });
    
    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            updateFileUploadDisplay(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            console.log('File selected:', e.target.files[0].name, e.target.files[0].size);
            updateFileUploadDisplay(e.target.files[0]);
        }
    });
}

function updateFileUploadDisplay(file) {
    const fileUploadArea = document.getElementById('fileUploadArea');
    const uploadIcon = fileUploadArea.querySelector('.upload-icon');
    const uploadText = fileUploadArea.querySelector('.upload-text');
    const uploadHint = fileUploadArea.querySelector('.upload-hint');
    
    uploadIcon.innerHTML = '<i class="fas fa-file-video"></i>';
    uploadText.textContent = file.name;
    uploadHint.textContent = `Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
}

// Form submission
function initializeForm() {
    videoForm.addEventListener('submit', async function(e) {
    e.preventDefault();

        if (!validateForm()) {
            return;
        }
        
        await submitForm();
    });
    
    // Add test upload button event listener
    const testUploadBtn = document.getElementById('testUploadBtn');
    if (testUploadBtn) {
        testUploadBtn.addEventListener('click', testUpload);
    }
}

function validateForm() {
    const ytUrl = document.getElementById('yt_url').value.trim();
    const videoFile = document.getElementById('video_file').files[0];
    
    console.log('Validation - ytUrl:', ytUrl);
    console.log('Validation - videoFile:', videoFile);
    
    if (!ytUrl && !videoFile) {
        showError('Please provide either a YouTube URL or upload a video file.');
        return false;
    }
    
    const maxBytes = getMaxFileSizeBytes();
    if (videoFile && videoFile.size > maxBytes) {
        const maxMB = Math.floor(maxBytes / (1024 * 1024));
        showError(`Video file size must be less than or equal to ${maxMB}MB.`);
        return false;
    }
    
    return true;
}

async function submitForm() {
    try {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        const formData = new FormData(videoForm);
        
        // Ensure the file is properly attached to FormData
        const videoFile = document.getElementById('video_file').files[0];
        if (videoFile) {
            // Remove any existing entry and add the file
            formData.delete('video_file');
            formData.append('video_file', videoFile);
            console.log('File attached to FormData:', videoFile.name, videoFile.size);
        } else {
            console.log('No file found in video_file input');
        }
        
        // Debug logging
        console.log('Form data contents:');
        for (let [key, value] of formData.entries()) {
            if (value instanceof File) {
                console.log(`${key}: File - ${value.name} (${value.size} bytes)`);
            } else {
                console.log(`${key}: ${value}`);
            }
        }
        
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        // Explicitly handle payload too large
        if (response.status === 413) {
            let data = {};
            try { data = await response.json(); } catch (_) {}
            const msg = (data && (data.message || data.error)) ? (data.message || data.error) : 'File too large';
            showError(msg);
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-play"></i> Start Processing';
            return;
        }

        if (!response.ok) {
            let text = '';
            try { text = await response.text(); } catch (_) {}
            throw new Error(text || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        currentVideoId = result.video_id;
        console.log('Processing started with video ID:', currentVideoId);
        showProcessingSection();
        startProgressTracking();
        
    } catch (error) {
        console.error('Error submitting form:', error);
        showError(`Error: ${error.message}`);
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-play"></i> Start Processing';
    }
}

// Test upload functionality
async function testUpload() {
    try {
        const testBtn = document.getElementById('testUploadBtn');
        testBtn.disabled = true;
        testBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
        
        const formData = new FormData();
        
        // Support either file or YouTube URL for testing
        const videoFile = document.getElementById('video_file').files[0];
        const ytUrl = document.getElementById('yt_url').value.trim();
        
        if (videoFile) {
            formData.append('video_file', videoFile);
            console.log('Test upload - File being sent:', videoFile.name, videoFile.size);
        } else if (ytUrl) {
            formData.append('yt_url', ytUrl);
            console.log('Test upload - URL being sent:', ytUrl);
        } else {
            showError('Please provide a YouTube URL or select a video file first.');
            testBtn.disabled = false;
            testBtn.innerHTML = '<i class="fas fa-bug"></i> Test Upload';
            return;
        }
        
        const response = await fetch('/api/test-upload', {
            method: 'POST',
            body: formData
        });
        // Handle payload too large
        if (response.status === 413) {
            let data = {};
            try { data = await response.json(); } catch (_) {}
            const msg = (data && (data.message || data.error)) ? (data.message || data.error) : 'File too large';
            showError(msg);
            return;
        }

        const result = await response.json();
        
        if (result.success) {
            if (result.type === 'file') {
                showSuccess(`Video uploaded! File: ${result.filename}, Size: ${(result.size / 1024 / 1024).toFixed(2)} MB`);
            } else if (result.type === 'url') {
                const label = result.title || result.yt_url || 'YouTube URL';
                showSuccess(`Video uploaded (URL): ${label}`);
            } else {
                showSuccess('Video input accepted for processing.');
            }
            console.log('Test upload result:', result);
        } else {
            showError(`Test failed: ${result.error}`);
            console.error('Test upload error:', result);
        }
        
    } catch (error) {
        console.error('Error in test upload:', error);
        showError(`Test error: ${error.message}`);
    } finally {
        const testBtn = document.getElementById('testUploadBtn');
        testBtn.disabled = false;
        testBtn.innerHTML = '<i class="fas fa-bug"></i> Test Upload';
    }
}

// Progress tracking
function startProgressTracking() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${currentVideoId}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const status = await response.json();
            console.log('Progress update:', status);
            updateProgress(status);
            
            if (status.status === 'completed') {
                console.log('Processing completed, showing results');
                clearInterval(progressInterval);
                showResults(status);
            } else if (status.status === 'error') {
                console.error('Processing failed:', status.error);
                clearInterval(progressInterval);
                showError(`Processing failed: ${status.error}`);
                showUploadSection();
            }
            
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 1000);
}

function updateProgress(status) {
    const progress = status.progress || 0;
    const message = status.message || 'Processing...';
    
    progressPercentage.textContent = `${progress}%`;
    progressMessage.textContent = message;
    progressFill.style.width = `${progress}%`;
    
    // Update step status based on both progress and message
    updateStepStatus(progress);
    updateStepStatusByStage(status);
}

function updateStepStatus(progress) {
    // Reset all steps
    Object.values(steps).forEach(step => {
        step.className = 'step pending';
    });
    
    // Update steps based on progress
    if (progress >= 5) {
        steps.step1.className = 'step completed';
    }
    if (progress >= 20) {
        steps.step2.className = 'step completed';
    }
    if (progress >= 40) {
        steps.step3.className = 'step completed';
    }
    if (progress >= 55) {
        steps.step4.className = 'step completed';
    }
    if (progress >= 80) {
        steps.step5.className = 'step active';
    }
    if (progress >= 100) {
        steps.step5.className = 'step completed';
    }
}

// Enhanced step status update based on processing stage
function updateStepStatusByStage(status) {
    const message = status.message || '';
    const progress = status.progress || 0;
    
    console.log('Updating step status by stage:', { message, progress });
    
    // Reset all steps
    Object.values(steps).forEach(step => {
        step.className = 'step pending';
    });
    
    // Update steps based on processing stage
    if (message.includes('Downloading') || message.includes('Uploading')) {
        steps.step1.className = 'step active';
        console.log('Step 1: Download/Upload - ACTIVE');
    } else if (message.includes('Transcribing')) {
        steps.step1.className = 'step completed';
        steps.step2.className = 'step active';
        console.log('Step 1: Download/Upload - COMPLETED');
        console.log('Step 2: Transcription - ACTIVE');
    } else if (message.includes('Extracting') || message.includes('key frames')) {
        steps.step1.className = 'step completed';
        steps.step2.className = 'step completed';
        steps.step3.className = 'step active';
        console.log('Step 1: Download/Upload - COMPLETED');
        console.log('Step 2: Transcription - COMPLETED');
        console.log('Step 3: Keyframes - ACTIVE');
    } else if (message.includes('Analyzing') || message.includes('Clustering') || message.includes('Generating summary')) {
        steps.step1.className = 'step completed';
        steps.step2.className = 'step completed';
        steps.step3.className = 'step completed';
        steps.step4.className = 'step active';
        console.log('Step 1: Download/Upload - COMPLETED');
        console.log('Step 2: Transcription - COMPLETED');
        console.log('Step 3: Keyframes - COMPLETED');
        console.log('Step 4: Analysis - ACTIVE');
    } else if (message.includes('Creating video') || message.includes('Rendering')) {
        steps.step1.className = 'step completed';
        steps.step2.className = 'step completed';
        steps.step3.className = 'step completed';
        steps.step4.className = 'step completed';
        steps.step5.className = 'step active';
        console.log('Step 1: Download/Upload - COMPLETED');
        console.log('Step 2: Transcription - COMPLETED');
        console.log('Step 3: Keyframes - COMPLETED');
        console.log('Step 4: Analysis - COMPLETED');
        console.log('Step 5: Video Generation - ACTIVE');
    } else if (message.includes('complete') || message.includes('Complete')) {
        Object.values(steps).forEach(step => {
            step.className = 'step completed';
        });
        console.log('All steps completed!');
    }
}

// Section visibility
function showProcessingSection() {
    uploadSection.style.display = 'none';
    processingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    hideMessages();
}

function showResults(data) {
    console.log('Showing results with data:', data);
    
    uploadSection.style.display = 'none';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Store video data globally for access in other functions
    window.currentVideoData = data;
    // Show final summary video if available
    try {
        if (data.final_summary_video) {
            const fname = getBasenameFromPath(data.final_summary_video);
            if (fname) {
                const url = `/processed/${fname}`;
                if (finalSummaryPlayer) {
                    finalSummaryPlayer.src = url;
                    // Ensure the source element updates too for better compatibility
                    const srcEl = finalSummaryPlayer.querySelector('source');
                    if (srcEl) srcEl.src = url;
                    finalSummaryPlayer.load();
                }
                if (finalSummaryDownload) {
                    finalSummaryDownload.href = url;
                    finalSummaryDownload.download = fname;
                }
                if (finalSummarySection) finalSummarySection.style.display = 'block';
            }
        } else if (finalSummarySection) {
            finalSummarySection.style.display = 'none';
        }
    } catch (e) {
        console.warn('Failed to set final summary video UI:', e);
    }
    
    // Check if we have the required data
    if (data.summaries && data.summaries.length > 0) {
        populateTopics(data);
        populateDownloadButtons(data);
        showSuccess('Video processing completed successfully!');
                } else {
        console.error('No summaries found in data:', data);
        showError('Processing completed but no summaries were generated.');
    }
}

function showUploadSection() {
    uploadSection.style.display = 'block';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-play"></i> Start Processing';
}

// Populate results
function populateTopics(data) {
    topicsGrid.innerHTML = '';
    
    if (!data.summaries || data.summaries.length === 0) {
        topicsGrid.innerHTML = '<p class="text-center">No topics found.</p>';
        return;
    }
    
    data.summaries.forEach((summary, index) => {
        const topicCard = createTopicCard(summary, data.keywords[index], index, data);
        topicsGrid.appendChild(topicCard);
    });
}

function createTopicCard(summary, keywords, index, data) {
    const card = document.createElement('div');
    card.className = 'topic-card';
    card.dataset.topicIndex = index;
    
    const keywordsHtml = keywords ? keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('') : '';
    
    card.innerHTML = `
        <div class="topic-header">
            <div class="topic-number">${index + 1}</div>
            <div class="topic-title">Topic ${index + 1}</div>
        </div>
        <div class="topic-summary">${summary}</div>
        <div class="topic-keywords">${keywordsHtml}</div>
        <div class="topic-actions">
            <button class="btn btn-sm" onclick="playTopic(${index})">
                <i class="fas fa-play"></i> Play Summary
            </button>
            <button class="btn btn-sm btn-secondary" onclick="downloadTopic(${index})">
                <i class="fas fa-download"></i> Download
            </button>
        </div>
    `;
    
    card.addEventListener('click', (e) => {
        if (!e.target.closest('button')) {
            playTopic(index);
        }
    });
    
    return card;
}

function populateDownloadButtons(data) {
    downloadButtons.innerHTML = '';
    
    if (!data.topic_videos || data.topic_videos.length === 0) {
        downloadButtons.innerHTML = '<p>No videos available for download.</p>';
        return;
    }
    
    data.topic_videos.forEach((tv, index) => {
        if (tv && tv.video_filename) {
            const downloadBtn = document.createElement('a');
            downloadBtn.href = `/processed/${tv.video_filename}`;
            downloadBtn.className = 'btn btn-secondary';
            const label = tv.name ? `Topic ${index + 1}: ${tv.name}` : `Topic ${index + 1}`;
            downloadBtn.download = tv.video_filename;
            downloadBtn.innerHTML = `<i class="fas fa-download"></i> ${label}`;
            downloadButtons.appendChild(downloadBtn);
        }
    });
}

// Topic interaction
function playTopic(topicIndex) {
    // Update active topic card
    document.querySelectorAll('.topic-card').forEach(card => {
        card.classList.remove('active');
    });
    
    const activeCard = document.querySelector(`[data-topic-index="${topicIndex}"]`);
    if (activeCard) {
        activeCard.classList.add('active');
    }
    
    currentTopicIndex = topicIndex;
    
    // Check if video exists for this topic
    if (!window.currentVideoData || !window.currentVideoData.topic_videos || !window.currentVideoData.topic_videos[topicIndex] || !window.currentVideoData.topic_videos[topicIndex].video_filename) {
        showError('Video for this topic is not available.');
        return;
    }
    
    // Update video player with organized path
    const tv = window.currentVideoData.topic_videos[topicIndex];
    const videoPath = `/processed/${tv.video_filename}`;
    videoPlayer.src = videoPath;
    
    // Update video info with topic details
    const titleKeywords = Array.isArray(tv.keywords) ? tv.keywords.slice(0, 3).join(', ') : '';
    const namePart = tv.name ? `: ${tv.name}` : (titleKeywords ? `: ${titleKeywords}` : ' Summary');
    videoTitle.textContent = `Topic ${topicIndex + 1}${namePart}`;
    videoDescription.textContent = tv.summary || 'Click play to start watching the summary video.';
    
    // Show video player
    videoPlayerContainer.style.display = 'block';
    
    // Load video metadata
    videoPlayer.load();
}

function downloadTopic(topicIndex) {
    if (!window.currentVideoData || !window.currentVideoData.topic_videos || !window.currentVideoData.topic_videos[topicIndex] || !window.currentVideoData.topic_videos[topicIndex].video_filename) {
        showError('Video for this topic is not available for download.');
        return;
    }
    
    const tv = window.currentVideoData.topic_videos[topicIndex];
    const videoPath = `/processed/${tv.video_filename}`;
    const link = document.createElement('a');
    link.href = videoPath;
    link.download = tv.video_filename || `topic_${topicIndex + 1}_summary.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showSuccess(`Downloading topic ${topicIndex + 1} summary video...`);
}

// Utility functions
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    setTimeout(() => {
        successMessage.style.display = 'none';
    }, 5000);
}

function hideMessages() {
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

// Path utility: get basename from absolute or relative path (handles Windows and POSIX)
function getBasenameFromPath(p) {
    try {
        if (!p || typeof p !== 'string') return '';
        const parts = p.split(/[\\/]+/);
        return parts[parts.length - 1] || '';
    } catch (_) {
        return '';
    }
}

// Reset functionality
function resetForm() {
    videoForm.reset();
    const maxMB = Math.floor(getMaxFileSizeBytes() / (1024 * 1024));
    document.getElementById('fileUploadArea').innerHTML = `
        <div class="upload-icon"><i class="fas fa-cloud-upload-alt"></i></div>
        <div class="upload-text">Click to upload or drag & drop</div>
        <div class="upload-hint">MP4, AVI, MOV, MKV, WEBM (Max ${maxMB}MB)</div>
    `;
    showUploadSection();
}

// Helper to read max file size (bytes) from backend-provided data attribute, with 100MB fallback
function getMaxFileSizeBytes() {
    const el = document.getElementById('app-config');
    if (el && el.dataset && el.dataset.maxFileSize) {
        const n = parseInt(el.dataset.maxFileSize, 10);
        if (!isNaN(n) && n > 0) return n;
    }
    return 100 * 1024 * 1024;
}

// Add reset button functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add reset button to the form
    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'btn btn-secondary';
    resetBtn.innerHTML = '<i class="fas fa-redo"></i> Reset';
    resetBtn.onclick = resetForm;
    
    const submitGroup = submitBtn.parentElement;
    submitGroup.appendChild(resetBtn);
});