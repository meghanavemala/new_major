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
    
    if (videoFile && videoFile.size > 100 * 1024 * 1024) {
        showError('Video file size must be less than 40MB.');
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
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
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
        
        // Get the video file
        const videoFile = document.getElementById('video_file').files[0];
        if (!videoFile) {
            showError('Please select a video file first.');
            testBtn.disabled = false;
            testBtn.innerHTML = '<i class="fas fa-bug"></i> Test Upload';
            return;
        }
        
        // Add the file to FormData
        formData.append('video_file', videoFile);
        
        console.log('Test upload - File being sent:', videoFile.name, videoFile.size);
        
        const response = await fetch('/api/test-upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSuccess(`Test successful! File: ${result.filename}, Size: ${(result.size / 1024 / 1024).toFixed(2)} MB`);
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
    
    if (!data.summary_videos || data.summary_videos.length === 0) {
        downloadButtons.innerHTML = '<p>No videos available for download.</p>';
        return;
    }
    
    data.summary_videos.forEach((videoPath, index) => {
        if (videoPath) {
            const downloadBtn = document.createElement('a');
            downloadBtn.href = `/processed/${videoPath}`;
            downloadBtn.className = 'btn btn-secondary';
            downloadBtn.download = `topic_${index + 1}_summary.mp4`;
            downloadBtn.innerHTML = `<i class="fas fa-download"></i> Topic ${index + 1}`;
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
    if (!window.currentVideoData || !window.currentVideoData.summary_videos || !window.currentVideoData.summary_videos[topicIndex]) {
        showError('Video for this topic is not available.');
        return;
    }
    
    // Update video player with organized path
    const videoPath = `/processed/${window.currentVideoData.summary_videos[topicIndex]}`;
    videoPlayer.src = videoPath;
    
    // Update video info with topic details
    const topicData = window.currentVideoData.clusters?.[topicIndex];
    if (topicData) {
        videoTitle.textContent = `Topic ${topicIndex + 1}: ${topicData.keywords?.slice(0, 3).join(', ') || 'Summary'}`;
        videoDescription.textContent = topicData.summary || 'Click play to start watching the summary video.';
    } else {
        videoTitle.textContent = `Topic ${topicIndex + 1} Summary`;
        videoDescription.textContent = 'Click play to start watching the summary video.';
    }
    
    // Show video player
    videoPlayerContainer.style.display = 'block';
    
    // Load video metadata
    videoPlayer.load();
}

function downloadTopic(topicIndex) {
    if (!window.currentVideoData || !window.currentVideoData.summary_videos || !window.currentVideoData.summary_videos[topicIndex]) {
        showError('Video for this topic is not available for download.');
        return;
    }
    
    const videoPath = `/processed/${window.currentVideoData.summary_videos[topicIndex]}`;
    const link = document.createElement('a');
    link.href = videoPath;
    link.download = `topic_${topicIndex + 1}_summary.mp4`;
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

// Reset functionality
function resetForm() {
    videoForm.reset();
    document.getElementById('fileUploadArea').innerHTML = `
        <div class="upload-icon"><i class="fas fa-cloud-upload-alt"></i></div>
        <div class="upload-text">Click to upload or drag & drop</div>
        <div class="upload-hint">MP4, AVI, MOV, MKV, WEBM (Max 40MB)</div>
    `;
    showUploadSection();
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