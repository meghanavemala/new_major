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

// Global state management
let currentVideoId = null;
let processingInterval = null;
let summaryData = null;

// DOM element references
const elements = {
    form: document.getElementById('main-form'),
    loader: document.getElementById('loader'),
    resultContainer: document.getElementById('result-container'),
    progressPercentage: document.getElementById('progress-percentage'),
    progressFill: document.getElementById('progress-fill'),
    processingTitle: document.getElementById('processing-title'),
    processingMessage: document.getElementById('processing-message'),
    topicsGrid: document.getElementById('topics-grid'),
    topicPlayer: document.getElementById('topic-player'),
    topicVideo: document.getElementById('topic-video'),
    currentTopicTitle: document.getElementById('current-topic-title'),
    topicSummaryText: document.getElementById('topic-summary-text'),
    topicKeywordsList: document.getElementById('topic-keywords-list'),
    closePlayerBtn: document.getElementById('close-player'),
    
    // Form elements
    sourceLanguage: document.getElementById('source-language'),
    targetLanguage: document.getElementById('target-language'),
    voice: document.getElementById('voice'),
    resolution: document.getElementById('resolution'),
    summaryLength: document.getElementById('summary_length'),
    enableOcr: document.getElementById('enable_ocr'),
    videoUpload: document.getElementById('video-upload'),
    ytUrl: document.getElementById('yt-url'),
    
    // Processing steps
    stepDownload: document.getElementById('step-download'),
    stepTranscribe: document.getElementById('step-transcribe'),
    stepKeyframes: document.getElementById('step-keyframes'),
    stepAnalyze: document.getElementById('step-analyze'),
    stepGenerate: document.getElementById('step-generate'),
};

// Processing step configuration
const PROCESSING_STEPS = {
    'downloading': { element: elements.stepDownload, message: 'Downloading video...', progress: [0, 10] },
    'transcribing': { element: elements.stepTranscribe, message: 'Converting speech to text...', progress: [10, 30] },
    'extracting': { element: elements.stepKeyframes, message: 'Extracting key frames...', progress: [30, 50] },
    'clustering': { element: elements.stepAnalyze, message: 'Analyzing content and topics...', progress: [50, 70] },
    'summarizing': { element: elements.stepAnalyze, message: 'Generating summaries...', progress: [70, 85] },
    'rendering': { element: elements.stepGenerate, message: 'Creating summary videos...', progress: [85, 95] },
    'completed': { element: null, message: 'Processing complete!', progress: [95, 100] }
};

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeLanguageHandlers();
    loadUserPreferences();
});

/**
 * Set up all event listeners for the application
 */
function initializeEventListeners() {
    // Form submission
    elements.form.addEventListener('submit', handleFormSubmission);
    
    // File upload handling
    elements.videoUpload.addEventListener('change', handleFileSelection);
    
    // Language change handlers
    elements.targetLanguage.addEventListener('change', updateVoiceOptions);
    
    // Player controls
    elements.closePlayerBtn.addEventListener('click', closeTopicPlayer);
    
    // Download buttons
    document.getElementById('download-all')?.addEventListener('click', downloadAllSummaries);
    document.getElementById('download-transcript')?.addEventListener('click', downloadTranscript);
    document.getElementById('download-keyframes')?.addEventListener('click', downloadKeyframes);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

/**
 * Initialize language-specific handlers
 */
function initializeLanguageHandlers() {
    // Auto-update voice options when target language changes
    updateVoiceOptions();
    
    // Handle auto-detect language changes
    elements.sourceLanguage.addEventListener('change', function() {
        if (this.value === 'auto') {
            showTooltip(this, 'Language will be automatically detected during processing');
        }
    });
}

/**
 * Load user preferences from localStorage
 */
function loadUserPreferences() {
    const preferences = JSON.parse(localStorage.getItem('videoSummarizerPrefs') || '{}');
    
    if (preferences.targetLanguage) {
        elements.targetLanguage.value = preferences.targetLanguage;
    }
    if (preferences.voice) {
        elements.voice.value = preferences.voice;
    }
    if (preferences.resolution) {
        elements.resolution.value = preferences.resolution;
    }
    if (preferences.summaryLength) {
        elements.summaryLength.value = preferences.summaryLength;
    }
    if (preferences.enableOcr !== undefined) {
        elements.enableOcr.checked = preferences.enableOcr;
    }
    
    updateVoiceOptions();
}

/**
 * Save user preferences to localStorage
 */
function saveUserPreferences() {
    const preferences = {
        targetLanguage: elements.targetLanguage.value,
        voice: elements.voice.value,
        resolution: elements.resolution.value,
        summaryLength: elements.summaryLength.value,
        enableOcr: elements.enableOcr.checked
    };
    
    localStorage.setItem('videoSummarizerPrefs', JSON.stringify(preferences));
}

/**
 * Handle form submission for video processing
 */
async function handleFormSubmission(e) {
    e.preventDefault();

    // Validate form
    if (!validateForm()) {
        return;
    }
    
    // Save user preferences
    saveUserPreferences();
    
    // Prepare form data
    const formData = new FormData(elements.form);
    
    // Show processing UI
    showProcessingUI();
    
    try {
        // Submit form
        const response = await fetch('/api/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.video_id) {
            currentVideoId = data.video_id;
            startProgressTracking();
        } else {
            throw new Error(data.error || 'Failed to start processing');
        }
        
    } catch (error) {
        showError('Failed to start processing: ' + error.message);
        resetUI();
    }
}

/**
 * Validate form before submission
 */
function validateForm() {
    const hasVideo = elements.videoUpload.files.length > 0;
    const hasUrl = elements.ytUrl.value.trim() !== '';
    
    if (!hasVideo && !hasUrl) {
        showError('Please upload a video file or provide a YouTube URL');
        return false;
    }
    
    if (hasVideo && hasUrl) {
        showError('Please provide either a video file OR a YouTube URL, not both');
        return false;
    }
    
    // Validate file size if uploading
    if (hasVideo) {
        const file = elements.videoUpload.files[0];
        const maxSize = 100 * 1024 * 1024; // 100MB
        
        if (file.size > maxSize) {
            showError('File size must be less than 100MB');
            return false;
        }
    }
    
    // Validate YouTube URL format
    if (hasUrl) {
        const ytUrlPattern = /^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[\w-]+/;
        if (!ytUrlPattern.test(elements.ytUrl.value.trim())) {
            showError('Please provide a valid YouTube URL');
            return false;
        }
    }
    
    return true;
}

/**
 * Handle file selection for upload
 */
function handleFileSelection(e) {
    const file = e.target.files[0];
    if (file) {
        // Clear YouTube URL if file is selected
        elements.ytUrl.value = '';
        
        // Update UI to show selected file
        const label = document.querySelector('.upload-label span');
        label.textContent = `Selected: ${file.name}`;
        label.parentElement.classList.add('file-selected');
    }
}

/**
 * Update voice options based on selected target language
 */
function updateVoiceOptions() {
    const targetLang = elements.targetLanguage.value;
    const voiceSelect = elements.voice;
    
    // Clear existing options except auto
    while (voiceSelect.children.length > 1) {
        voiceSelect.removeChild(voiceSelect.lastChild);
    }
    
    // Voice mapping for different languages
    const voiceOptions = {
        'english': [
            { value: 'en-US-Standard-A', text: '🇺🇸 US English (Male)' },
            { value: 'en-US-Standard-C', text: '🇺🇸 US English (Female)' },
            { value: 'en-GB-Standard-A', text: '🇬🇧 British English (Female)' },
            { value: 'en-AU-Standard-A', text: '🇦🇺 Australian English (Female)' }
        ],
        'hindi': [
            { value: 'hi-IN-Standard-A', text: '🇮🇳 Hindi (Female)' },
            { value: 'hi-IN-Standard-B', text: '🇮🇳 Hindi (Male)' }
        ],
        'bengali': [
            { value: 'bn-IN-Standard-A', text: '🇮🇳 Bengali (Female)' },
            { value: 'bn-IN-Standard-B', text: '🇮🇳 Bengali (Male)' }
        ],
        'tamil': [
            { value: 'ta-IN-Standard-A', text: '🇮🇳 Tamil (Female)' },
            { value: 'ta-IN-Standard-B', text: '🇮🇳 Tamil (Male)' }
        ],
        // Add more languages as needed
    };
    
    // Add language-specific voices
    const voices = voiceOptions[targetLang] || [
        { value: 'auto-' + targetLang, text: `🎯 Best ${targetLang.charAt(0).toUpperCase() + targetLang.slice(1)} Voice` }
    ];
    
    voices.forEach(voice => {
        const option = document.createElement('option');
        option.value = voice.value;
        option.textContent = voice.text;
        voiceSelect.appendChild(option);
    });
}

/**
 * Show processing UI and hide form
 */
function showProcessingUI() {
    elements.form.parentElement.style.display = 'none';
    elements.loader.classList.remove('hidden');
    elements.resultContainer.classList.add('hidden');
    
    // Reset processing steps
    resetProcessingSteps();
    updateProgress(0, 'Initializing...');
}

/**
 * Reset all processing step indicators
 */
function resetProcessingSteps() {
    Object.values(PROCESSING_STEPS).forEach(step => {
        if (step.element) {
            step.element.classList.remove('active', 'completed');
                }
            });
    }

/**
 * Start tracking processing progress
 */
function startProgressTracking() {
    if (processingInterval) {
        clearInterval(processingInterval);
    }
    
    processingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${currentVideoId}`);
            const status = await response.json();
            
            updateProcessingStatus(status);
            
            if (status.status === 'completed') {
                clearInterval(processingInterval);
                processingInterval = null;
                showResults(status);
            } else if (status.status === 'error') {
                clearInterval(processingInterval);
                processingInterval = null;
                showError(status.error || 'Processing failed');
                resetUI();
            }
            
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 2000); // Check every 2 seconds
}

/**
 * Update processing status display
 */
function updateProcessingStatus(status) {
    const progress = Math.min(status.progress || 0, 100);
    const message = status.message || 'Processing...';
    const currentStatus = status.status || 'processing';
    
    updateProgress(progress, message);
    updateProcessingSteps(currentStatus, progress);
}

/**
 * Update progress bar and percentage
 */
function updateProgress(percentage, message) {
    elements.progressPercentage.textContent = `${Math.round(percentage)}%`;
    elements.progressFill.style.width = `${percentage}%`;
    elements.processingMessage.textContent = message;
}

/**
 * Update processing step indicators
 */
function updateProcessingSteps(status, progress) {
    const stepConfig = PROCESSING_STEPS[status];
    
    if (stepConfig && stepConfig.element) {
        // Mark current step as active
        stepConfig.element.classList.add('active');
        
        // Mark previous steps as completed
        Object.values(PROCESSING_STEPS).forEach(step => {
            if (step.element && step.progress[1] <= progress) {
                step.element.classList.add('completed');
                step.element.classList.remove('active');
                }
            });
        }
}

/**
 * Show results after processing completion
 */
function showResults(data) {
    elements.loader.classList.add('hidden');
    elements.resultContainer.classList.remove('hidden');
    
    summaryData = data;
    
    // Populate topics grid
    populateTopicsGrid(data.summaries, data.keywords);
    
    // Show success message
    showSuccess('Video processing completed successfully!');
}

/**
 * Populate the topics grid with available topics
 */
function populateTopicsGrid(summaries, keywords) {
    elements.topicsGrid.innerHTML = '';
    
    summaries.forEach((summary, index) => {
        const topicCard = createTopicCard(index, summary, keywords[index] || []);
        elements.topicsGrid.appendChild(topicCard);
    });
}

/**
 * Create a topic card element
 */
function createTopicCard(index, summary, topicKeywords) {
    const card = document.createElement('div');
    card.className = 'topic-card';
    card.setAttribute('data-topic-index', index);
    
    // Generate topic title from keywords or use default
    const topicTitle = topicKeywords.length > 0 
        ? topicKeywords.slice(0, 3).join(', ')
        : `Topic ${index + 1}`;
    
    card.innerHTML = `
        <div class="topic-header">
            <h4><i class="fas fa-play-circle"></i> ${topicTitle}</h4>
            <span class="topic-duration">${estimateDuration(summary)}</span>
        </div>
        <div class="topic-preview">
            <p>${truncateText(summary, 100)}</p>
        </div>
        <div class="topic-keywords">
            ${topicKeywords.slice(0, 5).map(keyword => 
                `<span class="keyword-tag">${keyword}</span>`
            ).join('')}
        </div>
        <button class="play-topic-btn" onclick="playTopic(${index})">
            <i class="fas fa-play"></i> Play Summary
        </button>
    `;
    
    return card;
}

/**
 * Play a specific topic summary
 */
function playTopic(topicIndex) {
    if (!summaryData || !summaryData.summary_videos[topicIndex]) {
        showError('Topic video not available');
        return;
    }
    
    const summary = summaryData.summaries[topicIndex];
    const keywords = summaryData.keywords[topicIndex] || [];
    const videoPath = summaryData.summary_videos[topicIndex];
    
    // Update topic player
    elements.currentTopicTitle.textContent = keywords.length > 0 
        ? keywords.slice(0, 3).join(', ')
        : `Topic ${topicIndex + 1}`;
    
    elements.topicSummaryText.textContent = summary;
    
    // Populate keywords
    elements.topicKeywordsList.innerHTML = keywords
        .map(keyword => `<span class="keyword-tag">${keyword}</span>`)
        .join('');
    
    // Set video source
    elements.topicVideo.src = `/api/stream/${currentVideoId}/${topicIndex}`;
    
    // Show player
    elements.topicPlayer.classList.remove('hidden');
    
    // Scroll to player
    elements.topicPlayer.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Close the topic player
 */
function closeTopicPlayer() {
    elements.topicPlayer.classList.add('hidden');
    elements.topicVideo.pause();
    elements.topicVideo.src = '';
}

/**
 * Download all summaries as a ZIP file
 */
function downloadAllSummaries() {
    if (!currentVideoId) return;
    
    // Create download link
    const link = document.createElement('a');
    link.href = `/api/download/all/${currentVideoId}`;
    link.download = `video_summaries_${currentVideoId}.zip`;
    link.click();
}

/**
 * Download transcript file
 */
function downloadTranscript() {
    if (!currentVideoId) return;
    
    const link = document.createElement('a');
    link.href = `/srt/${currentVideoId}`;
    link.download = `transcript_${currentVideoId}.srt`;
    link.click();
}

/**
 * Download keyframes
 */
function downloadKeyframes() {
    if (!currentVideoId) return;
    
    const link = document.createElement('a');
    link.href = `/api/download/keyframes/${currentVideoId}`;
    link.download = `keyframes_${currentVideoId}.zip`;
    link.click();
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
    // Escape key to close player
    if (e.key === 'Escape' && !elements.topicPlayer.classList.contains('hidden')) {
        closeTopicPlayer();
    }
    
    // Space key to play/pause current video
    if (e.key === ' ' && !elements.topicPlayer.classList.contains('hidden')) {
        e.preventDefault();
        if (elements.topicVideo.paused) {
            elements.topicVideo.play();
        } else {
            elements.topicVideo.pause();
        }
    }
}

/**
 * Utility Functions
 */

function truncateText(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function estimateDuration(text) {
    // Rough estimate: 150 words per minute for speech
    const words = text.split(' ').length;
    const minutes = Math.ceil(words / 150);
    return `~${minutes} min`;
}

function showError(message) {
    showNotification(message, 'error');
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-circle' : type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="close-notification" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function showTooltip(element, message) {
    // Simple tooltip implementation
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = message;
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = (rect.bottom + 5) + 'px';
    
    setTimeout(() => tooltip.remove(), 3000);
}

function resetUI() {
    elements.form.parentElement.style.display = 'block';
    elements.loader.classList.add('hidden');
    elements.resultContainer.classList.add('hidden');
    
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    currentVideoId = null;
    summaryData = null;
}