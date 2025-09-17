// PlantDoc AI - Interactive JavaScript
class PlantDocApp {
    constructor() {
        this.selectedImage = null;
        this.isAnalyzing = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupAnimations();
        this.setupScrollEffects();
    }

    setupEventListeners() {
        // Upload area events
        const uploadArea = document.querySelector('.upload-area');
        const imageInput = document.getElementById('imageInput');

        if (uploadArea && imageInput) {
            uploadArea.addEventListener('click', () => imageInput.click());
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleDrop.bind(this));
            imageInput.addEventListener('change', this.handleImageUpload.bind(this));
        }

        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', this.handleSmoothScroll);
        });

        // Button events
        const analyzeBtn = document.querySelector('button[aria-label="Analyze plant disease"]');
        const clearBtn = document.querySelector('button[aria-label="Clear selected image"]');
        
        if (analyzeBtn) analyzeBtn.addEventListener('click', this.analyzeImage.bind(this));
        if (clearBtn) clearBtn.addEventListener('click', this.clearImage.bind(this));
    }

    setupAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.info-card, .result-card').forEach(el => {
            observer.observe(el);
        });

        // Add floating animation to upload icon
        this.addFloatingAnimation();
    }

    setupScrollEffects() {
        // Parallax effect for hero section
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const hero = document.querySelector('.hero');
            if (hero) {
                hero.style.transform = `translateY(${scrolled * 0.3}px)`;
                hero.style.opacity = 1 - scrolled / window.innerHeight;
            }
        });

        // Header shadow on scroll
        window.addEventListener('scroll', () => {
            const header = document.querySelector('.header');
            if (header) {
                if (window.scrollY > 100) {
                    header.style.boxShadow = '0 5px 30px rgba(0, 0, 0, 0.2)';
                } else {
                    header.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
                }
            }
        });
    }

    addFloatingAnimation() {
        const uploadIcon = document.querySelector('.upload-icon');
        if (uploadIcon) {
            setInterval(() => {
                uploadIcon.style.animation = 'none';
                setTimeout(() => {
                    uploadIcon.style.animation = 'bounce 2s infinite';
                }, 10);
            }, 10000);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
        this.showUploadFeedback('Drop your image here!', 'info');
    }

    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleImageFile(files[0]);
        }
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleImageFile(file);
        }
    }

    handleImageFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file (JPG, PNG, JPEG)');
            return;
        }

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size too large. Please choose an image under 10MB.');
            return;
        }

        this.selectedImage = file;
        this.showPreview(file);
        this.hideError();
        this.showUploadFeedback('Image uploaded successfully!', 'success');
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            const previewContainer = document.getElementById('previewContainer');
            
            if (previewImage && previewContainer) {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';
                
                // Add entrance animation
                previewContainer.style.opacity = '0';
                previewContainer.style.transform = 'scale(0.8)';
                
                setTimeout(() => {
                    previewContainer.style.transition = 'all 0.5s ease-out';
                    previewContainer.style.opacity = '1';
                    previewContainer.style.transform = 'scale(1)';
                }, 100);
            }
        };
        reader.readAsDataURL(file);
    }

    clearImage() {
        this.selectedImage = null;
        const previewContainer = document.getElementById('previewContainer');
        const resultsContainer = document.getElementById('resultsContainer');
        const imageInput = document.getElementById('imageInput');
        
        if (previewContainer) {
            previewContainer.style.transition = 'all 0.3s ease-out';
            previewContainer.style.opacity = '0';
            previewContainer.style.transform = 'scale(0.8)';
            setTimeout(() => {
                previewContainer.style.display = 'none';
            }, 300);
        }
        
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }
        
        if (imageInput) {
            imageInput.value = '';
        }
        
        this.hideError();
        this.showUploadFeedback('Ready for new image', 'info');
    }

    async analyzeImage() {
        if (!this.selectedImage) {
            this.showError('Please select an image first');
            return;
        }

        if (this.isAnalyzing) {
            return;
        }

        this.isAnalyzing = true;
        this.showLoading();
        this.hideError();

        const formData = new FormData();
        formData.append('file', this.selectedImage);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Full API response:', result); // Debug log
            console.log('Prediction data:', result.prediction); // Debug log
            await this.displayResults(result.prediction);
            
        } catch (error) {
            console.error('Error:', error);
            this.showError('Failed to analyze image. Please check your internet connection and try again.');
        } finally {
            this.hideLoading();
            this.isAnalyzing = false;
        }
    }

    showLoading() {
        const loading = document.getElementById('loading');
        const resultsContainer = document.getElementById('resultsContainer');
        
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }
        
        if (loading) {
            loading.style.display = 'block';
            loading.style.opacity = '0';
            setTimeout(() => {
                loading.style.transition = 'opacity 0.5s ease-in';
                loading.style.opacity = '1';
            }, 100);
        }

        // Add pulsing effect to spinner
        const spinner = document.querySelector('.spinner');
        if (spinner) {
            spinner.style.animation = 'spin 1s linear infinite, pulse 2s ease-in-out infinite';
        }
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        if (loading) {
            loading.style.opacity = '0';
            setTimeout(() => {
                loading.style.display = 'none';
            }, 500);
        }
    }

    async displayResults(prediction) {
        // Update main diagnosis without processing
        this.updateDiagnosisInfo(prediction);
        
        // Update confidence bar with animation
        this.updateConfidenceBar(prediction);
        
        // Update diagnosis title based on condition
        this.updateDiagnosisTitle(prediction);
        
        // Display top predictions with staggered animation
        this.displayTopPredictions(prediction);
        
        // Update treatment recommendations
        this.updateTreatmentRecommendations(prediction);
        
        // Show results with entrance animation
        this.showResults();
    }

    updateDiagnosisInfo(prediction) {
        const plantCondition = document.getElementById('plantCondition');
        const plantType = document.getElementById('plantType');
        const confidenceText = document.getElementById('confidenceText');
        
        // Display raw prediction results without any processing
        if (plantCondition) {
            plantCondition.textContent = prediction.disease || 'Unknown';
        }

        if (plantType) {
            plantType.textContent = prediction.plant || 'Unknown';
        }

        if (confidenceText) {
            confidenceText.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
        }
    }

    typeWriterEffect(element, text) {
        return new Promise((resolve) => {
            element.textContent = '';
            let i = 0;
            const timer = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                } else {
                    clearInterval(timer);
                    resolve();
                }
            }, 50);
        });
    }

    updateConfidenceBar(prediction) {
        const confidenceFill = document.getElementById('confidenceFill');
        if (confidenceFill) {
            // Reset width
            confidenceFill.style.width = '0%';
            
            // Animate to actual confidence
            setTimeout(() => {
                confidenceFill.style.width = `${prediction.confidence * 100}%`;
                
                // Color code confidence
                if (prediction.confidence > 0.8) {
                    confidenceFill.style.background = 'var(--gradient-success)';
                } else if (prediction.confidence > 0.6) {
                    confidenceFill.style.background = 'var(--gradient-warning)';
                } else {
                    confidenceFill.style.background = 'var(--gradient-error)';
                }
            }, 500);
        }
    }

    updateDiagnosisTitle(prediction) {
        const isHealthy = prediction.disease && prediction.disease.toLowerCase().includes('healthy');
        const diagnosisTitle = document.getElementById('diagnosisTitle');
        const diagnosisIcon = document.querySelector('.diagnosis-icon i');
        
        if (diagnosisTitle && diagnosisIcon) {
            if (isHealthy) {
                diagnosisTitle.textContent = '✅ Healthy Plant Detected';
                diagnosisTitle.style.color = 'var(--success-green)';
                diagnosisIcon.className = 'fas fa-check-circle';
                diagnosisIcon.parentElement.style.background = 'var(--gradient-success)';
            } else {
                diagnosisTitle.textContent = '⚠️ Disease Detected';
                diagnosisTitle.style.color = 'var(--error-red)';
                diagnosisIcon.className = 'fas fa-exclamation-triangle';
                diagnosisIcon.parentElement.style.background = 'var(--gradient-error)';
            }
            
            // Add pulse animation
            diagnosisIcon.parentElement.style.animation = 'pulseIcon 2s infinite';
        }
    }

    displayTopPredictions(prediction) {
        const topPredictions = document.getElementById('topPredictions');
        if (topPredictions && prediction.top_5) {
            topPredictions.innerHTML = '';
            
            prediction.top_5.slice(0, 3).forEach((pred, index) => {
                const predDiv = document.createElement('div');
                predDiv.className = 'prediction-item';
                predDiv.style.animationDelay = `${index * 0.2}s`;
                predDiv.innerHTML = `
                    <div class="prediction-name">${index + 1}. ${pred.class}</div>
                    <div class="prediction-confidence">${(pred.confidence * 100).toFixed(1)}%</div>
                `;
                topPredictions.appendChild(predDiv);
            });
        }
    }

    updateTreatmentRecommendations(prediction) {
        const treatmentList = document.getElementById('treatmentList');
        const isHealthy = prediction.disease && prediction.disease.toLowerCase().includes('healthy');
        
        if (treatmentList) {
            if (isHealthy) {
                treatmentList.innerHTML = `
                    <li style="animation-delay: 0.1s"><i class="fas fa-smile"></i>Great! Your plant appears healthy</li>
                    <li style="animation-delay: 0.2s"><i class="fas fa-water"></i>Continue regular watering schedule</li>
                    <li style="animation-delay: 0.3s"><i class="fas fa-sun"></i>Ensure adequate sunlight exposure</li>
                    <li style="animation-delay: 0.4s"><i class="fas fa-scissors"></i>Regular pruning for optimal growth</li>
                    <li style="animation-delay: 0.5s"><i class="fas fa-eye"></i>Monitor regularly for early disease detection</li>
                `;
            } else {
                const diseaseSpecificTreatments = this.getTreatmentForDisease(prediction.disease);
                treatmentList.innerHTML = diseaseSpecificTreatments;
            }
        }
    }

    getTreatmentForDisease(disease) {
        const treatments = {
            'bacterial_spot': `
                <li style="animation-delay: 0.1s"><i class="fas fa-cut"></i>Remove and destroy infected leaves immediately</li>
                <li style="animation-delay: 0.2s"><i class="fas fa-spray-can"></i>Apply copper-based bactericide</li>
                <li style="animation-delay: 0.3s"><i class="fas fa-ban"></i>Avoid overhead watering</li>
                <li style="animation-delay: 0.4s"><i class="fas fa-wind"></i>Improve air circulation</li>
                <li style="animation-delay: 0.5s"><i class="fas fa-clock"></i>Apply preventive sprays during humid weather</li>
            `,
            'early_blight': `
                <li style="animation-delay: 0.1s"><i class="fas fa-leaf"></i>Remove lower infected leaves</li>
                <li style="animation-delay: 0.2s"><i class="fas fa-spray-can"></i>Apply fungicide containing chlorothalonil</li>
                <li style="animation-delay: 0.3s"><i class="fas fa-water"></i>Water at soil level, not on leaves</li>
                <li style="animation-delay: 0.4s"><i class="fas fa-arrows-alt"></i>Increase plant spacing for air flow</li>
                <li style="animation-delay: 0.5s"><i class="fas fa-recycle"></i>Rotate crops annually</li>
            `,
            'late_blight': `
                <li style="animation-delay: 0.1s"><i class="fas fa-exclamation-triangle"></i>Act immediately - this spreads rapidly!</li>
                <li style="animation-delay: 0.2s"><i class="fas fa-spray-can"></i>Apply fungicide with metalaxyl or mancozeb</li>
                <li style="animation-delay: 0.3s"><i class="fas fa-trash"></i>Remove and destroy all infected plant material</li>
                <li style="animation-delay: 0.4s"><i class="fas fa-tint"></i>Avoid watering in evening hours</li>
                <li style="animation-delay: 0.5s"><i class="fas fa-phone"></i>Contact agricultural extension immediately</li>
            `,
            'default': `
                <li style="animation-delay: 0.1s"><i class="fas fa-leaf"></i>Remove affected leaves immediately to prevent spread</li>
                <li style="animation-delay: 0.2s"><i class="fas fa-tint"></i>Adjust watering schedule - avoid overhead watering</li>
                <li style="animation-delay: 0.3s"><i class="fas fa-wind"></i>Improve air circulation around plants</li>
                <li style="animation-delay: 0.4s"><i class="fas fa-spray-can"></i>Apply appropriate fungicide or treatment</li>
                <li style="animation-delay: 0.5s"><i class="fas fa-user-md"></i>Consult local agricultural extension service if symptoms persist</li>
            `
        };

        // Try to match disease name to treatment
        for (const [key, treatment] of Object.entries(treatments)) {
            if (disease && disease.toLowerCase().includes(key)) {
                return treatment;
            }
        }
        
        return treatments.default;
    }

    showResults() {
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
            resultsContainer.style.opacity = '0';
            resultsContainer.style.transform = 'translateY(50px)';
            
            setTimeout(() => {
                resultsContainer.style.transition = 'all 0.8s ease-out';
                resultsContainer.style.opacity = '1';
                resultsContainer.style.transform = 'translateY(0)';
                
                // Smooth scroll to results
                resultsContainer.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        }
    }

    showError(message) {
        const errorDiv = document.getElementById('errorMessage');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            errorDiv.style.animation = 'shake 0.5s ease-in-out';
            
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
    }

    hideError() {
        const errorDiv = document.getElementById('errorMessage');
        if (errorDiv) {
            errorDiv.style.display = 'none';
        }
    }

    showUploadFeedback(message, type = 'info') {
        // Create or update feedback element
        let feedback = document.querySelector('.upload-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'upload-feedback';
            feedback.style.cssText = `
                position: fixed;
                top: 100px;
                right: 20px;
                padding: 1rem;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                z-index: 1000;
                transform: translateX(100%);
                transition: transform 0.3s ease-out;
            `;
            document.body.appendChild(feedback);
        }

        // Set color based on type
        const colors = {
            success: 'var(--gradient-success)',
            error: 'var(--gradient-error)',
            info: 'var(--gradient-primary)'
        };
        
        feedback.style.background = colors[type] || colors.info;
        feedback.textContent = message;
        
        // Show feedback
        feedback.style.transform = 'translateX(0)';
        
        // Hide after 3 seconds
        setTimeout(() => {
            feedback.style.transform = 'translateX(100%)';
        }, 3000);
    }

    handleSmoothScroll(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    cleanText(text) {
        if (!text) return 'Unknown';
        
        // First, handle specific corrupted text patterns
        let cleanedText = text.toString();
        
        // Remove any weird character repetitions and fix common issues
        cleanedText = cleanedText
            .replace(/(.)\1{2,}/g, '$1') // Remove triple or more consecutive characters
            .replace(/[^a-zA-Z0-9\s_(),-]/g, '') // Remove special characters except basic ones
            .replace(/_+/g, ' ') // Replace underscores with spaces
            .replace(/\s+/g, ' ') // Replace multiple spaces with single space
            .trim(); // Remove leading/trailing spaces
        
        // Capitalize first letter of each word
        cleanedText = cleanedText.toLowerCase().replace(/\b\w/g, (char) => char.toUpperCase());
        
        return cleanedText || 'Unknown';
    }
}

// Utility functions for backward compatibility
window.analyzeImage = function() {
    if (window.plantDocApp) {
        window.plantDocApp.analyzeImage();
    }
};

window.clearImage = function() {
    if (window.plantDocApp) {
        window.plantDocApp.clearImage();
    }
};

window.handleImageUpload = function(event) {
    if (window.plantDocApp) {
        window.plantDocApp.handleImageUpload(event);
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.plantDocApp = new PlantDocApp();
    
    // Add loading screen fade out
    const loadingScreen = document.querySelector('.loading-screen');
    if (loadingScreen) {
        setTimeout(() => {
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }, 1000);
    }
    
    // Add particles effect to background (optional)
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: { value: 50 },
                color: { value: '#ffffff' },
                opacity: { value: 0.1 },
                size: { value: 3 },
                move: { enable: true, speed: 1 }
            }
        });
    }
});

// Add some CSS dynamically for enhanced effects
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        animation: fadeInUp 0.8s ease-out forwards;
    }
    
    .upload-feedback {
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
    }
    
    .diagnosis-icon.healthy {
        background: var(--gradient-success) !important;
    }
    
    .diagnosis-icon.disease {
        background: var(--gradient-error) !important;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.plantDocApp = new PlantDocApp();
    console.log('PlantDoc AI application initialized successfully!');
});