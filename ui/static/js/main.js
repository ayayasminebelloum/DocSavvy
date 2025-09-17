/* 
 * DocClassifier 
 * Main JavaScript functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeUploadArea();
    
    // Remove debug log div
});

// Remove debug log function
function addDebugLog(message) {
    // Only log to console, not to UI
    console.log(message);
}

// File Upload Functionality
function initializeUploadArea() {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.querySelector('.file-input');
    const uploadForm = document.getElementById('upload-form');
    
    if (!uploadArea || !fileInput) {
        console.log('Error: Upload area or file input not found in the DOM');
        return;
    }
    
    console.log('Upload area initialized');
    
    // Drag & Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('drag-over');
        console.log('Drag over upload area');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('drag-over');
        console.log('Drag left upload area');
    }
    
    // Handle file drops
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        console.log('File dropped');
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        
        // Update file name display
        if (files.length > 0) {
            const fileName = files[0].name;
            console.log(`File selected: ${fileName}`);
            document.querySelector('.upload-file-name').textContent = fileName;
        }
    }
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        console.log('File selected via input');
        if (this.files.length > 0) {
            const fileName = this.files[0].name;
            console.log(`Selected file: ${fileName}`);
            document.querySelector('.upload-file-name').textContent = fileName;
        }
    });
    
    // Click on upload area triggers file input
    uploadArea.addEventListener('click', function() {
        console.log('Upload area clicked, triggering file input click');
        fileInput.click();
    });
}

// Utility function to show notifications
function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let notificationContainer = document.querySelector('.notification-container');
    
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.className = 'notification-container';
        document.body.appendChild(notificationContainer);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type} fade-in`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="notification-icon fas ${getIconForType(type)}"></i>
            <span class="notification-message">${message}</span>
        </div>
        <button class="notification-close" aria-label="Close">×</button>
    `;
    
    // Add notification to container
    notificationContainer.appendChild(notification);
    
    // Close button functionality
    notification.querySelector('.notification-close').addEventListener('click', function() {
        notification.remove();
    });
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
    
    function getIconForType(type) {
        switch(type) {
            case 'success': return 'fa-check-circle';
            case 'error': return 'fa-exclamation-circle';
            case 'warning': return 'fa-exclamation-triangle';
            default: return 'fa-info-circle';
        }
    }
} 