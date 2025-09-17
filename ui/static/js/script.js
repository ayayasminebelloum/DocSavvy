// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // File Upload Interaction
    const fileInput = document.getElementById('document');
    const uploadContainer = document.querySelector('.document-upload-container');
    const selectedFileDiv = document.querySelector('.selected-file');
    const fileNameSpan = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const uploadBtn = document.getElementById('upload-btn');

    // Only proceed if we're on the upload page
    if (fileInput) {
        // Handle file selection
        fileInput.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                const fileName = this.files[0].name;
                fileNameSpan.textContent = fileName;
                selectedFileDiv.classList.remove('d-none');
                uploadContainer.style.borderColor = '#28a745';
                uploadBtn.disabled = false;
            } else {
                resetFileUpload();
            }
        });

        // Handle drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadContainer.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            uploadContainer.style.borderColor = '#007bff';
            uploadContainer.style.backgroundColor = 'rgba(0, 123, 255, 0.1)';
        }

        function unhighlight() {
            uploadContainer.style.borderColor = '#ccc';
            uploadContainer.style.backgroundColor = 'transparent';
        }

        // Handle file drop
        uploadContainer.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                const fileName = files[0].name;
                fileNameSpan.textContent = fileName;
                selectedFileDiv.classList.remove('d-none');
                uploadContainer.style.borderColor = '#28a745';
                uploadBtn.disabled = false;
            }
        }

        // Handle remove file button
        if (removeFileBtn) {
            removeFileBtn.addEventListener('click', function() {
                resetFileUpload();
            });
        }

        function resetFileUpload() {
            fileInput.value = '';
            selectedFileDiv.classList.add('d-none');
            uploadContainer.style.borderColor = '#ccc';
            uploadContainer.style.backgroundColor = 'transparent';
            uploadBtn.disabled = true;
        }
    }

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Handle accordions in the extraction results
    const accordionButtons = document.querySelectorAll('.accordion-button');
    accordionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const target = document.querySelector(this.dataset.bsTarget);
            if (target) {
                const expanded = this.getAttribute('aria-expanded') === 'true';
                this.setAttribute('aria-expanded', !expanded);
                target.classList.toggle('show');
            }
        });
    });
}); 