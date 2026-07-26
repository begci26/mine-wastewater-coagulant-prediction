class ToastNotification {
    constructor() {
        // Create container if not exists
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            this.container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            this.container.style.zIndex = '9999';
            document.body.appendChild(this.container);
        }
    }

    /**
     * Show a toast message.
     * @param {string} title - The title of the toast.
     * @param {string} message - The content of the toast.
     * @param {string} type - The alert type: 'success', 'danger', 'warning', 'info'.
     */
    show(title, message, type = 'info') {
        const toastId = 'toast_' + Date.now();
        let bgClass = 'bg-primary';
        let iconClass = 'fa-info-circle';
        let textClass = 'text-white';
        
        if (type === 'success') {
            bgClass = 'bg-success';
            iconClass = 'fa-check-circle';
        } else if (type === 'danger' || type === 'error') {
            bgClass = 'bg-danger';
            iconClass = 'fa-exclamation-circle';
        } else if (type === 'warning') {
            bgClass = 'bg-warning';
            iconClass = 'fa-exclamation-triangle';
            textClass = 'text-dark';
        }
        
        const toastHTML = `
            <div id="${toastId}" class="toast align-items-center ${textClass} ${bgClass} border-0 shadow-lg" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas ${iconClass} me-2"></i><strong>${title}</strong>: ${message}
                    </div>
                    <button type="button" class="btn-close ${type === 'warning' ? '' : 'btn-close-white'} me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        this.container.insertAdjacentHTML('beforeend', toastHTML);
        const toastElement = document.getElementById(toastId);
        
        // Initialize bootstrap toast
        const bsToast = new bootstrap.Toast(toastElement, {
            delay: 5000,
            autohide: true
        });
        bsToast.show();
        
        // Cleanup element from DOM when closed
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
}

// Instantiate globally
const toast = new ToastNotification();
window.toast = toast;
