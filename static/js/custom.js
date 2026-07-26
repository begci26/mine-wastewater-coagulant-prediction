/**
 * Custom JavaScript Utilities
 * Coagulant Dosage Predictor - Enterprise Software House Style
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Sidebar Active State Helper
    const currentPath = window.location.pathname;
    const sidebarLinks = document.querySelectorAll('.nav-sidebar .nav-link');
    
    sidebarLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && currentPath.startsWith(href) && href !== '/' || (href === '/' && currentPath === '/')) {
            link.classList.add('active');
            
            // If inside a dropdown tree, open the tree
            const treeview = link.closest('.has-treeview');
            if (treeview) {
                treeview.classList.add('menu-open');
                const mainLink = treeview.querySelector('.nav-link');
                if (mainLink) mainLink.classList.add('active');
            }
        }
    });

    // 2. Auto-initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

/**
 * Show a full-page loading spinner.
 * @param {string} message - Optional loading message.
 */
function showLoading(message = "Processing, please wait...") {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay d-flex flex-column justify-content-center align-items-center position-fixed top-0 start-0 w-100 h-100';
        overlay.style.zIndex = '99999';
        overlay.style.backgroundColor = 'rgba(255, 255, 255, 0.85)';
        overlay.innerHTML = `
            <div class="spinner-border text-primary mb-3" style="width: 3rem; height: 3rem;" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="h5 text-secondary fw-semibold text-center px-3" id="loading-overlay-text">${message}</div>
        `;
        document.body.appendChild(overlay);
    } else {
        document.getElementById('loading-overlay-text').innerText = message;
        overlay.classList.remove('d-none');
        overlay.classList.add('d-flex');
    }
}

/**
 * Hide the loading spinner.
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('d-flex');
        overlay.classList.add('d-none');
    }
}

/**
 * API fetch wrapper with automatic error handling.
 * @param {string} url - Request URL.
 * @param {object} options - Request options (headers, body, method, etc.).
 * @returns {Promise<any>}
 */
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || `Request failed with status ${response.status}`);
        }
        return data;
    } catch (error) {
        console.error('API Error:', error);
        if (window.toast) {
            window.toast.show('System Error', error.message || 'An unexpected error occurred.', 'danger');
        }
        throw error;
    }
}
