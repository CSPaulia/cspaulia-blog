function closeAlert(element) {
    const alert = element.closest('.alert');
    if (!alert) return;
    
    alert.style.opacity = '0';
    alert.style.transform = 'translateY(-10px)';
    
    setTimeout(() => {
        alert.remove();
    }, 300);
}

function createAlert(type, title, message, options = {}) {
    const {
        closable = true,
        autoClose = false,
        autoCloseDelay = 5000,
        container = document.body
    } = options;

    const icons = {
        success: '✓',
        error: '!',
        warning: '⚠',
        info: 'i'
    };

    const icon = icons[type] || 'i';
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type}`;
    
    alertElement.innerHTML = `
        <div class="alert-icon">${icon}</div>
        <div class="alert-content">
            ${title ? `<div class="alert-title">${title}</div>` : ''}
            <div class="alert-message">${message}</div>
        </div>
        ${closable ? '<button class="alert-close" onclick="closeAlert(this)">&times;</button>' : ''}
    `;

    container.appendChild(alertElement);

    if (autoClose) {
        setTimeout(() => {
            if (alertElement.parentNode) {
                const closeBtn = alertElement.querySelector('.alert-close');
                if (closeBtn) {
                    closeAlert(closeBtn);
                }
            }
        }, autoCloseDelay);
    }

    return alertElement;
}

window.showAlert = {
    success: (title, message, options) => createAlert('success', title, message, options),
    error: (title, message, options) => createAlert('error', title, message, options),
    warning: (title, message, options) => createAlert('warning', title, message, options),
    info: (title, message, options) => createAlert('info', title, message, options)
};

document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const alerts = document.querySelectorAll('.alert');
            alerts.forEach(alert => {
                const closeBtn = alert.querySelector('.alert-close');
                if (closeBtn) {
                    closeAlert(closeBtn);
                }
            });
        }
    });
});