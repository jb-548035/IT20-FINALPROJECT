// ============================================================
// site.js  –  Global JavaScript for Shopping Channel Predictor
// Runs on every page after Bootstrap 5 bundle loads.
// ============================================================

'use strict';

// ── Auto-dismiss alerts after 5 seconds ──────────────────────────────
document.querySelectorAll('.alert.alert-success').forEach(el => {
    setTimeout(() => {
        const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
        bsAlert.close();
    }, 5000);
});

// ── Smooth scroll-to-top button (appears after scrolling 300px) ───────
(function () {
    const btn = document.createElement('button');
    btn.id = 'scrollTopBtn';
    btn.innerHTML = '<i class="bi bi-arrow-up"></i>';
    btn.className = 'btn btn-primary btn-sm rounded-circle shadow position-fixed';
    btn.style.cssText = 'bottom:1.5rem;right:1.5rem;width:40px;height:40px;display:none;z-index:999;';
    document.body.appendChild(btn);

    window.addEventListener('scroll', () => {
        btn.style.display = window.scrollY > 300 ? 'block' : 'none';
    });

    btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
})();

// ── Tooltip initialisation (Bootstrap 5) ─────────────────────────────
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el);
});

// ── Active nav-link highlight ─────────────────────────────────────────
(function () {
    const path = window.location.pathname.toLowerCase();
    document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
        const href = link.getAttribute('href')?.toLowerCase() ?? '';
        if (href && href !== '/' && path.startsWith(href)) {
            link.classList.add('active', 'fw-bold');
        } else if (href === '/' && path === '/') {
            link.classList.add('active', 'fw-bold');
        }
    });
})();
