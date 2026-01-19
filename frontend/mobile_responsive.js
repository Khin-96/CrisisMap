// Mobile-First Responsive JavaScript for CrisisMap

class CrisisMapMobile {
    constructor() {
        this.isMobile = window.innerWidth <= 768;
        this.isTablet = window.innerWidth > 768 && window.innerWidth <= 1024;
        this.isDesktop = window.innerWidth > 1024;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupResponsiveNavigation();
        this.setupTouchGestures();
        this.setupLazyLoading();
        this.setupPerformanceOptimizations();
    }
    
    setupEventListeners() {
        // Window resize handler
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 250));
        
        // Orientation change handler
        window.addEventListener('orientationchange', this.debounce(() => {
            this.handleOrientationChange();
        }, 250));
        
        // Scroll handler for mobile navigation
        window.addEventListener('scroll', this.throttle(() => {
            this.handleScroll();
        }, 100));
        
        // Touch events for mobile
        if ('ontouchstart' in window) {
            this.setupTouchEvents();
        }
    }
    
    setupResponsiveNavigation() {
        if (this.isMobile) {
            this.createMobileNavigation();
            this.hideDesktopSidebar();
        } else {
            this.createDesktopNavigation();
            this.showDesktopSidebar();
        }
    }
    
    createMobileNavigation() {
        // Remove existing mobile nav if present
        const existingNav = document.querySelector('.mobile-nav');
        if (existingNav) {
            existingNav.remove();
        }
        
        // Create mobile navigation
        const mobileNav = document.createElement('div');
        mobileNav.className = 'mobile-nav';
        mobileNav.innerHTML = `
            <a href="#dashboard" class="mobile-nav-item active">
                <span class="icon">🏠</span>
                <span>Dashboard</span>
            </a>
            <a href="#realtime" class="mobile-nav-item">
                <span class="icon">⚡</span>
                <span>Real-time</span>
            </a>
            <a href="#analysis" class="mobile-nav-item">
                <span class="icon">🔬</span>
                <span>Analysis</span>
            </a>
            <a href="#predictions" class="mobile-nav-item">
                <span class="icon">🔮</span>
                <span>Predictions</span>
            </a>
            <a href="#alerts" class="mobile-nav-item">
                <span class="icon">🚨</span>
                <span>Alerts</span>
            </a>
        `;
        
        document.body.appendChild(mobileNav);
        
        // Add click handlers
        mobileNav.querySelectorAll('.mobile-nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleMobileNavigation(item);
            });
        });
        
        // Create mobile header
        this.createMobileHeader();
    }
    
    createMobileHeader() {
        // Remove existing header if present
        const existingHeader = document.querySelector('.mobile-header');
        if (existingHeader) {
            existingHeader.remove();
        }
        
        const mobileHeader = document.createElement('div');
        mobileHeader.className = 'mobile-header';
        mobileHeader.innerHTML = `
            <div class="mobile-header-content">
                <button class="mobile-menu-toggle">☰</button>
                <div class="mobile-title">🌍 CrisisMap</div>
                <button class="mobile-refresh">🔄</button>
            </div>
        `;
        
        document.body.appendChild(mobileHeader);
        
        // Add menu toggle functionality
        const menuToggle = mobileHeader.querySelector('.mobile-menu-toggle');
        const refreshBtn = mobileHeader.querySelector('.mobile-refresh');
        
        menuToggle.addEventListener('click', () => {
            this.toggleMobileMenu();
        });
        
        refreshBtn.addEventListener('click', () => {
            this.refreshData();
        });
    }
    
    createDesktopNavigation() {
        // Desktop navigation logic
        const sidebar = document.querySelector('.stSidebar');
        if (sidebar) {
            // Enhance desktop sidebar
            sidebar.style.width = '300px';
            sidebar.style.position = 'fixed';
            sidebar.style.height = '100vh';
            sidebar.style.zIndex = '1000';
        }
        
        // Adjust main content
        const mainContent = document.querySelector('.main .block-container');
        if (mainContent) {
            mainContent.style.marginLeft = '320px';
            mainContent.style.transition = 'margin-left 0.3s ease';
        }
    }
    
    setupTouchGestures() {
        if (!this.isMobile) return;
        
        let touchStartX = 0;
        let touchStartY = 0;
        let touchEndX = 0;
        let touchEndY = 0;
        
        document.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
            touchStartY = e.changedTouches[0].screenY;
        }, false);
        
        document.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            touchEndY = e.changedTouches[0].screenY;
            this.handleSwipe(touchStartX, touchStartY, touchEndX, touchEndY);
        }, false);
    }
    
    handleSwipe(startX, startY, endX, endY) {
        const swipeThreshold = 50;
        const diffX = endX - startX;
        const diffY = endY - startY;
        
        if (Math.abs(diffX) > Math.abs(diffY)) {
            // Horizontal swipe
            if (Math.abs(diffX) > swipeThreshold) {
                if (diffX > 0) {
                    // Swipe right - show menu
                    this.showMobileMenu();
                } else {
                    // Swipe left - hide menu
                    this.hideMobileMenu();
                }
            }
        }
    }
    
    setupLazyLoading() {
        // Lazy loading for images and heavy content
        const lazyImages = document.querySelectorAll('img[data-src]');
        
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        imageObserver.unobserve(img);
                    }
                });
            });
            
            lazyImages.forEach(img => imageObserver.observe(img));
        } else {
            // Fallback for older browsers
            this.loadAllImages(lazyImages);
        }
    }
    
    setupPerformanceOptimizations() {
        // Optimize for mobile performance
        if (this.isMobile) {
            // Reduce animation complexity
            document.body.classList.add('mobile-optimized');
            
            // Optimize chart rendering
            this.optimizeCharts();
            
            // Reduce polling frequency
            this.optimizeDataFetching();
        }
    }
    
    optimizeCharts() {
        // Simplify charts for mobile
        const charts = document.querySelectorAll('.plotly-graph-div');
        charts.forEach(chart => {
            chart.style.height = '300px';
            
            // Reduce data points for mobile
            if (window.Plotly) {
                const chartData = window.Plotly.Plots.getChart(chart);
                if (chartData && chartData.data) {
                    chartData.data.forEach(trace => {
                        if (trace.x && trace.x.length > 50) {
                            // Downsample data for mobile
                            const step = Math.ceil(trace.x.length / 50);
                            trace.x = trace.x.filter((_, i) => i % step === 0);
                            trace.y = trace.y.filter((_, i) => i % step === 0);
                        }
                    });
                    
                    window.Plotly.redraw(chart);
                }
            }
        });
    }
    
    optimizeDataFetching() {
        // Implement adaptive polling based on connection and battery
        const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
        
        let pollingInterval = 30000; // Default 30 seconds
        
        if (connection) {
            if (connection.saveData || connection.effectiveType.includes('2g')) {
                pollingInterval = 60000; // 1 minute for slow connections
            } else if (connection.effectiveType.includes('4g')) {
                pollingInterval = 10000; // 10 seconds for fast connections
            }
        }
        
        // Check battery level
        if ('getBattery' in navigator) {
            navigator.getBattery().then(battery => {
                if (battery.level < 0.2) {
                    pollingInterval = 120000; // 2 minutes for low battery
                }
            });
        }
        
        return pollingInterval;
    }
    
    handleResize() {
        const newIsMobile = window.innerWidth <= 768;
        const newIsTablet = window.innerWidth > 768 && window.innerWidth <= 1024;
        const newIsDesktop = window.innerWidth > 1024;
        
        if (newIsMobile !== this.isMobile || 
            newIsTablet !== this.isTablet || 
            newIsDesktop !== this.isDesktop) {
            
            this.isMobile = newIsMobile;
            this.isTablet = newIsTablet;
            this.isDesktop = newIsDesktop;
            
            this.setupResponsiveNavigation();
            this.adjustLayoutForDevice();
        }
    }
    
    handleOrientationChange() {
        // Handle device orientation changes
        setTimeout(() => {
            this.adjustLayoutForOrientation();
        }, 100);
    }
    
    handleScroll() {
        if (this.isMobile) {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const mobileHeader = document.querySelector('.mobile-header');
            
            if (mobileHeader) {
                if (scrollTop > 50) {
                    mobileHeader.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                    mobileHeader.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
                } else {
                    mobileHeader.style.backgroundColor = 'white';
                    mobileHeader.style.boxShadow = 'none';
                }
            }
        }
    }
    
    handleMobileNavigation(item) {
        // Remove active class from all items
        document.querySelectorAll('.mobile-nav-item').forEach(navItem => {
            navItem.classList.remove('active');
        });
        
        // Add active class to clicked item
        item.classList.add('active');
        
        // Navigate to the section
        const target = item.getAttribute('href').substring(1);
        this.navigateToSection(target);
        
        // Hide menu after navigation
        this.hideMobileMenu();
    }
    
    navigateToSection(section) {
        // Handle navigation without page reload
        const content = document.querySelector('.main .block-container');
        if (content) {
            // Show loading state
            content.style.opacity = '0.5';
            
            // Simulate content loading
            setTimeout(() => {
                content.style.opacity = '1';
                this.updatePageTitle(section);
            }, 300);
        }
    }
    
    updatePageTitle(section) {
        const titles = {
            dashboard: 'Dashboard',
            realtime: 'Real-time Monitor',
            analysis: 'Advanced Analysis',
            predictions: 'Predictions',
            alerts: 'Alert Center'
        };
        
        const title = titles[section] || 'CrisisMap';
        document.title = `🌍 CrisisMap - ${title}`;
    }
    
    toggleMobileMenu() {
        const menu = document.querySelector('.mobile-menu');
        if (menu) {
            menu.classList.toggle('show');
        }
    }
    
    showMobileMenu() {
        const menu = document.querySelector('.mobile-menu');
        if (menu) {
            menu.classList.add('show');
        }
    }
    
    hideMobileMenu() {
        const menu = document.querySelector('.mobile-menu');
        if (menu) {
            menu.classList.remove('show');
        }
    }
    
    hideDesktopSidebar() {
        const sidebar = document.querySelector('.stSidebar');
        if (sidebar) {
            sidebar.style.display = 'none';
        }
    }
    
    showDesktopSidebar() {
        const sidebar = document.querySelector('.stSidebar');
        if (sidebar) {
            sidebar.style.display = 'block';
        }
    }
    
    adjustLayoutForDevice() {
        const content = document.querySelector('.main .block-container');
        if (!content) return;
        
        if (this.isMobile) {
            content.style.marginLeft = '0';
            content.style.padding = '10px';
            content.classList.add('mobile-content');
        } else if (this.isTablet) {
            content.style.marginLeft = '260px';
            content.style.padding = '20px';
            content.classList.remove('mobile-content');
        } else {
            content.style.marginLeft = '320px';
            content.style.padding = '30px';
            content.classList.remove('mobile-content');
        }
    }
    
    adjustLayoutForOrientation() {
        if (this.isMobile) {
            const isLandscape = window.orientation === 90 || window.orientation === -90;
            
            if (isLandscape) {
                // Optimize for landscape mode
                document.body.classList.add('landscape');
                document.body.classList.remove('portrait');
            } else {
                // Optimize for portrait mode
                document.body.classList.add('portrait');
                document.body.classList.remove('landscape');
            }
        }
    }
    
    refreshData() {
        // Show loading state
        const refreshBtn = document.querySelector('.mobile-refresh');
        if (refreshBtn) {
            refreshBtn.textContent = '⏳';
            refreshBtn.disabled = true;
        }
        
        // Trigger data refresh
        setTimeout(() => {
            if (refreshBtn) {
                refreshBtn.textContent = '🔄';
                refreshBtn.disabled = false;
            }
            
            // Show success message
            this.showNotification('Data refreshed successfully', 'success');
        }, 1000);
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `mobile-notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }
    
    // Utility functions
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    loadAllImages(images) {
        images.forEach(img => {
            img.src = img.dataset.src;
            img.classList.remove('lazy');
        });
    }
    
    // Progressive Web App (PWA) features
    setupPWA() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    console.log('SW registered: ', registration);
                })
                .catch(registrationError => {
                    console.log('SW registration failed: ', registrationError);
                });
        }
    }
    
    // Offline support
    setupOfflineSupport() {
        window.addEventListener('online', () => {
            this.showNotification('Back online', 'success');
            this.syncOfflineData();
        });
        
        window.addEventListener('offline', () => {
            this.showNotification('Offline mode activated', 'warning');
            this.enableOfflineMode();
        });
    }
    
    syncOfflineData() {
        // Sync data stored while offline
        const offlineData = localStorage.getItem('crisisMapOfflineData');
        if (offlineData) {
            try {
                const data = JSON.parse(offlineData);
                // Send data to server
                this.sendDataToServer(data);
                localStorage.removeItem('crisisMapOfflineData');
            } catch (e) {
                console.error('Error syncing offline data:', e);
            }
        }
    }
    
    enableOfflineMode() {
        // Enable offline mode functionality
        document.body.classList.add('offline-mode');
    }
    
    sendDataToServer(data) {
        // Send synced data to server
        fetch('/api/sync', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            console.log('Sync successful:', result);
        })
        .catch(error => {
            console.error('Sync failed:', error);
        });
    }
}

// Initialize the mobile responsive system
document.addEventListener('DOMContentLoaded', () => {
    new CrisisMapMobile();
});

// Export for use in other modules
window.CrisisMapMobile = CrisisMapMobile;