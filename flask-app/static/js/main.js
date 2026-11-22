(function () {
    'use strict';

    // =========================================================================
    // 1. STATE & DOM REFERENCES
    // =========================================================================
    const state = {
        activeTab: 'predictor',
        sidebarCollapsed: false,
        mobileDrawerOpen: false,
        analyticsLoaded: false,
        apartmentsList: []
    };

    const sidebar = document.getElementById('app-sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileCloseBtn = document.getElementById('mobile-drawer-close');
    const collapseBtn = document.getElementById('sidebar-collapse-btn');
    const mobileActiveLabel = document.getElementById('mobile-active-label');

    const TAB_TITLES = {
        predictor: 'Price Predictor',
        analytics: 'Market Analytics',
        recommender: 'Recommender'
    };

    const PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: false
    };

    // =========================================================================
    // 2. TOASTS & API STATUS INDICATORS
    // =========================================================================
    function showToast(message, tone = 'indigo') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const styles = {
            indigo: 'border-indigo-400/40 text-indigo-100 bg-slate-900/95',
            emerald: 'border-emerald-400/40 text-emerald-100 bg-slate-900/95',
            cyan: 'border-cyan-400/40 text-cyan-100 bg-slate-900/95',
            rose: 'border-rose-400/40 text-rose-100 bg-slate-900/95'
        };
        const dots = {
            indigo: 'bg-indigo-400',
            emerald: 'bg-emerald-400',
            cyan: 'bg-cyan-400',
            rose: 'bg-rose-400'
        };

        const el = document.createElement('div');
        el.className = `toast-item pointer-events-auto flex items-center gap-2.5 px-4 py-2.5 rounded-xl border shadow-2xl text-xs font-medium backdrop-blur-xl ${styles[tone] || styles.indigo}`;
        el.innerHTML = `<span class="w-2 h-2 rounded-full ${dots[tone] || dots.indigo} shrink-0"></span><span>${escapeHtml(message)}</span>`;
        container.appendChild(el);

        setTimeout(() => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(8px)';
            el.style.transition = 'all 240ms ease';
            setTimeout(() => el.remove(), 250);
        }, 3000);
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function setApiStatus(connected, detailText) {
        const badge = document.getElementById('api-status-badge');
        const dot = document.getElementById('api-status-dot');
        const text = document.getElementById('api-status-text');
        const sub = document.getElementById('api-status-sub');
        const miniDot = document.getElementById('api-status-dot-mini');
        const mobilePill = document.getElementById('mobile-api-pill');

        if (connected) {
            if (badge) badge.className = 'inline-flex items-center gap-2 text-xs font-semibold text-emerald-400';
            if (dot) dot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
            if (text) text.textContent = 'Flask API Connected';
            if (sub) sub.textContent = detailText || 'Live endpoints ready.';
            if (miniDot) miniDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.85)]';
            if (mobilePill) {
                mobilePill.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/25';
                mobilePill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>Live';
            }
        } else {
            if (badge) badge.className = 'inline-flex items-center gap-2 text-xs font-semibold text-rose-400';
            if (dot) dot.className = 'w-2 h-2 rounded-full bg-rose-400';
            if (text) text.textContent = 'API Unreachable';
            if (sub) sub.textContent = detailText || 'Start Flask server on port 5000.';
            if (miniDot) miniDot.className = 'w-2.5 h-2.5 rounded-full bg-rose-400';
            if (mobilePill) {
                mobilePill.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-rose-500/10 text-rose-300 border border-rose-500/25';
                mobilePill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-rose-400"></span>Offline';
            }
        }
    }

    // =========================================================================
    // 3. INTERACTIVE SIDEBAR (DESKTOP MINI-RAIL + MOBILE DRAWER + SWIPE)
    // =========================================================================
    function setDesktopCollapsed(collapsed) {
        state.sidebarCollapsed = collapsed;
        sidebar.classList.toggle('sidebar-collapsed', collapsed);
        try {
            localStorage.setItem('gurgaon_re_sidebar_collapsed', collapsed ? '1' : '0');
        } catch (_) { /* ignore */ }
        setTimeout(resizePlotlyCharts, 340);
    }

    function setMobileDrawer(open) {
        state.mobileDrawerOpen = open;
        if (open) {
            sidebar.classList.remove('-translate-x-full');
            backdrop.classList.remove('opacity-0', 'pointer-events-none');
            mobileMenuBtn.classList.add('hamburger-open');
            mobileMenuBtn.setAttribute('aria-expanded', 'true');
        } else {
            sidebar.classList.add('-translate-x-full');
            backdrop.classList.add('opacity-0', 'pointer-events-none');
            mobileMenuBtn.classList.remove('hamburger-open');
            mobileMenuBtn.setAttribute('aria-expanded', 'false');
        }
    }

    if (collapseBtn) collapseBtn.addEventListener('click', () => setDesktopCollapsed(!state.sidebarCollapsed));
    if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', () => setMobileDrawer(!state.mobileDrawerOpen));
    if (mobileCloseBtn) mobileCloseBtn.addEventListener('click', () => setMobileDrawer(false));
    if (backdrop) backdrop.addEventListener('click', () => setMobileDrawer(false));

    // Swipe left to close drawer on touch devices
    let touchStartX = null;
    sidebar.addEventListener('touchstart', (e) => {
        if (window.innerWidth < 1024 && e.touches.length === 1) {
            touchStartX = e.touches[0].clientX;
        }
    }, { passive: true });
    sidebar.addEventListener('touchend', (e) => {
        if (touchStartX !== null && e.changedTouches.length === 1) {
            if (e.changedTouches[0].clientX - touchStartX < -55) setMobileDrawer(false);
        }
        touchStartX = null;
    }, { passive: true });

    // Keyboard shortcuts: Escape, Ctrl/Cmd+B, 1/2/3
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.mobileDrawerOpen) {
            setMobileDrawer(false);
            return;
        }
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
            e.preventDefault();
            if (window.innerWidth >= 1024) {
                setDesktopCollapsed(!state.sidebarCollapsed);
            } else {
                setMobileDrawer(!state.mobileDrawerOpen);
            }
            return;
        }
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
        if (e.key === '1') switchTab('predictor');
        if (e.key === '2') switchTab('analytics');
        if (e.key === '3') switchTab('recommender');
    });

    try {
        if (localStorage.getItem('gurgaon_re_sidebar_collapsed') === '1' && window.innerWidth >= 1024) {
            sidebar.classList.add('sidebar-collapsed');
            state.sidebarCollapsed = true;
        }
    } catch (_) { /* ignore */ }

    // =========================================================================
    // 4. TAB SWITCHING & PLOTLY RESIZE
    // =========================================================================
    function switchTab(tabName) {
        if (!TAB_TITLES[tabName]) return;
        state.activeTab = tabName;

        ['predictor', 'analytics', 'recommender'].forEach((name) => {
            const sec = document.getElementById(`${name}-section`);
            const btn = document.getElementById(`tab-${name}`);
            const isActive = name === tabName;
            if (sec) {
                sec.classList.toggle('hidden-section', !isActive);
                if (isActive) {
                    sec.classList.remove('section-enter');
                    void sec.offsetWidth;
                    sec.classList.add('section-enter');
                }
            }
            if (btn) {
                btn.classList.toggle('is-active', isActive);
                btn.classList.toggle('text-slate-400', !isActive);
                btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
            }
        });

        document.querySelectorAll('[data-mobile-tab]').forEach((btn) => {
            const active = btn.getAttribute('data-mobile-tab') === tabName;
            btn.classList.toggle('text-indigo-400', active);
            btn.classList.toggle('bg-indigo-500/15', active);
            btn.classList.toggle('text-slate-400', !active);
        });
        if (mobileActiveLabel) mobileActiveLabel.textContent = TAB_TITLES[tabName];

        if (window.innerWidth < 1024) setMobileDrawer(false);

        if (tabName === 'analytics') {
            if (!state.analyticsLoaded) {
                loadAllAnalytics();
            } else {
                setTimeout(resizePlotlyCharts, 140);
            }
        }
    }
    window.switchTab = switchTab;

    document.querySelectorAll('[data-tab]').forEach((btn) => {
        btn.addEventListener('click', () => switchTab(btn.getAttribute('data-tab')));
    });
    document.querySelectorAll('[data-mobile-tab]').forEach((btn) => {
        btn.addEventListener('click', () => switchTab(btn.getAttribute('data-mobile-tab')));
    });

    function resizePlotlyCharts() {
        if (state.activeTab !== 'analytics') return;
        ['plot-geomap', 'plot-scatter', 'plot-pie', 'plot-boxplot'].forEach((id) => {
            const el = document.getElementById(id);
            if (el && el.data) Plotly.Plots.resize(el);
        });
    }

    window.addEventListener('resize', () => {
        if (window.innerWidth >= 1024 && state.mobileDrawerOpen) setMobileDrawer(false);
        resizePlotlyCharts();
    });

    // =========================================================================
    // 5. LOAD DROPDOWN OPTIONS FROM GET /api/options
    // =========================================================================
    function populateSelect(id, values, labelFormatter) {
        const select = document.getElementById(id);
        if (!select || !Array.isArray(values)) return;
        select.innerHTML = '';
        values.forEach((val) => {
            const option = document.createElement('option');
            option.value = val;
            option.textContent = labelFormatter ? labelFormatter(val) : String(val);
            select.appendChild(option);
        });
    }

    function formatFurnishingLabel(val) {
        const num = Number(val);
        if (num === 0) return 'Unfurnished (0)';
        if (num === 1) return 'Semi-Furnished (1)';
        if (num === 2) return 'Furnished (2)';
        return String(val);
    }

    async function loadOptions() {
        try {
            const response = await fetch('/api/options');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();

            populateSelect('p_sector', data.sectors);
            populateSelect('p_bed', data.bedrooms);
            populateSelect('p_bath', data.bathrooms);
            populateSelect('p_balc', data.balconies);
            populateSelect('p_age', data.property_age);
            populateSelect('p_furnish', data.furnishing_type, formatFurnishingLabel);
            populateSelect('p_luxury', data.luxury_category);
            populateSelect('p_floor', data.floor_category);

            populateSelect('plot-pie-select', data.analytics_sectors);
            populateSelect('r_location', data.locations);
            populateSelect('r_apartment', data.apartments);

            state.apartmentsList = Array.isArray(data.apartments) ? data.apartments : [];

            setApiStatus(
                true,
                `${(data.sectors || []).length} sectors & ${state.apartmentsList.length} properties loaded.`
            );
        } catch (err) {
            console.error('Failed to fetch /api/options:', err);
            setApiStatus(false, 'Could not load /api/options. Check Flask server.');
            showToast('Failed to connect to Flask /api/options', 'rose');
        }
    }

    // =========================================================================
    // 6. PRICE PREDICTOR — POST /api/predict
    // =========================================================================
    function animatePriceRange(el, low, high, duration = 480) {
        const startTime = performance.now();
        function step(now) {
            const t = Math.min((now - startTime) / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3);
            el.textContent = `${(low * ease).toFixed(2)} – ${(high * ease).toFixed(2)}`;
            if (t < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const btn = document.getElementById('predict-submit-btn');
            const btnText = document.getElementById('predict-text');
            const spinner = document.getElementById('predict-spinner');
            const resultBox = document.getElementById('prediction-result');
            const errorBox = document.getElementById('prediction-error');
            const errorText = document.getElementById('prediction-error-text');

            errorBox.classList.add('hidden');
            btn.disabled = true;
            btnText.textContent = 'Predicting...';
            spinner.classList.remove('hidden');

            const builtUpArea = parseFloat(document.getElementById('p_area').value);

            // Payload keys match app.py `/api/predict` required_fields exactly
            const payload = {
                property_type: document.getElementById('p_type').value,
                sector: document.getElementById('p_sector').value,
                bedrooms: parseFloat(document.getElementById('p_bed').value),
                bathroom: parseFloat(document.getElementById('p_bath').value),
                balcony: document.getElementById('p_balc').value,
                property_age: document.getElementById('p_age').value,
                built_up_area: builtUpArea,
                servant_room: parseFloat(document.getElementById('p_servant').value),
                store_room: parseFloat(document.getElementById('p_store').value),
                furnishing_type: parseFloat(document.getElementById('p_furnish').value),
                luxury_category: document.getElementById('p_luxury').value,
                floor_category: document.getElementById('p_floor').value
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();

                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || `Prediction failed (HTTP ${response.status})`);
                }

                const low = Number(data.low);
                const high = Number(data.high);
                const mid = (low + high) / 2;
                const ppsqft = builtUpArea > 0 ? Math.round((mid * 10000000) / builtUpArea) : 0;

                resultBox.classList.remove('hidden');
                resultBox.classList.remove('section-enter');
                void resultBox.offsetWidth;
                resultBox.classList.add('section-enter');

                animatePriceRange(document.getElementById('price-output'), low, high);
                document.getElementById('result-ppsqft').textContent = ppsqft > 0 ? `₹${ppsqft.toLocaleString('en-IN')}/sqft` : '—';
                document.getElementById('result-midpoint').textContent = `₹${mid.toFixed(2)} Cr`;
                document.getElementById('price-meta-line').textContent = `${payload.bedrooms} BHK ${payload.property_type} • ${builtUpArea.toLocaleString('en-IN')} sq.ft in ${payload.sector}`;

                const tier = mid >= 6 ? 'Ultra-Luxury' : mid >= 2.5 ? 'Premium' : 'Mid-Segment';
                document.getElementById('result-tier-badge').textContent = tier;

                const pct = Math.min(100, Math.max(5, ((mid - 0.5) / (12 - 0.5)) * 100));
                document.getElementById('result-position-bar').style.width = `${pct.toFixed(1)}%`;
                document.getElementById('result-position-label').textContent = `Midpoint ₹${mid.toFixed(2)} Cr (${tier})`;

                showToast(`Predicted: ₹${low.toFixed(2)} – ₹${high.toFixed(2)} Cr`, 'emerald');
            } catch (err) {
                console.error('Prediction error:', err);
                errorText.textContent = err.message || 'Could not compute prediction.';
                errorBox.classList.remove('hidden');
                showToast(err.message || 'Prediction failed', 'rose');
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Predict Price';
                spinner.classList.add('hidden');
            }
        });
    }

    const resetPredictBtn = document.getElementById('reset-predict-btn');
    if (resetPredictBtn) {
        resetPredictBtn.addEventListener('click', () => {
            predictForm.reset();
            document.getElementById('prediction-result').classList.add('hidden');
            document.getElementById('prediction-error').classList.add('hidden');
            showToast('Predictor form reset.', 'indigo');
        });
    }

    // =========================================================================
    // 7. ANALYTICS APIs (/api/analytics/*)
    // =========================================================================
    function toggleLoader(loaderId, show) {
        const el = document.getElementById(loaderId);
        if (el) el.classList.toggle('hidden', !show);
    }

    function showChartError(containerId, message) {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = `<div class="h-full min-h-[180px] flex items-center justify-center text-xs text-rose-300 px-4 text-center">${escapeHtml(message)}</div>`;
    }

    async function renderPlotlyFromEndpoint(url, containerId, loaderId, extraLayout = {}) {
        toggleLoader(loaderId, true);
        try {
            const response = await fetch(url);
            const fig = await response.json();
            if (!response.ok || fig.status === 'error') {
                throw new Error(fig.message || `Failed to load ${url}`);
            }
            const mergedLayout = {
                ...fig.layout,
                autosize: true,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter, system-ui, sans-serif', color: '#cbd5e1' },
                ...extraLayout
            };
            await Plotly.newPlot(containerId, fig.data, mergedLayout, PLOTLY_CONFIG);
        } catch (err) {
            console.error(`Analytics error (${url}):`, err);
            showChartError(containerId, err.message || 'Failed to render chart.');
        } finally {
            toggleLoader(loaderId, false);
        }
    }

    async function loadGeomap() {
        return renderPlotlyFromEndpoint('/api/analytics/geomap', 'plot-geomap', 'geomap-loader', {
            margin: { l: 0, r: 0, t: 0, b: 0 }
        });
    }

    async function loadScatterPlot() {
        const propertyType = document.getElementById('plot-scatter-select').value || 'flat';
        return renderPlotlyFromEndpoint(
            `/api/analytics/scatter?property_type=${encodeURIComponent(propertyType)}`,
            'plot-scatter',
            'scatter-loader',
            { margin: { l: 50, r: 16, t: 36, b: 44 } }
        );
    }
    window.loadScatterPlot = loadScatterPlot;

    async function loadPieChart() {
        const sector = document.getElementById('plot-pie-select').value || 'overall';
        return renderPlotlyFromEndpoint(
            `/api/analytics/pie?sector=${encodeURIComponent(sector)}`,
            'plot-pie',
            'pie-loader',
            { margin: { l: 16, r: 16, t: 36, b: 16 } }
        );
    }
    window.loadPieChart = loadPieChart;

    async function loadBoxPlot() {
        return renderPlotlyFromEndpoint(
            '/api/analytics/boxplot',
            'plot-boxplot',
            'boxplot-loader',
            { margin: { l: 50, r: 16, t: 36, b: 44 } }
        );
    }

    async function loadBase64Image(url, imgId, loaderId) {
        const img = document.getElementById(imgId);
        toggleLoader(loaderId, true);
        if (img) img.classList.add('hidden');
        try {
            const response = await fetch(url);
            const data = await response.json();
            if (!response.ok || data.status === 'error' || !data.image) {
                throw new Error(data.message || `Failed to load ${url}`);
            }
            if (img) {
                img.src = `data:image/png;base64,${data.image}`;
                img.classList.remove('hidden');
            }
        } catch (err) {
            console.error(`Image API error (${url}):`, err);
            const loader = document.getElementById(loaderId);
            if (loader) {
                loader.innerHTML = `<span class="text-xs text-rose-300">${escapeHtml(err.message)}</span>`;
                loader.classList.remove('hidden');
                return;
            }
        }
        toggleLoader(loaderId, false);
    }

    async function loadAllAnalytics() {
        state.analyticsLoaded = true;
        await Promise.all([
            loadGeomap(),
            loadScatterPlot(),
            loadPieChart(),
            loadBoxPlot(),
            loadBase64Image('/api/analytics/distplot', 'plot-distplot', 'distplot-loader'),
            loadBase64Image('/api/analytics/wordcloud', 'plot-wordcloud', 'wordcloud-loader')
        ]);
        setTimeout(resizePlotlyCharts, 120);
    }

    const scatterSelect = document.getElementById('plot-scatter-select');
    if (scatterSelect) scatterSelect.addEventListener('change', loadScatterPlot);

    const pieSelect = document.getElementById('plot-pie-select');
    if (pieSelect) pieSelect.addEventListener('change', loadPieChart);

    const refreshAnalyticsBtn = document.getElementById('refresh-analytics-btn');
    if (refreshAnalyticsBtn) {
        refreshAnalyticsBtn.addEventListener('click', () => {
            loadAllAnalytics();
            showToast('Reloading analytics charts...', 'cyan');
        });
    }

    // =========================================================================
    // 8. RECOMMENDER APIs — POST /api/recommend/search & /api/recommend/apartments
    // =========================================================================
    const radiusSlider = document.getElementById('r_radius_slider');
    const radiusInput = document.getElementById('r_radius');
    const radiusBadge = document.getElementById('radius-live-badge');

    function updateRadiusBadge(val) {
        const num = parseFloat(val);
        if (radiusBadge && Number.isFinite(num)) {
            radiusBadge.textContent = `${num.toFixed(1)} km`;
        }
    }

    if (radiusSlider && radiusInput) {
        radiusSlider.addEventListener('input', () => {
            radiusInput.value = radiusSlider.value;
            updateRadiusBadge(radiusSlider.value);
        });
        radiusInput.addEventListener('input', () => {
            const val = parseFloat(radiusInput.value);
            if (Number.isFinite(val) && val <= 20) radiusSlider.value = val;
            updateRadiusBadge(val);
        });
    }

    const radiusForm = document.getElementById('radius-form');
    if (radiusForm) {
        radiusForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById('radius-submit-btn');
            const resultsDiv = document.getElementById('radius-results');
            const location = document.getElementById('r_location').value;
            const radius = parseFloat(document.getElementById('r_radius').value);

            submitBtn.disabled = true;
            try {
                const response = await fetch('/api/recommend/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ location, radius })
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || 'Location search failed');
                }

                const items = Array.isArray(data.results) ? data.results : [];
                resultsDiv.classList.remove('hidden');
                resultsDiv.innerHTML = '';

                if (items.length === 0) {
                    resultsDiv.innerHTML = `
                        <div class="col-span-full rounded-xl bg-slate-900/70 border border-slate-800 p-5 text-center text-sm text-slate-400">
                            No properties found within <span class="text-white font-semibold">${radius} km</span> of ${escapeHtml(location)}.
                        </div>`;
                    return;
                }

                items.forEach((item, idx) => {
                    const card = document.createElement('div');
                    card.className = 'stagger-item rounded-xl bg-slate-900/75 border border-slate-700/80 hover:border-cyan-400/50 p-4 flex flex-col justify-between gap-3 transition-all';
                    card.style.animationDelay = `${Math.min(idx * 45, 450)}ms`;

                    const canRecommend = state.apartmentsList.some(
                        (a) => a.toLowerCase() === String(item.name).toLowerCase()
                    );

                    card.innerHTML = `
                        <div class="flex items-start justify-between gap-2">
                            <h4 class="text-sm font-bold text-white leading-snug">${escapeHtml(item.name)}</h4>
                            <span class="px-2 py-0.5 rounded-full text-[11px] font-bold bg-cyan-500/15 text-cyan-300 border border-cyan-400/25 shrink-0">${escapeHtml(item.distance)} km</span>
                        </div>
                        ${canRecommend ? `<button type="button" data-find-similar="${escapeHtml(item.name)}" class="w-full py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 text-xs font-semibold text-slate-200 hover:text-white transition-colors">Find Similar Properties →</button>` : ''}
                    `;
                    resultsDiv.appendChild(card);
                });

                resultsDiv.querySelectorAll('[data-find-similar]').forEach((btn) => {
                    btn.addEventListener('click', () => {
                        const propName = btn.getAttribute('data-find-similar');
                        const aptSelect = document.getElementById('r_apartment');
                        if (aptSelect) {
                            aptSelect.value = propName;
                            recommendForm.dispatchEvent(new Event('submit', { cancelable: true }));
                            aptSelect.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                    });
                });

                showToast(`Found ${items.length} propert${items.length === 1 ? 'y' : 'ies'} within ${radius} km`, 'cyan');
            } catch (err) {
                console.error('Radius search error:', err);
                showToast(err.message || 'Location search failed', 'rose');
            } finally {
                submitBtn.disabled = false;
            }
        });
    }

    const recommendForm = document.getElementById('recommend-form');
    if (recommendForm) {
        recommendForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById('recommend-submit-btn');
            const tableContainer = document.getElementById('recommend-table-container');
            const tbody = document.getElementById('recommend-tbody');
            const apartment = document.getElementById('r_apartment').value;

            submitBtn.disabled = true;
            try {
                const response = await fetch('/api/recommend/apartments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ apartment })
                });
                const data = await response.json();
                if (!response.ok || data.status === 'error') {
                    throw new Error(data.message || 'Recommendation request failed');
                }

                const items = Array.isArray(data.results) ? data.results : [];
                tableContainer.classList.remove('hidden');
                tbody.innerHTML = '';

                if (items.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="3" class="px-5 py-6 text-center text-sm text-slate-400">No similar apartments returned.</td></tr>`;
                    return;
                }

                items.forEach((item, idx) => {
                    const rawScore = Number(item.score);
                    const pct = Number.isFinite(rawScore)
                        ? Math.min(100, Math.max(0, Math.round((rawScore <= 1 ? rawScore * 100 : rawScore))))
                        : 0;

                    const tr = document.createElement('tr');
                    tr.className = 'stagger-item hover:bg-slate-800/60 transition-colors';
                    tr.style.animationDelay = `${idx * 60}ms`;
                    tr.innerHTML = `
                        <td class="px-5 py-4 text-xs font-mono text-slate-500">#${idx + 1}</td>
                        <td class="px-5 py-4 text-sm font-bold text-white">${escapeHtml(item.property)}</td>
                        <td class="px-5 py-4 min-w-[200px]">
                            <div class="flex items-center gap-3">
                                <div class="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
                                    <div class="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-400" style="width: ${pct}%"></div>
                                </div>
                                <span class="text-xs font-bold text-emerald-300 font-mono w-14 text-right">${Number.isFinite(rawScore) ? rawScore.toFixed(4) : '—'}</span>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });

                showToast(`Top ${items.length} recommendations loaded`, 'emerald');
            } catch (err) {
                console.error('Recommender error:', err);
                showToast(err.message || 'Could not load recommendations', 'rose');
            } finally {
                submitBtn.disabled = false;
            }
        });
    }

    // =========================================================================
    // 9. INITIALIZE APP
    // =========================================================================
    loadOptions();
})();
