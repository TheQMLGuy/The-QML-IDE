/**
 * Tab Manager - Chrome-style Tabbed Interface
 * Each tab contains one dataset + one model (ML or DL)
 */

// ===========================
// Tab State
// ===========================

const TabState = {
    tabs: [],
    activeTabId: null,
    tabCounter: 0
};

// ===========================
// Tab Class
// ===========================

class Tab {
    constructor(id, name) {
        this.id = id;
        this.name = name;
        this.dataset = null;      // { data: [], columns: [], dtypes: {}, name: '' }
        this.model = null;        // { type: 'ml'|'dl', algorithm: '', params: {} }
        this.cells = [];          // Cell IDs belonging to this tab
        this.analysisResults = null; // Cached analysis results
        this.mode = 'ml';         // Default mode for this tab
        this.createdAt = Date.now();
    }

    setDataset(dataset) {
        this.dataset = dataset;
        this.analysisResults = null; // Reset analysis when dataset changes
    }

    setModel(model) {
        this.model = model;
    }

    addCell(cellId) {
        if (!this.cells.includes(cellId)) {
            this.cells.push(cellId);
        }
    }

    removeCell(cellId) {
        this.cells = this.cells.filter(id => id !== cellId);
    }
}

// ===========================
// Tab Manager
// ===========================

const TabManager = {
    init() {
        console.log('Initializing Tab Manager...');
        this.setupTabBar();
        this.setupEventListeners();

        // Create initial tab
        if (TabState.tabs.length === 0) {
            this.createTab('Workspace 1');
        }

        console.log('Tab Manager initialized!');
    },

    setupTabBar() {
        const tabBar = document.getElementById('tabBar');
        if (tabBar) {
            tabBar.innerHTML = `
                <div class="tabs-container" id="tabsContainer"></div>
                <button class="new-tab-btn" id="newTabBtn" title="New Tab (Ctrl+T)">
                    <span>+</span>
                </button>
            `;
        }
    },

    setupEventListeners() {
        // New tab button
        const newTabBtn = document.getElementById('newTabBtn');
        if (newTabBtn) {
            newTabBtn.addEventListener('click', () => this.createTab());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 't') {
                e.preventDefault();
                this.createTab();
            }
            if (e.ctrlKey && e.key === 'w') {
                e.preventDefault();
                if (TabState.activeTabId && TabState.tabs.length > 1) {
                    this.closeTab(TabState.activeTabId);
                }
            }
        });
    },

    createTab(name = null) {
        const id = ++TabState.tabCounter;
        const tabName = name || `Workspace ${id}`;
        const tab = new Tab(id, tabName);

        TabState.tabs.push(tab);
        this.switchTab(id);
        this.renderTabs();

        // Create initial cell for this tab
        if (typeof addCell === 'function') {
            const cellId = addCell();
            tab.addCell(cellId);
        }

        return tab;
    },

    closeTab(tabId) {
        const tabIndex = TabState.tabs.findIndex(t => t.id === tabId);
        if (tabIndex === -1) return;

        // Don't close the last tab
        if (TabState.tabs.length <= 1) {
            console.log('Cannot close the last tab');
            return;
        }

        const tab = TabState.tabs[tabIndex];

        // Remove cells belonging to this tab
        tab.cells.forEach(cellId => {
            if (typeof deleteCell === 'function') {
                deleteCell(cellId);
            }
        });

        TabState.tabs.splice(tabIndex, 1);

        // Switch to another tab if closing active
        if (TabState.activeTabId === tabId) {
            const newActiveTab = TabState.tabs[Math.min(tabIndex, TabState.tabs.length - 1)];
            this.switchTab(newActiveTab.id);
        }

        this.renderTabs();
    },

    switchTab(tabId) {
        const tab = TabState.tabs.find(t => t.id === tabId);
        if (!tab) return;

        const previousTabId = TabState.activeTabId;
        TabState.activeTabId = tabId;

        // Hide cells from previous tab
        if (previousTabId && previousTabId !== tabId) {
            const previousTab = TabState.tabs.find(t => t.id === previousTabId);
            if (previousTab) {
                previousTab.cells.forEach(cellId => {
                    const cellEl = document.getElementById(`cell-${cellId}`);
                    if (cellEl) cellEl.style.display = 'none';
                });
            }
        }

        // Show cells from current tab
        tab.cells.forEach(cellId => {
            const cellEl = document.getElementById(`cell-${cellId}`);
            if (cellEl) cellEl.style.display = '';
        });

        // Switch to tab's mode
        if (typeof switchMode === 'function') {
            switchMode(tab.mode);
        }

        // Update tab bar styling
        this.renderTabs();

        // Update sidebar with tab's dataset if present
        if (tab.dataset && typeof DataAnalysis !== 'undefined') {
            DataAnalysis.displayResults(tab.analysisResults);
        }

        console.log(`Switched to tab: ${tab.name}`);
    },

    renameTab(tabId, newName) {
        const tab = TabState.tabs.find(t => t.id === tabId);
        if (tab) {
            tab.name = newName;
            this.renderTabs();
        }
    },

    renderTabs() {
        const container = document.getElementById('tabsContainer');
        if (!container) return;

        container.innerHTML = TabState.tabs.map(tab => `
            <div class="tab ${tab.id === TabState.activeTabId ? 'active' : ''}" 
                 data-tab-id="${tab.id}"
                 title="${tab.name}${tab.dataset ? ' - ' + tab.dataset.name : ''}">
                <span class="tab-icon">${this.getTabIcon(tab)}</span>
                <span class="tab-name">${tab.name}</span>
                ${tab.dataset ? `<span class="tab-dataset-badge">${tab.dataset.name}</span>` : ''}
                <button class="tab-close" data-tab-id="${tab.id}" title="Close Tab">×</button>
            </div>
        `).join('');

        // Add click listeners
        container.querySelectorAll('.tab').forEach(tabEl => {
            tabEl.addEventListener('click', (e) => {
                if (!e.target.classList.contains('tab-close')) {
                    const tabId = parseInt(tabEl.dataset.tabId);
                    this.switchTab(tabId);
                }
            });

            // Double-click to rename
            tabEl.addEventListener('dblclick', (e) => {
                if (!e.target.classList.contains('tab-close')) {
                    this.showRenameDialog(parseInt(tabEl.dataset.tabId));
                }
            });
        });

        // Close button listeners
        container.querySelectorAll('.tab-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const tabId = parseInt(btn.dataset.tabId);
                this.closeTab(tabId);
            });
        });
    },

    getTabIcon(tab) {
        if (tab.model) {
            return tab.model.type === 'dl' ? '🧠' : '🔬';
        }
        if (tab.dataset) {
            return '📊';
        }
        return '📄';
    },

    showRenameDialog(tabId) {
        const tab = TabState.tabs.find(t => t.id === tabId);
        if (!tab) return;

        const newName = prompt('Rename tab:', tab.name);
        if (newName && newName.trim()) {
            this.renameTab(tabId, newName.trim());
        }
    },

    getActiveTab() {
        return TabState.tabs.find(t => t.id === TabState.activeTabId);
    },

    setActiveTabDataset(dataset) {
        const tab = this.getActiveTab();
        if (tab) {
            tab.setDataset(dataset);
            this.renderTabs();
        }
    },

    setActiveTabModel(model) {
        const tab = this.getActiveTab();
        if (tab) {
            tab.setModel(model);
            this.renderTabs();
        }
    },

    setActiveTabMode(mode) {
        const tab = this.getActiveTab();
        if (tab) {
            tab.mode = mode;
        }
    },

    addCellToActiveTab(cellId) {
        const tab = this.getActiveTab();
        if (tab) {
            tab.addCell(cellId);
        }
    },

    // Persistence
    saveState() {
        const state = {
            tabs: TabState.tabs.map(tab => ({
                id: tab.id,
                name: tab.name,
                dataset: tab.dataset,
                model: tab.model,
                mode: tab.mode,
                cells: tab.cells
            })),
            activeTabId: TabState.activeTabId,
            tabCounter: TabState.tabCounter
        };
        localStorage.setItem('qmlide_tabs', JSON.stringify(state));
    },

    loadState() {
        try {
            const saved = localStorage.getItem('qmlide_tabs');
            if (saved) {
                const state = JSON.parse(saved);
                // Restore tabs...
                console.log('Tab state restored');
            }
        } catch (e) {
            console.error('Failed to load tab state:', e);
        }
    }
};

// Export for global access
window.TabManager = TabManager;
window.TabState = TabState;
window.Tab = Tab;
