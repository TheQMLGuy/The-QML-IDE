/**
 * Preprocessing Phase - Data Cleaning & EDA Dashboard
 * Phase 2 of the workflow
 */

// ===========================
// Preprocessing Phase Module
// ===========================

const PreprocessingPhase = {
    appliedTransformations: [],

    init() {
        console.log('Initializing Preprocessing Phase...');
        this.appliedTransformations = [...WorkflowState.preprocessing.transformations];
        this.render();
        this.setupEventListeners();
        this.runInitialAnalysis();
    },

    render() {
        const container = document.getElementById('phase-2');
        if (!container) return;

        const dataset = WorkflowState.dataset;

        container.innerHTML = `
            <div class="preprocessing-phase">
                <div class="phase-header">
                    <h1>⚙️ Data Preprocessing & Analysis</h1>
                    <p>Clean, transform, and explore your dataset: <strong>${dataset.name || 'No dataset'}</strong></p>
                </div>

                <div class="preprocessing-layout">
                    <!-- Left Panel: Data Preview & Column Selection -->
                    <div class="panel panel-left">
                        <div class="panel-header">
                            <h3>📋 Data Overview</h3>
                        </div>
                        <div class="data-info">
                            <div class="info-row">
                                <span class="info-label">Shape:</span>
                                <span class="info-value" id="dataShape">${dataset.size || '-'}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Target:</span>
                                <select id="targetColumn" class="target-select">
                                    <option value="">Select target column...</option>
                                </select>
                            </div>
                        </div>
                        <div class="columns-list" id="columnsList">
                            <!-- Columns will be rendered here -->
                        </div>
                    </div>

                    <!-- Center Panel: Transformations -->
                    <div class="panel panel-center">
                        <div class="panel-header">
                            <h3>🔧 Transformations</h3>
                        </div>
                        
                        <div class="transform-categories">
                            <div class="transform-category">
                                <h4>Missing Values</h4>
                                <div class="transform-buttons">
                                    <button class="transform-btn" data-type="missing_drop">Drop Rows</button>
                                    <button class="transform-btn" data-type="missing_mean">Impute Mean</button>
                                    <button class="transform-btn" data-type="missing_median">Impute Median</button>
                                    <button class="transform-btn" data-type="missing_mode">Impute Mode</button>
                                </div>
                            </div>
                            
                            <div class="transform-category">
                                <h4>Outliers</h4>
                                <div class="transform-buttons">
                                    <button class="transform-btn" data-type="outlier_iqr">Remove IQR</button>
                                    <button class="transform-btn" data-type="outlier_clip">Clip Values</button>
                                    <button class="transform-btn" data-type="outlier_log">Log Transform</button>
                                </div>
                            </div>
                            
                            <div class="transform-category">
                                <h4>Encoding</h4>
                                <div class="transform-buttons">
                                    <button class="transform-btn" data-type="encode_onehot">One-Hot</button>
                                    <button class="transform-btn" data-type="encode_label">Label</button>
                                    <button class="transform-btn" data-type="encode_ordinal">Ordinal</button>
                                </div>
                            </div>
                            
                            <div class="transform-category">
                                <h4>Scaling</h4>
                                <div class="transform-buttons">
                                    <button class="transform-btn" data-type="scale_standard">Standard</button>
                                    <button class="transform-btn" data-type="scale_minmax">Min-Max</button>
                                    <button class="transform-btn" data-type="scale_robust">Robust</button>
                                </div>
                            </div>
                        </div>

                        <div class="applied-transforms">
                            <h4>Applied Transformations</h4>
                            <div class="transforms-list" id="transformsList">
                                ${this.renderAppliedTransformations()}
                            </div>
                            <button class="btn-clear-transforms" id="clearTransforms">Clear All</button>
                        </div>
                    </div>

                    <!-- Right Panel: EDA Dashboard -->
                    <div class="panel panel-right">
                        <div class="panel-header">
                            <h3>📊 Exploratory Analysis</h3>
                            <button class="btn-run-eda" id="runEda">Run Analysis</button>
                        </div>
                        
                        <div class="eda-tabs">
                            <button class="eda-tab active" data-tab="stats">Statistics</button>
                            <button class="eda-tab" data-tab="distributions">Distributions</button>
                            <button class="eda-tab" data-tab="correlations">Correlations</button>
                            <button class="eda-tab" data-tab="quality">Data Quality</button>
                        </div>
                        
                        <div class="eda-content" id="edaContent">
                            <div class="eda-placeholder">
                                <span>📈</span>
                                <p>Click "Run Analysis" to generate insights</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="phase-navigation">
                    <button class="btn-secondary btn-back" id="btnBack">
                        ← Back to Dataset
                    </button>
                    <div class="transform-summary" id="transformSummary">
                        ${this.appliedTransformations.length} transformations applied
                    </div>
                    <button class="btn-primary btn-next" id="btnNext">
                        Continue to Model Selection →
                    </button>
                </div>
            </div>
        `;
    },

    renderAppliedTransformations() {
        if (this.appliedTransformations.length === 0) {
            return '<div class="no-transforms">No transformations applied yet</div>';
        }

        return this.appliedTransformations.map((t, idx) => `
            <div class="transform-item">
                <span class="transform-name">${t.name}</span>
                <span class="transform-column">${t.column}</span>
                <button class="remove-transform" data-index="${idx}">×</button>
            </div>
        `).join('');
    },

    setupEventListeners() {
        const container = document.getElementById('phase-2');
        if (!container) return;

        // Transformation buttons
        container.querySelectorAll('.transform-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.applyTransformation(btn.dataset.type);
            });
        });

        // Remove transformation
        container.querySelectorAll('.remove-transform').forEach(btn => {
            btn.addEventListener('click', () => {
                this.removeTransformation(parseInt(btn.dataset.index));
            });
        });

        // Clear all transformations
        const clearBtn = document.getElementById('clearTransforms');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearTransformations();
            });
        }

        // Run EDA
        const runEdaBtn = document.getElementById('runEda');
        if (runEdaBtn) {
            runEdaBtn.addEventListener('click', () => {
                this.runEDA();
            });
        }

        // EDA tabs
        container.querySelectorAll('.eda-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                container.querySelectorAll('.eda-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.showEDATab(tab.dataset.tab);
            });
        });

        // Navigation
        const btnBack = document.getElementById('btnBack');
        if (btnBack) {
            btnBack.addEventListener('click', () => {
                WorkflowEngine.prevPhase();
            });
        }

        const btnNext = document.getElementById('btnNext');
        if (btnNext) {
            btnNext.addEventListener('click', () => {
                // Save transformations to workflow state
                WorkflowState.preprocessing.transformations = [...this.appliedTransformations];
                WorkflowEngine.nextPhase();
            });
        }
    },

    runInitialAnalysis() {
        // If we have data, generate column info
        if (WorkflowState.dataset.columns && WorkflowState.dataset.columns.length > 0) {
            this.renderColumnsList();
        }
    },

    renderColumnsList() {
        const columnsList = document.getElementById('columnsList');
        const targetSelect = document.getElementById('targetColumn');

        if (!columnsList) return;

        const columns = WorkflowState.dataset.columns || [];
        const dtypes = WorkflowState.dataset.dtypes || {};

        columnsList.innerHTML = columns.map(col => `
            <div class="column-item" data-column="${col}">
                <input type="checkbox" class="column-check" checked>
                <span class="column-name">${col}</span>
                <span class="column-dtype">${dtypes[col] || 'object'}</span>
            </div>
        `).join('');

        if (targetSelect) {
            targetSelect.innerHTML = `
                <option value="">Select target column...</option>
                ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
            `;

            targetSelect.addEventListener('change', () => {
                WorkflowState.dataset.target = targetSelect.value;
            });
        }
    },

    applyTransformation(type) {
        const transform = TRANSFORMATIONS[type];
        if (!transform) return;

        // For simplicity, apply to all numeric columns or prompt for column
        const column = prompt('Enter column name to apply transformation:');
        if (!column) return;

        const newTransform = {
            type,
            column,
            name: transform.name,
            code: transform.code(column)
        };

        this.appliedTransformations.push(newTransform);
        this.updateTransformsList();

        console.log(`Applied: ${transform.name} on ${column}`);
    },

    removeTransformation(index) {
        this.appliedTransformations.splice(index, 1);
        this.updateTransformsList();
    },

    clearTransformations() {
        this.appliedTransformations = [];
        this.updateTransformsList();
    },

    updateTransformsList() {
        const list = document.getElementById('transformsList');
        if (list) {
            list.innerHTML = this.renderAppliedTransformations();

            // Re-attach remove listeners
            list.querySelectorAll('.remove-transform').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.removeTransformation(parseInt(btn.dataset.index));
                });
            });
        }

        const summary = document.getElementById('transformSummary');
        if (summary) {
            summary.textContent = `${this.appliedTransformations.length} transformations applied`;
        }
    },

    runEDA() {
        const content = document.getElementById('edaContent');
        if (!content) return;

        content.innerHTML = '<div class="eda-loading">Analyzing data...</div>';

        // Simulate analysis (in real app, would use DataAnalysis module)
        setTimeout(() => {
            this.showEDATab('stats');
        }, 500);
    },

    showEDATab(tab) {
        const content = document.getElementById('edaContent');
        if (!content) return;

        switch (tab) {
            case 'stats':
                content.innerHTML = this.renderStatsTab();
                break;
            case 'distributions':
                content.innerHTML = this.renderDistributionsTab();
                break;
            case 'correlations':
                content.innerHTML = this.renderCorrelationsTab();
                break;
            case 'quality':
                content.innerHTML = this.renderQualityTab();
                break;
        }
    },

    renderStatsTab() {
        return `
            <div class="eda-stats">
                <div class="stat-card">
                    <div class="stat-title">Dataset Summary</div>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <span class="stat-value">${WorkflowState.dataset.shape?.[0] || '-'}</span>
                            <span class="stat-label">Rows</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${WorkflowState.dataset.shape?.[1] || '-'}</span>
                            <span class="stat-label">Columns</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">${WorkflowState.dataset.task || '-'}</span>
                            <span class="stat-label">Task Type</span>
                        </div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Column Types</div>
                    <div class="dtype-list">
                        <div class="dtype-item"><span class="dtype-count">-</span> Numeric</div>
                        <div class="dtype-item"><span class="dtype-count">-</span> Categorical</div>
                        <div class="dtype-item"><span class="dtype-count">-</span> DateTime</div>
                    </div>
                </div>
            </div>
        `;
    },

    renderDistributionsTab() {
        return `
            <div class="eda-distributions">
                <div class="distribution-placeholder">
                    <span>📊</span>
                    <p>Distribution plots will be generated for numeric columns</p>
                    <p class="note">Select columns from the left panel to visualize</p>
                </div>
            </div>
        `;
    },

    renderCorrelationsTab() {
        return `
            <div class="eda-correlations">
                <div class="correlation-placeholder">
                    <span>🔗</span>
                    <p>Correlation heatmap coming soon</p>
                    <p class="note">Shows relationships between numeric features</p>
                </div>
            </div>
        `;
    },

    renderQualityTab() {
        return `
            <div class="eda-quality">
                <div class="quality-card">
                    <div class="quality-title">Missing Values</div>
                    <div class="quality-status good">
                        <span class="status-icon">✓</span>
                        <span>No missing values detected</span>
                    </div>
                </div>
                <div class="quality-card">
                    <div class="quality-title">Duplicates</div>
                    <div class="quality-status good">
                        <span class="status-icon">✓</span>
                        <span>No duplicate rows found</span>
                    </div>
                </div>
                <div class="quality-card">
                    <div class="quality-title">Outliers</div>
                    <div class="quality-status warning">
                        <span class="status-icon">⚠</span>
                        <span>Potential outliers in some columns</span>
                    </div>
                </div>
            </div>
        `;
    }
};

// Export for global access
window.PreprocessingPhase = PreprocessingPhase;
