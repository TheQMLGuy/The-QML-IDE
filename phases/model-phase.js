/**
 * Model Phase - Model Selection & Configuration
 * Phase 3 of the workflow
 */

// ===========================
// Model Phase Module
// ===========================

const ModelPhase = {
    selectedType: 'classification',
    selectedAlgorithm: null,
    hyperparams: {},

    init() {
        console.log('Initializing Model Phase...');

        // Restore state if exists
        if (WorkflowState.model.type) {
            this.selectedType = WorkflowState.model.type;
            this.selectedAlgorithm = WorkflowState.model.algorithm;
            this.hyperparams = { ...WorkflowState.model.hyperparams };
        }

        this.render();
        this.setupEventListeners();
    },

    render() {
        const container = document.getElementById('phase-3');
        if (!container) return;

        const dataset = WorkflowState.dataset;
        const taskType = dataset.task?.toLowerCase() || 'classification';

        container.innerHTML = `
            <div class="model-phase">
                <div class="phase-header">
                    <h1>🤖 Select Your Model</h1>
                    <p>Choose a model for <strong>${dataset.name || 'your dataset'}</strong></p>
                </div>

                <div class="model-type-tabs">
                    <button class="type-tab ${this.selectedType === 'classification' ? 'active' : ''}" data-type="classification">
                        🎯 Classification
                    </button>
                    <button class="type-tab ${this.selectedType === 'regression' ? 'active' : ''}" data-type="regression">
                        📈 Regression
                    </button>
                    <button class="type-tab ${this.selectedType === 'clustering' ? 'active' : ''}" data-type="clustering">
                        🔮 Clustering
                    </button>
                    <button class="type-tab ${this.selectedType === 'deep_learning' ? 'active' : ''}" data-type="deep_learning">
                        🧠 Deep Learning
                    </button>
                </div>

                <div class="model-grid" id="modelGrid">
                    ${this.renderModelCards()}
                </div>

                <div class="model-config ${this.selectedAlgorithm ? '' : 'hidden'}" id="modelConfig">
                    <h3>⚙️ Model Configuration</h3>
                    <div class="config-content" id="configContent">
                        ${this.renderConfigPanel()}
                    </div>
                </div>

                <div class="phase-navigation">
                    <button class="btn-secondary btn-back" id="btnBack">
                        ← Back to Preprocessing
                    </button>
                    <div class="model-summary" id="modelSummary">
                        ${this.selectedAlgorithm
                ? `Selected: ${MODELS[this.selectedType]?.[this.selectedAlgorithm]?.name || 'Unknown'}`
                : 'No model selected'}
                    </div>
                    <button class="btn-primary btn-next" id="btnNext" ${!this.selectedAlgorithm ? 'disabled' : ''}>
                        Generate Code & Open IDE →
                    </button>
                </div>
            </div>
        `;
    },

    renderModelCards() {
        const models = MODELS[this.selectedType] || {};

        return Object.entries(models).map(([key, model]) => `
            <div class="model-card ${this.selectedAlgorithm === key ? 'selected' : ''}" data-algorithm="${key}">
                <div class="model-icon">${this.getModelIcon(key)}</div>
                <div class="model-name">${model.name}</div>
                <div class="model-desc">${this.getModelDescription(key)}</div>
                <div class="model-params">
                    ${Object.keys(model.params).slice(0, 2).map(p => `<span class="param-tag">${p}</span>`).join('')}
                </div>
            </div>
        `).join('');
    },

    getModelIcon(algorithm) {
        const icons = {
            logistic: '📊',
            svm: '⚡',
            random_forest: '🌲',
            knn: '🎯',
            decision_tree: '🌳',
            xgboost: '🚀',
            linear: '📏',
            ridge: '📐',
            lasso: '✂️',
            svr: '⚡',
            kmeans: '🔵',
            dbscan: '🔍',
            hierarchical: '🏔️',
            mlp: '🧠',
            mlp_regressor: '🧠'
        };
        return icons[algorithm] || '🤖';
    },

    getModelDescription(algorithm) {
        const descriptions = {
            logistic: 'Linear model for binary/multiclass classification',
            svm: 'Maximum margin classifier with kernel trick',
            random_forest: 'Ensemble of decision trees for robust predictions',
            knn: 'Instance-based learning using nearest neighbors',
            decision_tree: 'Tree-based model with interpretable rules',
            xgboost: 'Gradient boosting for high performance',
            linear: 'Simple linear relationship modeling',
            ridge: 'Linear regression with L2 regularization',
            lasso: 'Linear regression with L1 regularization',
            svr: 'Support Vector Regression for non-linear relationships',
            kmeans: 'Partition data into K clusters',
            dbscan: 'Density-based clustering for arbitrary shapes',
            hierarchical: 'Nested cluster hierarchy',
            mlp: 'Multi-layer perceptron neural network',
            mlp_regressor: 'Neural network for regression tasks'
        };
        return descriptions[algorithm] || 'Machine learning model';
    },

    renderConfigPanel() {
        if (!this.selectedAlgorithm) {
            return '<p class="config-placeholder">Select a model to configure hyperparameters</p>';
        }

        const model = MODELS[this.selectedType]?.[this.selectedAlgorithm];
        if (!model) return '';

        const params = this.hyperparams;

        let html = `<div class="hyperparams">`;

        Object.entries(model.params).forEach(([key, defaultValue]) => {
            const currentValue = params[key] !== undefined ? params[key] : defaultValue;

            html += `
                <div class="hyperparam-group">
                    <label for="param-${key}">${this.formatParamName(key)}</label>
                    ${this.renderParamInput(key, currentValue, defaultValue)}
                </div>
            `;
        });

        html += `
            <div class="hyperparam-group">
                <label for="trainTestSplit">Train/Test Split</label>
                <div class="split-control">
                    <input type="range" id="trainTestSplit" min="10" max="50" value="${(WorkflowState.model.trainTestSplit || 0.2) * 100}">
                    <span class="split-value" id="splitValue">${((1 - (WorkflowState.model.trainTestSplit || 0.2)) * 100).toFixed(0)}% / ${((WorkflowState.model.trainTestSplit || 0.2) * 100).toFixed(0)}%</span>
                </div>
            </div>
        </div>`;

        return html;
    },

    formatParamName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    },

    renderParamInput(key, currentValue, defaultValue) {
        const type = typeof defaultValue;

        if (type === 'number') {
            if (Number.isInteger(defaultValue)) {
                return `<input type="number" id="param-${key}" value="${currentValue}" min="1" class="param-input">`;
            } else {
                return `<input type="number" id="param-${key}" value="${currentValue}" step="0.01" class="param-input">`;
            }
        } else if (type === 'boolean') {
            return `<input type="checkbox" id="param-${key}" ${currentValue ? 'checked' : ''} class="param-checkbox">`;
        } else if (key === 'kernel') {
            return `
                <select id="param-${key}" class="param-select">
                    <option value="linear" ${currentValue === 'linear' ? 'selected' : ''}>Linear</option>
                    <option value="rbf" ${currentValue === 'rbf' ? 'selected' : ''}>RBF</option>
                    <option value="poly" ${currentValue === 'poly' ? 'selected' : ''}>Polynomial</option>
                </select>
            `;
        } else if (defaultValue === null) {
            return `<input type="text" id="param-${key}" value="${currentValue || 'None'}" class="param-input">`;
        } else {
            return `<input type="text" id="param-${key}" value="${currentValue}" class="param-input">`;
        }
    },

    setupEventListeners() {
        const container = document.getElementById('phase-3');
        if (!container) return;

        // Type tabs
        container.querySelectorAll('.type-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                this.selectedType = tab.dataset.type;
                this.selectedAlgorithm = null;
                this.hyperparams = {};
                this.render();
                this.setupEventListeners();
            });
        });

        // Model cards
        container.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectModel(card.dataset.algorithm);
            });
        });

        // Hyperparameter inputs
        container.querySelectorAll('.param-input, .param-select, .param-checkbox').forEach(input => {
            input.addEventListener('change', () => {
                const key = input.id.replace('param-', '');
                let value = input.type === 'checkbox' ? input.checked : input.value;
                if (input.type === 'number') value = parseFloat(value);
                this.hyperparams[key] = value;
                this.updateWorkflowModel();
            });
        });

        // Train/test split
        const splitSlider = document.getElementById('trainTestSplit');
        if (splitSlider) {
            splitSlider.addEventListener('input', () => {
                const testRatio = parseInt(splitSlider.value) / 100;
                WorkflowEngine.setTrainTestSplit(testRatio);
                const splitValue = document.getElementById('splitValue');
                if (splitValue) {
                    splitValue.textContent = `${(100 - testRatio * 100).toFixed(0)}% / ${(testRatio * 100).toFixed(0)}%`;
                }
            });
        }

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
                if (this.selectedAlgorithm) {
                    this.updateWorkflowModel();
                    WorkflowEngine.nextPhase();
                }
            });
        }
    },

    selectModel(algorithm) {
        this.selectedAlgorithm = algorithm;
        const model = MODELS[this.selectedType]?.[algorithm];
        if (model) {
            this.hyperparams = { ...model.params };
        }

        // Update UI
        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.toggle('selected', card.dataset.algorithm === algorithm);
        });

        // Show config panel
        const configPanel = document.getElementById('modelConfig');
        const configContent = document.getElementById('configContent');
        if (configPanel) configPanel.classList.remove('hidden');
        if (configContent) configContent.innerHTML = this.renderConfigPanel();

        // Update summary and enable next button
        const summary = document.getElementById('modelSummary');
        if (summary) {
            summary.textContent = `Selected: ${model?.name || algorithm}`;
        }

        const btnNext = document.getElementById('btnNext');
        if (btnNext) btnNext.disabled = false;

        // Update workflow state
        this.updateWorkflowModel();

        // Re-attach param listeners
        this.setupEventListeners();
    },

    updateWorkflowModel() {
        WorkflowEngine.setModel(this.selectedType, this.selectedAlgorithm, this.hyperparams);
    }
};

// Export for global access
window.ModelPhase = ModelPhase;
