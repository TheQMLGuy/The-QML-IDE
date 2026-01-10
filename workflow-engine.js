/**
 * Workflow Engine - Central State Machine for Phase-Based IDE
 * Manages workflow state across all phases
 */

// ===========================
// Workflow State
// ===========================

const WorkflowState = {
    currentPhase: 1, // 1: Dataset, 2: Preprocessing, 3: Model, 4: IDE

    // Phase 1: Dataset
    dataset: {
        source: null,      // 'sklearn', 'kaggle', 'uci', 'huggingface', 'upload'
        id: null,
        name: null,
        data: null,        // Raw data array
        columns: [],
        dtypes: {},
        shape: [0, 0],
        target: null       // Target column name
    },

    // Phase 2: Preprocessing
    preprocessing: {
        transformations: [], // Array of applied transformations
        processedData: null,
        eda: null           // EDA results
    },

    // Phase 3: Model
    model: {
        type: null,         // 'classification', 'regression', 'clustering', 'deep_learning'
        algorithm: null,    // 'random_forest', 'svm', etc.
        hyperparams: {},
        trainTestSplit: 0.2
    },

    // Generated code
    generatedCode: {
        python: '',
        r: ''
    }
};

// ===========================
// Phase Definitions
// ===========================

const PHASES = {
    1: { name: 'Dataset Selection', icon: '📊', shortName: 'Dataset' },
    2: { name: 'Preprocessing & Analysis', icon: '⚙️', shortName: 'Preprocess' },
    3: { name: 'Model Selection', icon: '🤖', shortName: 'Model' },
    4: { name: 'IDE', icon: '💻', shortName: 'IDE' }
};

// ===========================
// Transformation Types
// ===========================

const TRANSFORMATIONS = {
    missing_drop: { name: 'Drop Missing', category: 'missing', code: (col) => `df.dropna(subset=['${col}'])` },
    missing_mean: { name: 'Impute Mean', category: 'missing', code: (col) => `df['${col}'].fillna(df['${col}'].mean())` },
    missing_median: { name: 'Impute Median', category: 'missing', code: (col) => `df['${col}'].fillna(df['${col}'].median())` },
    missing_mode: { name: 'Impute Mode', category: 'missing', code: (col) => `df['${col}'].fillna(df['${col}'].mode()[0])` },

    outlier_iqr: { name: 'Remove IQR Outliers', category: 'outliers', code: (col) => `# IQR outlier removal for ${col}` },
    outlier_clip: { name: 'Clip Outliers', category: 'outliers', code: (col) => `df['${col}'].clip(lower=q1, upper=q3)` },
    outlier_log: { name: 'Log Transform', category: 'outliers', code: (col) => `np.log1p(df['${col}'])` },

    encode_onehot: { name: 'One-Hot Encode', category: 'encoding', code: (col) => `pd.get_dummies(df, columns=['${col}'])` },
    encode_label: { name: 'Label Encode', category: 'encoding', code: (col) => `LabelEncoder().fit_transform(df['${col}'])` },
    encode_ordinal: { name: 'Ordinal Encode', category: 'encoding', code: (col) => `OrdinalEncoder().fit_transform(df[['${col}']])` },

    scale_standard: { name: 'Standard Scaler', category: 'scaling', code: (col) => `StandardScaler().fit_transform(df[['${col}']])` },
    scale_minmax: { name: 'Min-Max Scaler', category: 'scaling', code: (col) => `MinMaxScaler().fit_transform(df[['${col}']])` },
    scale_robust: { name: 'Robust Scaler', category: 'scaling', code: (col) => `RobustScaler().fit_transform(df[['${col}']])` }
};

// ===========================
// Model Definitions
// ===========================

const MODELS = {
    classification: {
        logistic: { name: 'Logistic Regression', import: 'LogisticRegression', params: { C: 1.0, max_iter: 100 } },
        svm: { name: 'SVM Classifier', import: 'SVC', params: { C: 1.0, kernel: 'rbf' } },
        random_forest: { name: 'Random Forest', import: 'RandomForestClassifier', params: { n_estimators: 100, max_depth: null } },
        knn: { name: 'K-Nearest Neighbors', import: 'KNeighborsClassifier', params: { n_neighbors: 5 } },
        decision_tree: { name: 'Decision Tree', import: 'DecisionTreeClassifier', params: { max_depth: null } },
        xgboost: { name: 'XGBoost', import: 'XGBClassifier', params: { n_estimators: 100, learning_rate: 0.1 } }
    },
    regression: {
        linear: { name: 'Linear Regression', import: 'LinearRegression', params: {} },
        ridge: { name: 'Ridge Regression', import: 'Ridge', params: { alpha: 1.0 } },
        lasso: { name: 'Lasso Regression', import: 'Lasso', params: { alpha: 1.0 } },
        svr: { name: 'SVR', import: 'SVR', params: { C: 1.0, kernel: 'rbf' } },
        random_forest: { name: 'Random Forest Regressor', import: 'RandomForestRegressor', params: { n_estimators: 100 } }
    },
    clustering: {
        kmeans: { name: 'K-Means', import: 'KMeans', params: { n_clusters: 3 } },
        dbscan: { name: 'DBSCAN', import: 'DBSCAN', params: { eps: 0.5, min_samples: 5 } },
        hierarchical: { name: 'Agglomerative', import: 'AgglomerativeClustering', params: { n_clusters: 3 } }
    },
    deep_learning: {
        mlp: { name: 'MLP Classifier', import: 'MLPClassifier', params: { hidden_layer_sizes: [100, 50], max_iter: 300 } },
        mlp_regressor: { name: 'MLP Regressor', import: 'MLPRegressor', params: { hidden_layer_sizes: [100, 50], max_iter: 300 } }
    }
};

// ===========================
// Workflow Engine
// ===========================

const WorkflowEngine = {
    init() {
        console.log('Initializing Workflow Engine...');
        this.renderPhaseIndicator();
        this.goToPhase(1);
        console.log('Workflow Engine initialized!');
    },

    // Phase Navigation
    goToPhase(phase) {
        if (phase < 1 || phase > 4) return false;

        // Validation before advancing
        if (phase > WorkflowState.currentPhase) {
            if (!this.validatePhase(WorkflowState.currentPhase)) {
                console.log(`Cannot advance: Phase ${WorkflowState.currentPhase} not complete`);
                return false;
            }
        }

        WorkflowState.currentPhase = phase;
        this.renderPhaseIndicator();
        this.showPhase(phase);

        console.log(`Navigated to Phase ${phase}: ${PHASES[phase].name}`);
        return true;
    },

    nextPhase() {
        return this.goToPhase(WorkflowState.currentPhase + 1);
    },

    prevPhase() {
        return this.goToPhase(WorkflowState.currentPhase - 1);
    },

    validatePhase(phase) {
        switch (phase) {
            case 1:
                // Check if dataset is selected (id is enough, data may load on-demand)
                return WorkflowState.dataset.id !== null;
            case 2:
                return true; // Preprocessing is optional
            case 3:
                return WorkflowState.model.algorithm !== null;
            default:
                return true;
        }
    },

    // Phase Rendering
    showPhase(phase) {
        // Hide all phase containers
        document.querySelectorAll('.phase-container').forEach(el => {
            el.classList.add('hidden');
        });

        // Show current phase
        const phaseEl = document.getElementById(`phase-${phase}`);
        if (phaseEl) {
            phaseEl.classList.remove('hidden');
        }

        // Trigger phase-specific initialization
        switch (phase) {
            case 1:
                if (typeof DatasetPhase !== 'undefined') DatasetPhase.init();
                break;
            case 2:
                if (typeof PreprocessingPhase !== 'undefined') PreprocessingPhase.init();
                break;
            case 3:
                if (typeof ModelPhase !== 'undefined') ModelPhase.init();
                break;
            case 4:
                this.generateCode();
                this.initializeIDE();
                break;
        }
    },

    // Initialize IDE with generated code
    initializeIDE() {
        console.log('Initializing IDE with generated code...');

        // Setup IDE event listeners if not done
        if (typeof setupModeToggle === 'function') setupModeToggle();
        if (typeof setupHeaderButtons === 'function') setupHeaderButtons();
        if (typeof setupConsoleToggle === 'function') setupConsoleToggle();
        if (typeof setupSnippetButtons === 'function') setupSnippetButtons();
        if (typeof setupLearningRateSlider === 'function') setupLearningRateSlider();

        // Create cell with generated code
        if (typeof addCell === 'function') {
            const cellId = addCell(WorkflowState.generatedCode.python);
            console.log(`Created cell ${cellId} with generated code`);
        }

        // Initialize tools
        if (typeof initializeMLTools === 'function') initializeMLTools();
        if (typeof initializePandasTools === 'function') initializePandasTools();

        // Log Pyodide status for debugging
        if (typeof AppState !== 'undefined') {
            if (AppState.pyodideReady) {
                console.log('Pyodide is ready - code can be executed');
                this.logToConsole('✅ Python runtime ready with NumPy, Pandas, and scikit-learn', 'success');
            } else {
                console.log('Pyodide still loading - please wait before running code');
                this.logToConsole('⏳ Python packages loading... Please wait before running code', 'info');
            }
        }
    },

    // Helper to log to console
    logToConsole(message, type = 'info') {
        const outputArea = document.getElementById('outputArea');
        if (outputArea) {
            const msgEl = document.createElement('div');
            msgEl.className = `output-message ${type}`;
            msgEl.textContent = message;
            outputArea.appendChild(msgEl);
        }
    },

    renderPhaseIndicator() {
        const indicator = document.getElementById('phaseIndicator');
        if (!indicator) return;

        let html = '<div class="phase-stepper">';
        for (let i = 1; i <= 4; i++) {
            const phase = PHASES[i];
            const status = i < WorkflowState.currentPhase ? 'completed' :
                i === WorkflowState.currentPhase ? 'active' : 'pending';
            html += `
                <div class="phase-step ${status}" data-phase="${i}">
                    <div class="step-icon">${phase.icon}</div>
                    <div class="step-name">${phase.shortName}</div>
                </div>
                ${i < 4 ? '<div class="step-connector"></div>' : ''}
            `;
        }
        html += '</div>';
        indicator.innerHTML = html;

        // Add click handlers
        indicator.querySelectorAll('.phase-step').forEach(step => {
            step.addEventListener('click', () => {
                const targetPhase = parseInt(step.dataset.phase);
                if (targetPhase <= WorkflowState.currentPhase ||
                    this.validatePhase(WorkflowState.currentPhase)) {
                    this.goToPhase(targetPhase);
                }
            });
        });
    },

    // Dataset Operations
    setDataset(dataset) {
        WorkflowState.dataset = { ...WorkflowState.dataset, ...dataset };
        WorkflowState.preprocessing.processedData = dataset.data ? [...dataset.data] : null;
        console.log('Dataset set:', dataset.name);
    },

    // Preprocessing Operations
    addTransformation(type, column, params = {}) {
        const transform = TRANSFORMATIONS[type];
        if (!transform) return;

        WorkflowState.preprocessing.transformations.push({
            type,
            column,
            params,
            name: transform.name,
            code: transform.code(column)
        });

        console.log(`Added transformation: ${transform.name} on ${column}`);
    },

    removeTransformation(index) {
        WorkflowState.preprocessing.transformations.splice(index, 1);
    },

    clearTransformations() {
        WorkflowState.preprocessing.transformations = [];
    },

    // Model Operations
    setModel(type, algorithm, hyperparams = {}) {
        const modelDef = MODELS[type]?.[algorithm];
        if (!modelDef) return;

        WorkflowState.model = {
            type,
            algorithm,
            hyperparams: { ...modelDef.params, ...hyperparams },
            trainTestSplit: WorkflowState.model.trainTestSplit
        };

        console.log(`Model set: ${modelDef.name}`);
    },

    setTrainTestSplit(ratio) {
        WorkflowState.model.trainTestSplit = ratio;
    },

    // Code Generation
    generateCode() {
        const { dataset, preprocessing, model } = WorkflowState;

        let code = `# ============================================\n`;
        code += `# Auto-generated by QML IDE Workflow\n`;
        code += `# Dataset: ${dataset.name || 'Unknown'}\n`;
        code += `# Model: ${model.algorithm || 'None'}\n`;
        code += `# ============================================\n`;
        code += `# NOTE: Wait for "Python Ready" status before running!\n`;
        code += `# Required packages: numpy, pandas, scikit-learn\n\n`;

        // Imports
        code += `# ===== IMPORTS =====\n`;
        code += `import numpy as np\n`;
        code += `import pandas as pd\n`;
        code += `from sklearn.model_selection import train_test_split\n`;

        if (preprocessing.transformations.length > 0) {
            code += `from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder\n`;
            code += `from sklearn.impute import SimpleImputer\n`;
        }

        // Add model-specific imports
        if (model.algorithm) {
            const modelDef = MODELS[model.type]?.[model.algorithm];
            if (modelDef) {
                if (model.type === 'clustering') {
                    code += `from sklearn.cluster import ${modelDef.import}\n`;
                } else if (model.algorithm === 'xgboost') {
                    code += `from xgboost import ${modelDef.import}\n`;
                } else if (model.type === 'deep_learning') {
                    code += `from sklearn.neural_network import ${modelDef.import}\n`;
                } else {
                    // Determine correct sklearn module
                    let module = 'linear_model';
                    if (model.algorithm.includes('forest') || model.algorithm === 'decision_tree') {
                        module = 'ensemble';
                    } else if (model.algorithm === 'svm' || model.algorithm === 'svr') {
                        module = 'svm';
                    } else if (model.algorithm === 'knn') {
                        module = 'neighbors';
                    }
                    code += `from sklearn.${module} import ${modelDef.import}\n`;
                }
                code += `from sklearn.metrics import accuracy_score, classification_report\n`;
            }
        }

        code += `\n`;

        // Dataset Loading
        code += `# ===== DATASET LOADING =====\n`;
        if (dataset.source === 'sklearn') {
            code += `from sklearn.datasets import load_${dataset.id}\n`;
            code += `data = load_${dataset.id}()\n`;
            code += `X, y = data.data, data.target\n`;
            code += `df = pd.DataFrame(X, columns=data.feature_names if hasattr(data, 'feature_names') else [f'feature_{i}' for i in range(X.shape[1])])\n`;
            code += `df['target'] = y\n`;
        } else {
            code += `# Load your dataset here\n`;
            code += `# df = pd.read_csv('your_data.csv')\n`;
        }
        code += `print(f"Dataset shape: {df.shape}")\n\n`;

        // Preprocessing
        if (preprocessing.transformations.length > 0) {
            code += `# ===== PREPROCESSING =====\n`;
            preprocessing.transformations.forEach(t => {
                code += `# ${t.name} on '${t.column}'\n`;
                code += `${t.code}\n\n`;
            });
        }

        // Train/Test Split
        code += `# ===== TRAIN/TEST SPLIT =====\n`;
        code += `X = df.drop('target', axis=1)\n`;
        code += `y = df['target']\n`;
        code += `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${model.trainTestSplit}, random_state=42)\n\n`;

        // Model Training
        if (model.algorithm) {
            const modelDef = MODELS[model.type]?.[model.algorithm];
            if (modelDef) {
                code += `# ===== MODEL TRAINING =====\n`;
                const paramsStr = Object.entries(model.hyperparams)
                    .map(([k, v]) => `${k}=${typeof v === 'string' ? `'${v}'` : v}`)
                    .join(', ');
                code += `model = ${modelDef.import}(${paramsStr})\n`;
                code += `model.fit(X_train, y_train)\n\n`;

                // Evaluation
                code += `# ===== EVALUATION =====\n`;
                code += `predictions = model.predict(X_test)\n`;
                if (model.type === 'classification') {
                    code += `print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")\n`;
                    code += `print("\\nClassification Report:")\n`;
                    code += `print(classification_report(y_test, predictions))\n`;
                } else if (model.type === 'regression') {
                    code += `from sklearn.metrics import mean_squared_error, r2_score\n`;
                    code += `print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")\n`;
                    code += `print(f"R2 Score: {r2_score(y_test, predictions):.4f}")\n`;
                }
            }
        }

        WorkflowState.generatedCode.python = code;
        console.log('Code generated successfully');
        return code;
    },

    // State Persistence
    saveState() {
        localStorage.setItem('qmlide_workflow', JSON.stringify(WorkflowState));
    },

    loadState() {
        try {
            const saved = localStorage.getItem('qmlide_workflow');
            if (saved) {
                Object.assign(WorkflowState, JSON.parse(saved));
                console.log('Workflow state restored');
            }
        } catch (e) {
            console.error('Failed to load workflow state:', e);
        }
    },

    resetWorkflow() {
        WorkflowState.currentPhase = 1;
        WorkflowState.dataset = { source: null, id: null, name: null, data: null, columns: [], dtypes: {}, shape: [0, 0], target: null };
        WorkflowState.preprocessing = { transformations: [], processedData: null, eda: null };
        WorkflowState.model = { type: null, algorithm: null, hyperparams: {}, trainTestSplit: 0.2 };
        WorkflowState.generatedCode = { python: '', r: '' };
        this.goToPhase(1);
    }
};

// Export for global access
window.WorkflowEngine = WorkflowEngine;
window.WorkflowState = WorkflowState;
window.PHASES = PHASES;
window.TRANSFORMATIONS = TRANSFORMATIONS;
window.MODELS = MODELS;
