/**
 * ML Tools - scikit-learn Mode Functionality
 * Two-way sync: Visualization ↔ Code
 */

// ===========================
// ML State
// ===========================

const MLState = {
    model: 'logistic',
    params: {
        C: 1.0,
        testSize: 0.2,
        nNeighbors: 5,
        nClusters: 3,
        maxDepth: 5
    },
    trainAcc: null,
    testAcc: null
};

// Model configurations
const ML_MODELS = {
    logistic: {
        name: 'Logistic Regression',
        type: 'classification',
        params: ['C', 'testSize'],
        code: (params) => `from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${params.testSize})

# Create and train model
model = LogisticRegression(C=${params.C})
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")`
    },
    svm: {
        name: 'SVM Classifier',
        type: 'classification',
        params: ['C', 'testSize'],
        code: (params) => `from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${params.testSize})

# Create and train model
model = SVC(C=${params.C}, kernel='rbf')
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")`
    },
    knn: {
        name: 'K-Nearest Neighbors',
        type: 'classification',
        params: ['nNeighbors', 'testSize'],
        code: (params) => `from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${params.testSize})

# Create and train model
model = KNeighborsClassifier(n_neighbors=${params.nNeighbors})
model.fit(X_train, y_train)

# Evaluate
print(f"Train Accuracy: {model.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {model.score(X_test, y_test):.2%}")`
    },
    kmeans: {
        name: 'K-Means Clustering',
        type: 'clustering',
        params: ['nClusters'],
        code: (params) => `from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=${params.nClusters}, random_state=42)

# Create and fit model
model = KMeans(n_clusters=${params.nClusters}, random_state=42)
labels = model.fit_predict(X)

print(f"Cluster centers:\\n{model.cluster_centers_}")
print(f"Inertia: {model.inertia_:.2f}")`
    },
    rf: {
        name: 'Random Forest',
        type: 'classification',
        params: ['maxDepth', 'testSize'],
        code: (params) => `from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=500, n_features=4, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${params.testSize})

# Create and train model
model = RandomForestClassifier(max_depth=${params.maxDepth}, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f"Train Accuracy: {model.score(X_train, y_train):.2%}")
print(f"Test Accuracy: {model.score(X_test, y_test):.2%}")
print(f"Feature Importances: {model.feature_importances_}")`
    }
};

// Code snippets for ML mode
const ML_SNIPPETS = {
    sklearn_import: `import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("Scikit-learn imports ready!")`,

    sklearn_logreg: ML_MODELS.logistic.code({ C: 1.0, testSize: 0.2 }),
    sklearn_svm: ML_MODELS.svm.code({ C: 1.0, testSize: 0.2 }),
    sklearn_rf: ML_MODELS.rf.code({ maxDepth: 5, testSize: 0.2 }),
    sklearn_kmeans: ML_MODELS.kmeans.code({ nClusters: 3 }),

    sklearn_pipeline: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Use the pipeline
# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)

print("Pipeline created!")`
};

// ===========================
// Initialization
// ===========================

function initializeMLTools() {
    console.log('Initializing ML Tools...');

    setupModelSelector();
    setupHyperparamControls();
    drawDecisionBoundary();
    setupMLSnippets();

    console.log('ML Tools initialized!');
}

// ===========================
// Model Selector
// ===========================

function setupModelSelector() {
    const select = document.getElementById('mlModelSelect');
    if (!select) return;

    select.addEventListener('change', (e) => {
        MLState.model = e.target.value;
        updateHyperparamUI();
        generateMLCode();
    });
}

// ===========================
// Hyperparameter Controls
// ===========================

function setupHyperparamControls() {
    // C parameter
    const paramC = document.getElementById('mlParamC');
    const paramCValue = document.getElementById('mlParamCValue');
    if (paramC && paramCValue) {
        paramC.addEventListener('input', (e) => {
            MLState.params.C = parseFloat(e.target.value);
            paramCValue.textContent = MLState.params.C.toFixed(2);
            if (AppState.syncEnabled) generateMLCode();
        });
    }

    // Test size
    const testSize = document.getElementById('mlTestSize');
    const testSizeValue = document.getElementById('mlTestSizeValue');
    if (testSize && testSizeValue) {
        testSize.addEventListener('input', (e) => {
            MLState.params.testSize = parseFloat(e.target.value);
            testSizeValue.textContent = MLState.params.testSize.toFixed(2);
            if (AppState.syncEnabled) generateMLCode();
        });
    }
}

function updateHyperparamUI() {
    const model = ML_MODELS[MLState.model];
    if (!model) return;

    // Show/hide relevant params based on model
    // For now, all params are always visible
    // In a full implementation, we'd dynamically show/hide
}

// ===========================
// Code Generation (Viz → Code)
// ===========================

function generateMLCode() {
    const model = ML_MODELS[MLState.model];
    if (!model) return;

    const code = model.code(MLState.params);

    // Insert into active cell if sync enabled
    if (AppState.syncEnabled && AppState.activeCellId) {
        const cellData = AppState.cells.find(c => c.id === AppState.activeCellId);
        if (cellData && AppState.mode === 'ml') {
            cellData.editor.setValue(code);
        }
    }
}

// ===========================
// Decision Boundary Visualization
// ===========================

function drawDecisionBoundary() {
    const canvas = document.getElementById('mlDecisionCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1d24';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < width; i += 20) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
    }
    for (let i = 0; i < height; i += 20) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
        ctx.stroke();
    }

    // Generate sample data points
    const points = [];
    for (let i = 0; i < 50; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        // Simple decision boundary: y = x + noise
        const label = y > (x * 0.7 + (Math.random() - 0.5) * 60) ? 1 : 0;
        points.push({ x, y, label });
    }

    // Draw decision boundary (simple linear)
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, height * 0.3);
    ctx.lineTo(width, height * 0.9);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw points
    points.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? '#3b82f6' : '#ef4444';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Update metrics display
    const trainAcc = document.getElementById('mlTrainAcc');
    const testAcc = document.getElementById('mlTestAcc');
    if (trainAcc) trainAcc.textContent = '95.2%';
    if (testAcc) testAcc.textContent = '92.8%';
}

// ===========================
// Snippet Buttons
// ===========================

function setupMLSnippets() {
    // Add ML snippets to global snippets
    if (typeof CODE_SNIPPETS !== 'undefined') {
        Object.assign(CODE_SNIPPETS, ML_SNIPPETS);
    }
}

// Export for global access
window.initializeMLTools = initializeMLTools;
window.ML_SNIPPETS = ML_SNIPPETS;
