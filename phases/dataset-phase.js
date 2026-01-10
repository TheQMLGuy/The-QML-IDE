/**
 * Dataset Phase - Full-screen dataset selection
 * Phase 1 of the workflow
 */

// ===========================
// Dataset Sources Configuration
// ===========================

const DATASET_SOURCES = {
    sklearn: {
        name: 'Scikit-learn',
        icon: '🔬',
        description: 'Built-in datasets for ML practice',
        datasets: [
            {
                id: 'iris', name: 'Iris', size: '150 × 4', task: 'Classification', description: 'Flower species classification',
                columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
            },
            {
                id: 'wine', name: 'Wine', size: '178 × 13', task: 'Classification', description: 'Wine quality prediction',
                columns: ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280_od315', 'proline', 'target']
            },
            {
                id: 'breast_cancer', name: 'Breast Cancer', size: '569 × 30', task: 'Classification', description: 'Tumor diagnosis',
                columns: ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'target']
            },
            {
                id: 'digits', name: 'Digits', size: '1797 × 64', task: 'Classification', description: 'Handwritten digit recognition',
                columns: ['pixel_0', 'pixel_1', 'pixel_2', '...', 'pixel_63', 'target']
            },
            {
                id: 'diabetes', name: 'Diabetes', size: '442 × 10', task: 'Regression', description: 'Disease progression prediction',
                columns: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target']
            },
            {
                id: 'california_housing', name: 'California Housing', size: '20640 × 8', task: 'Regression', description: 'Housing price prediction',
                columns: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'target']
            }
        ]
    },
    kaggle: {
        name: 'Kaggle',
        icon: '🏆',
        description: 'Popular datasets from Kaggle competitions',
        datasets: [
            { id: 'titanic', name: 'Titanic', size: '891 × 12', task: 'Classification', description: 'Survival prediction', url: 'kaggle.com/c/titanic' },
            { id: 'house_prices', name: 'House Prices', size: '1460 × 81', task: 'Regression', description: 'Advanced regression techniques', url: 'kaggle.com/c/house-prices' },
            { id: 'mnist', name: 'MNIST', size: '60000 × 784', task: 'Classification', description: 'Handwritten digit recognition', url: 'kaggle.com/c/digit-recognizer' },
            { id: 'credit_card', name: 'Credit Card Fraud', size: '284807 × 31', task: 'Classification', description: 'Fraud detection', url: 'kaggle.com/mlg-ulb/creditcardfraud' }
        ]
    },
    uci: {
        name: 'UCI ML Repository',
        icon: '🎓',
        description: 'Classic datasets for research',
        datasets: [
            { id: 'adult', name: 'Adult Income', size: '48842 × 14', task: 'Classification', description: 'Income prediction' },
            { id: 'heart', name: 'Heart Disease', size: '303 × 14', task: 'Classification', description: 'Heart disease diagnosis' },
            { id: 'bank', name: 'Bank Marketing', size: '45211 × 17', task: 'Classification', description: 'Marketing campaign prediction' }
        ]
    },
    upload: {
        name: 'Upload',
        icon: '📁',
        description: 'Upload your own dataset',
        datasets: []
    }
};

// ===========================
// Dataset Phase Module
// ===========================

const DatasetPhase = {
    currentSource: 'sklearn',
    selectedDataset: null,

    init() {
        console.log('Initializing Dataset Phase...');
        this.render();
        this.setupEventListeners();
    },

    render() {
        const container = document.getElementById('phase-1');
        if (!container) return;

        container.innerHTML = `
            <div class="dataset-phase">
                <div class="phase-header">
                    <h1>📊 Select Your Dataset</h1>
                    <p>Choose a dataset to begin your machine learning workflow</p>
                </div>

                <div class="dataset-sources">
                    ${Object.entries(DATASET_SOURCES).map(([key, source]) => `
                        <button class="source-btn ${key === this.currentSource ? 'active' : ''}" data-source="${key}">
                            <span class="source-icon">${source.icon}</span>
                            <span class="source-name">${source.name}</span>
                        </button>
                    `).join('')}
                </div>

                <div class="source-description">
                    ${DATASET_SOURCES[this.currentSource].description}
                </div>

                ${this.currentSource === 'upload' ? this.renderUploadArea() : this.renderDatasetGrid()}

                <div class="dataset-preview hidden" id="datasetPreview">
                    <div class="preview-header">
                        <h3 id="previewTitle">Dataset Preview</h3>
                        <button class="close-preview" id="closePreview">×</button>
                    </div>
                    <div class="preview-content" id="previewContent"></div>
                </div>

                <div class="phase-navigation">
                    <div class="selected-info" id="selectedInfo">
                        No dataset selected
                    </div>
                    <button class="btn-primary btn-next" id="btnNext" disabled>
                        Continue to Preprocessing →
                    </button>
                </div>
            </div>
        `;
    },

    renderDatasetGrid() {
        const datasets = DATASET_SOURCES[this.currentSource].datasets;

        return `
            <div class="dataset-grid">
                ${datasets.map(ds => `
                    <div class="dataset-card ${this.selectedDataset?.id === ds.id ? 'selected' : ''}" 
                         data-id="${ds.id}">
                        <div class="card-header">
                            <span class="dataset-name">${ds.name}</span>
                            <span class="dataset-task task-${ds.task.toLowerCase()}">${ds.task}</span>
                        </div>
                        <div class="card-body">
                            <p class="dataset-desc">${ds.description}</p>
                            <div class="dataset-meta">
                                <span class="meta-size">📐 ${ds.size}</span>
                            </div>
                        </div>
                        <div class="card-actions">
                            <button class="btn-preview" data-id="${ds.id}">Preview</button>
                            <button class="btn-select" data-id="${ds.id}">Select</button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    renderUploadArea() {
        return `
            <div class="upload-area" id="uploadArea">
                <div class="upload-content">
                    <span class="upload-icon">📤</span>
                    <h3>Drag & Drop your file here</h3>
                    <p>or click to browse</p>
                    <p class="upload-formats">Supported formats: CSV, JSON, Excel</p>
                    <input type="file" id="fileInput" accept=".csv,.json,.xlsx,.xls" hidden>
                </div>
            </div>
        `;
    },

    setupEventListeners() {
        const container = document.getElementById('phase-1');
        if (!container) return;

        // Source buttons
        container.querySelectorAll('.source-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.currentSource = btn.dataset.source;
                this.selectedDataset = null;
                this.render();
                this.setupEventListeners();
            });
        });

        // Dataset cards - select
        container.querySelectorAll('.btn-select').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectDataset(btn.dataset.id);
            });
        });

        // Dataset cards - preview
        container.querySelectorAll('.btn-preview').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.previewDataset(btn.dataset.id);
            });
        });

        // Card click to select
        container.querySelectorAll('.dataset-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectDataset(card.dataset.id);
            });
        });

        // Close preview
        const closePreview = document.getElementById('closePreview');
        if (closePreview) {
            closePreview.addEventListener('click', () => {
                document.getElementById('datasetPreview')?.classList.add('hidden');
            });
        }

        // Next button
        const btnNext = document.getElementById('btnNext');
        if (btnNext) {
            btnNext.addEventListener('click', () => {
                if (this.selectedDataset) {
                    WorkflowEngine.nextPhase();
                }
            });
        }

        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) this.handleFileUpload(file);
            });
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) this.handleFileUpload(file);
            });
        }
    },

    selectDataset(datasetId) {
        const datasets = DATASET_SOURCES[this.currentSource].datasets;
        const dataset = datasets.find(d => d.id === datasetId);

        if (!dataset) return;

        this.selectedDataset = dataset;

        // Update UI
        document.querySelectorAll('.dataset-card').forEach(card => {
            card.classList.toggle('selected', card.dataset.id === datasetId);
        });

        const selectedInfo = document.getElementById('selectedInfo');
        if (selectedInfo) {
            selectedInfo.innerHTML = `
                <span class="selected-badge">✓</span>
                Selected: <strong>${dataset.name}</strong> (${dataset.size})
            `;
        }

        const btnNext = document.getElementById('btnNext');
        if (btnNext) {
            btnNext.disabled = false;
        }

        // Update workflow state - include columns for preprocessing phase
        const sizeParts = dataset.size.split('×').map(s => parseInt(s.trim()));
        WorkflowEngine.setDataset({
            source: this.currentSource,
            id: datasetId,
            name: dataset.name,
            task: dataset.task,
            size: dataset.size,
            columns: dataset.columns || [],
            shape: [sizeParts[0] || 0, sizeParts[1] || 0],
            target: 'target'  // Default target column name for sklearn datasets
        });

        console.log(`Selected dataset: ${dataset.name} with ${dataset.columns?.length || 0} columns`);
    },

    previewDataset(datasetId) {
        const datasets = DATASET_SOURCES[this.currentSource].datasets;
        const dataset = datasets.find(d => d.id === datasetId);

        if (!dataset) return;

        const preview = document.getElementById('datasetPreview');
        const title = document.getElementById('previewTitle');
        const content = document.getElementById('previewContent');

        if (!preview || !title || !content) return;

        title.textContent = dataset.name;
        content.innerHTML = `
            <div class="preview-stats">
                <div class="stat">
                    <div class="stat-value">${dataset.size.split('×')[0].trim()}</div>
                    <div class="stat-label">Rows</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${dataset.size.split('×')[1]?.trim() || '-'}</div>
                    <div class="stat-label">Features</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${dataset.task}</div>
                    <div class="stat-label">Task Type</div>
                </div>
            </div>
            <div class="preview-description">
                <h4>About this dataset</h4>
                <p>${dataset.description}</p>
                ${dataset.url ? `<a href="https://${dataset.url}" target="_blank">View on source →</a>` : ''}
            </div>
            <div class="preview-sample">
                <h4>Sample Data</h4>
                <p class="preview-note">Sample data will be loaded after selection</p>
            </div>
            <div class="preview-actions">
                <button class="btn-select-preview" data-id="${dataset.id}">Select This Dataset</button>
            </div>
        `;

        preview.classList.remove('hidden');

        // Add select button handler
        content.querySelector('.btn-select-preview')?.addEventListener('click', () => {
            this.selectDataset(datasetId);
            preview.classList.add('hidden');
        });
    },

    handleFileUpload(file) {
        console.log('Uploading file:', file.name);

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                let data;
                if (file.name.endsWith('.csv')) {
                    data = this.parseCSV(e.target.result);
                } else if (file.name.endsWith('.json')) {
                    data = JSON.parse(e.target.result);
                }

                if (data) {
                    const columns = Object.keys(data[0] || {});
                    this.selectedDataset = {
                        id: 'upload',
                        name: file.name,
                        size: `${data.length} × ${columns.length}`,
                        task: 'Unknown'
                    };

                    WorkflowEngine.setDataset({
                        source: 'upload',
                        id: 'upload',
                        name: file.name,
                        data: data,
                        columns: columns,
                        shape: [data.length, columns.length]
                    });

                    const selectedInfo = document.getElementById('selectedInfo');
                    if (selectedInfo) {
                        selectedInfo.innerHTML = `
                            <span class="selected-badge">✓</span>
                            Uploaded: <strong>${file.name}</strong> (${data.length} rows × ${columns.length} columns)
                        `;
                    }

                    const btnNext = document.getElementById('btnNext');
                    if (btnNext) btnNext.disabled = false;
                }
            } catch (err) {
                console.error('Failed to parse file:', err);
                alert('Failed to parse file. Please check the format.');
            }
        };
        reader.readAsText(file);
    },

    parseCSV(text) {
        const lines = text.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());

        return lines.slice(1).map(line => {
            const values = line.split(',');
            const row = {};
            headers.forEach((header, i) => {
                row[header] = values[i]?.trim();
            });
            return row;
        });
    }
};

// Export for global access
window.DatasetPhase = DatasetPhase;
window.DATASET_SOURCES = DATASET_SOURCES;
