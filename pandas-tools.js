/**
 * Pandas Tools - DataFrame Visualization Mode
 * Visualizes pandas DataFrames and provides common operations
 */

// ===========================
// Pandas State
// ===========================

const PandasState = {
    dataframeName: 'df',
    columns: [],
    shape: [0, 0],
    dtypes: {}
};

// Pandas code snippets
const PANDAS_SNIPPETS = {
    pandas_import: `import pandas as pd
import numpy as np

print("Pandas version:", pd.__version__)`,

    pandas_read: `import pandas as pd

# Read CSV file
df = pd.read_csv('data.csv')

# Display first few rows
print(df.head())
print(f"Shape: {df.shape}")`,

    pandas_filter: `# Filter rows based on condition
filtered_df = df[df['column'] > value]

# Multiple conditions
filtered_df = df[(df['col1'] > 10) & (df['col2'] == 'value')]

print(filtered_df)`,

    pandas_groupby: `# Group by and aggregate
grouped = df.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'other_col': 'max'
})

print(grouped)`,

    pandas_merge: `# Merge two DataFrames
merged_df = pd.merge(
    df1, 
    df2, 
    on='key_column',  # or left_on, right_on
    how='inner'       # 'left', 'right', 'outer'
)

print(merged_df)`,

    pandas_plot: `import matplotlib.pyplot as plt

# Simple plot
df['column'].plot(kind='hist')
plt.title('Distribution')
plt.show()

# Multiple columns
df[['col1', 'col2']].plot(kind='bar')
plt.show()`
};

// DataFrame operation snippets
const DF_OPERATIONS = {
    head: `print(df.head())`,
    tail: `print(df.tail())`,
    describe: `print(df.describe())`,
    info: `print(df.info())`,
    nunique: `print(df.nunique())`,
    isnull: `print(df.isnull().sum())`
};

// ===========================
// Initialization
// ===========================

function initializePandasTools() {
    console.log('Initializing Pandas Tools...');

    setupDfControls();
    setupDfOperations();
    setupPandasSnippets();
    loadSampleDataFrame();

    console.log('Pandas Tools initialized!');
}

// ===========================
// DataFrame Controls
// ===========================

function setupDfControls() {
    const loadSampleBtn = document.getElementById('loadSampleDf');
    const refreshBtn = document.getElementById('refreshDf');

    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', loadSampleDataFrame);
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshDataFrame);
    }
}

function loadSampleDataFrame() {
    // Sample data
    const sampleData = {
        columns: ['name', 'age', 'score', 'city'],
        data: [
            ['Alice', 25, 92.5, 'NYC'],
            ['Bob', 30, 88.0, 'LA'],
            ['Charlie', 35, 95.2, 'Chicago'],
            ['Diana', 28, 91.0, 'NYC'],
            ['Eve', 22, 87.5, 'LA']
        ],
        dtypes: {
            'name': 'object',
            'age': 'int64',
            'score': 'float64',
            'city': 'object'
        }
    };

    updateDataFrameDisplay(sampleData);

    // Generate code for creating this DataFrame
    if (AppState.syncEnabled && AppState.activeCellId && AppState.mode === 'pandas') {
        const code = `import pandas as pd

# Create sample DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 22],
    'score': [92.5, 88.0, 95.2, 91.0, 87.5],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
}

df = pd.DataFrame(data)
print(df)`;

        const cellData = AppState.cells.find(c => c.id === AppState.activeCellId);
        if (cellData) {
            cellData.editor.setValue(code);
        }
    }
}

function updateDataFrameDisplay(data) {
    const tableContainer = document.getElementById('dfTable');
    const shapeEl = document.getElementById('dfShape');
    const columnInfoEl = document.getElementById('columnInfo');

    if (!tableContainer) return;

    // Update table
    let tableHtml = '<table class="df-table"><thead><tr><th></th>';
    data.columns.forEach(col => {
        tableHtml += `<th>${col}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';

    data.data.forEach((row, idx) => {
        tableHtml += `<tr><td class="row-idx">${idx}</td>`;
        row.forEach(val => {
            tableHtml += `<td>${val}</td>`;
        });
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';

    tableContainer.innerHTML = tableHtml;

    // Update shape
    if (shapeEl) {
        shapeEl.textContent = `Shape: (${data.data.length}, ${data.columns.length})`;
    }

    // Update column info
    if (columnInfoEl) {
        let infoHtml = '';
        data.columns.forEach(col => {
            const dtype = data.dtypes[col] || 'object';
            infoHtml += `<div class="col-item">
                <span class="col-name">${col}</span>
                <span class="col-dtype">${dtype}</span>
            </div>`;
        });
        columnInfoEl.innerHTML = infoHtml;
    }

    // Update state
    PandasState.columns = data.columns;
    PandasState.shape = [data.data.length, data.columns.length];
    PandasState.dtypes = data.dtypes;
}

function refreshDataFrame() {
    // In a real implementation, this would parse the code to extract DataFrame
    // For now, just reload sample
    loadSampleDataFrame();
}

// ===========================
// DataFrame Operations
// ===========================

function setupDfOperations() {
    document.querySelectorAll('#pandasSidebar .op-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const op = btn.dataset.op;
            insertDfOperation(op);
        });
    });
}

function insertDfOperation(op) {
    const code = DF_OPERATIONS[op];
    if (!code) return;

    // Insert into active cell
    if (AppState.activeCellId) {
        const cellData = AppState.cells.find(c => c.id === AppState.activeCellId);
        if (cellData) {
            const currentCode = cellData.editor.getValue();
            cellData.editor.setValue(currentCode + '\n' + code);
        }
    }
}

// ===========================
// Snippet Buttons
// ===========================

function setupPandasSnippets() {
    if (typeof CODE_SNIPPETS !== 'undefined') {
        Object.assign(CODE_SNIPPETS, PANDAS_SNIPPETS);
    }
}

// Export for global access
window.initializePandasTools = initializePandasTools;
window.PANDAS_SNIPPETS = PANDAS_SNIPPETS;
