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
    data: [],
    shape: [0, 0],
    dtypes: {},
    // Full-screen state
    fullscreen: {
        currentPage: 1,
        pageSize: 50,
        sortColumn: null,
        sortDirection: 'asc',
        searchQuery: '',
        filteredData: []
    }
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
    setupFullscreenModal();

    // Show empty state on init - don't load sample data automatically
    showEmptyDataFrameState();

    console.log('Pandas Tools initialized!');
}

// ===========================
// Empty State
// ===========================

function showEmptyDataFrameState() {
    const tableContainer = document.getElementById('dfTable');
    const shapeEl = document.getElementById('dfShape');
    const columnInfoEl = document.getElementById('columnInfo');

    if (tableContainer) {
        tableContainer.innerHTML = `
            <div class="empty-state" style="text-align:center;padding:20px;color:var(--text-muted);">
                <div style="font-size:2rem;margin-bottom:8px;">📊</div>
                <div style="font-size:0.85rem;">No data loaded</div>
                <div style="font-size:0.75rem;margin-top:4px;">Load a dataset to view DataFrame</div>
            </div>
        `;
    }

    if (shapeEl) {
        shapeEl.textContent = 'Shape: -';
    }

    if (columnInfoEl) {
        columnInfoEl.innerHTML = `
            <div class="col-item" style="color:var(--text-muted);font-style:italic;">
                No columns
            </div>
        `;
    }

    // Clear state
    PandasState.columns = [];
    PandasState.data = [];
    PandasState.shape = [0, 0];
    PandasState.dtypes = {};
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
    // Sample data with more rows for demonstration
    const sampleData = {
        columns: ['name', 'age', 'score', 'city', 'department'],
        data: [
            ['Alice', 25, 92.5, 'NYC', 'Engineering'],
            ['Bob', 30, 88.0, 'LA', 'Marketing'],
            ['Charlie', 35, 95.2, 'Chicago', 'Engineering'],
            ['Diana', 28, 91.0, 'NYC', 'Design'],
            ['Eve', 22, 87.5, 'LA', 'Marketing'],
            ['Frank', 45, 78.5, 'Seattle', 'Sales'],
            ['Grace', 31, 94.0, 'NYC', 'Engineering'],
            ['Henry', 29, 82.3, 'Chicago', 'Sales'],
            ['Ivy', 26, 89.7, 'LA', 'Design'],
            ['Jack', 33, 91.5, 'Seattle', 'Engineering'],
            ['Karen', 27, 86.2, 'NYC', 'Marketing'],
            ['Leo', 38, 79.8, 'Chicago', 'Sales'],
            ['Mia', 24, 93.1, 'LA', 'Design'],
            ['Noah', 41, 85.4, 'Seattle', 'Engineering'],
            ['Olivia', 23, 90.9, 'NYC', 'Marketing']
        ],
        dtypes: {
            'name': 'object',
            'age': 'int64',
            'score': 'float64',
            'city': 'object',
            'department': 'object'
        }
    };

    updateDataFrameDisplay(sampleData);

    // Generate code for creating this DataFrame
    if (typeof AppState !== 'undefined' && AppState.syncEnabled && AppState.activeCellId && AppState.mode === 'pandas') {
        const code = `import pandas as pd

# Create sample DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack', 'Karen', 'Leo', 'Mia', 'Noah', 'Olivia'],
    'age': [25, 30, 35, 28, 22, 45, 31, 29, 26, 33, 27, 38, 24, 41, 23],
    'score': [92.5, 88.0, 95.2, 91.0, 87.5, 78.5, 94.0, 82.3, 89.7, 91.5, 86.2, 79.8, 93.1, 85.4, 90.9],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Seattle', 'NYC', 'Chicago', 'LA', 'Seattle', 'NYC', 'Chicago', 'LA', 'Seattle', 'NYC'],
    'department': ['Engineering', 'Marketing', 'Engineering', 'Design', 'Marketing', 'Sales', 'Engineering', 'Sales', 'Design', 'Engineering', 'Marketing', 'Sales', 'Design', 'Engineering', 'Marketing']
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

    // Show only first 5 rows in sidebar preview
    const previewRows = data.data.slice(0, 5);
    previewRows.forEach((row, idx) => {
        tableHtml += `<tr><td class="row-idx">${idx}</td>`;
        row.forEach(val => {
            tableHtml += `<td>${val}</td>`;
        });
        tableHtml += '</tr>';
    });

    if (data.data.length > 5) {
        tableHtml += `<tr><td colspan="${data.columns.length + 1}" style="text-align:center;color:var(--text-muted);font-size:0.75rem;">... ${data.data.length - 5} more rows</td></tr>`;
    }
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
    PandasState.data = data.data;
    PandasState.shape = [data.data.length, data.columns.length];
    PandasState.dtypes = data.dtypes;
    PandasState.fullscreen.filteredData = [...data.data];

    // Update active tab's dataset if TabManager exists
    if (typeof TabManager !== 'undefined') {
        TabManager.setActiveTabDataset({
            name: PandasState.dataframeName,
            columns: data.columns,
            data: data.data,
            dtypes: data.dtypes
        });
    }
}

function refreshDataFrame() {
    // In a real implementation, this would parse the code to extract DataFrame
    // For now, just reload sample
    loadSampleDataFrame();
}

// ===========================
// Full-Screen Modal
// ===========================

function setupFullscreenModal() {
    const modal = document.getElementById('fullscreenDfModal');
    if (!modal) return;

    // Close button
    const closeBtn = document.getElementById('closeFullscreenBtn');
    if (closeBtn) {
        closeBtn.addEventListener('click', closeFullscreenModal);
    }

    // Escape key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeFullscreenModal();
        }
    });

    // Click outside to close
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeFullscreenModal();
        }
    });

    // Search
    const searchInput = document.getElementById('dfSearchInput');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            PandasState.fullscreen.searchQuery = e.target.value.toLowerCase();
            PandasState.fullscreen.currentPage = 1;
            filterAndRenderFullscreen();
        });
    }

    // Page size
    const pageSizeSelect = document.getElementById('dfPageSize');
    if (pageSizeSelect) {
        pageSizeSelect.addEventListener('change', (e) => {
            PandasState.fullscreen.pageSize = e.target.value === 'all' ? Infinity : parseInt(e.target.value);
            PandasState.fullscreen.currentPage = 1;
            renderFullscreenTable();
        });
    }

    // Pagination
    const prevBtn = document.getElementById('dfPrevPage');
    const nextBtn = document.getElementById('dfNextPage');
    if (prevBtn) prevBtn.addEventListener('click', () => changePage(-1));
    if (nextBtn) nextBtn.addEventListener('click', () => changePage(1));

    // Sort button
    const sortBtn = document.getElementById('dfSortBtn');
    if (sortBtn) {
        sortBtn.addEventListener('click', showSortDialog);
    }

    // Filter button
    const filterBtn = document.getElementById('dfFilterBtn');
    if (filterBtn) {
        filterBtn.addEventListener('click', showFilterDialog);
    }

    // Stats button
    const statsBtn = document.getElementById('dfStatsBtn');
    if (statsBtn) {
        statsBtn.addEventListener('click', showStatsDialog);
    }

    // Export button
    const exportBtn = document.getElementById('dfExportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportDataFrame);
    }

    // Add fullscreen button to sidebar
    addFullscreenButton();
}

function addFullscreenButton() {
    const dfControls = document.querySelector('#pandasSidebar .df-controls');
    if (dfControls && !document.getElementById('openFullscreenBtn')) {
        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.id = 'openFullscreenBtn';
        fullscreenBtn.className = 'btn-small';
        fullscreenBtn.textContent = '⛶ Full Screen';
        fullscreenBtn.title = 'View DataFrame in full screen';
        fullscreenBtn.addEventListener('click', openFullscreenModal);
        dfControls.appendChild(fullscreenBtn);
    }
}

function openFullscreenModal() {
    const modal = document.getElementById('fullscreenDfModal');
    if (!modal) return;

    modal.classList.remove('hidden');
    PandasState.fullscreen.currentPage = 1;
    PandasState.fullscreen.filteredData = [...PandasState.data];
    PandasState.fullscreen.searchQuery = '';

    const searchInput = document.getElementById('dfSearchInput');
    if (searchInput) searchInput.value = '';

    renderFullscreenTable();
    updateFooter();
}

function closeFullscreenModal() {
    const modal = document.getElementById('fullscreenDfModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

function filterAndRenderFullscreen() {
    const query = PandasState.fullscreen.searchQuery;

    if (!query) {
        PandasState.fullscreen.filteredData = [...PandasState.data];
    } else {
        PandasState.fullscreen.filteredData = PandasState.data.filter(row =>
            row.some(cell => String(cell).toLowerCase().includes(query))
        );
    }

    renderFullscreenTable();
}

function renderFullscreenTable() {
    const container = document.getElementById('fullscreenDfTable');
    if (!container) return;

    const { filteredData, currentPage, pageSize, sortColumn, sortDirection } = PandasState.fullscreen;

    // Sort if needed
    let sortedData = [...filteredData];
    if (sortColumn !== null) {
        const colIndex = PandasState.columns.indexOf(sortColumn);
        if (colIndex !== -1) {
            sortedData.sort((a, b) => {
                const aVal = a[colIndex];
                const bVal = b[colIndex];
                if (typeof aVal === 'number' && typeof bVal === 'number') {
                    return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
                }
                return sortDirection === 'asc'
                    ? String(aVal).localeCompare(String(bVal))
                    : String(bVal).localeCompare(String(aVal));
            });
        }
    }

    // Paginate
    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, sortedData.length);
    const pageData = sortedData.slice(startIdx, endIdx);
    const totalPages = Math.ceil(sortedData.length / pageSize);

    // Build table
    let html = '<table><thead><tr><th class="row-idx">#</th>';
    PandasState.columns.forEach(col => {
        const sortClass = col === sortColumn
            ? (sortDirection === 'asc' ? 'sorted-asc' : 'sorted-desc')
            : '';
        html += `<th class="${sortClass}" data-column="${col}">${col}</th>`;
    });
    html += '</tr></thead><tbody>';

    pageData.forEach((row, idx) => {
        const actualIdx = startIdx + idx;
        html += `<tr><td class="row-idx">${actualIdx}</td>`;
        row.forEach(val => {
            html += `<td>${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';

    container.innerHTML = html;

    // Add column click handlers for sorting
    container.querySelectorAll('th[data-column]').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.column;
            if (PandasState.fullscreen.sortColumn === col) {
                PandasState.fullscreen.sortDirection =
                    PandasState.fullscreen.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                PandasState.fullscreen.sortColumn = col;
                PandasState.fullscreen.sortDirection = 'asc';
            }
            renderFullscreenTable();
        });
    });

    // Update pagination info
    const pageInfo = document.getElementById('dfPageInfo');
    if (pageInfo) {
        pageInfo.textContent = `Page ${currentPage} of ${totalPages || 1}`;
    }

    // Update button states
    const prevBtn = document.getElementById('dfPrevPage');
    const nextBtn = document.getElementById('dfNextPage');
    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;
}

function changePage(delta) {
    const { filteredData, pageSize, currentPage } = PandasState.fullscreen;
    const totalPages = Math.ceil(filteredData.length / pageSize);
    const newPage = currentPage + delta;

    if (newPage >= 1 && newPage <= totalPages) {
        PandasState.fullscreen.currentPage = newPage;
        renderFullscreenTable();
    }
}

function updateFooter() {
    const rowCount = document.getElementById('dfRowCount');
    const memory = document.getElementById('dfMemory');

    if (rowCount) {
        rowCount.textContent = `${PandasState.shape[0]} rows × ${PandasState.shape[1]} columns`;
    }
    if (memory) {
        const memEstimate = (JSON.stringify(PandasState.data).length / 1024).toFixed(2);
        memory.textContent = `Memory: ~${memEstimate} KB`;
    }
}

function showSortDialog() {
    const col = prompt('Sort by column:\n' + PandasState.columns.join(', '), PandasState.columns[0]);
    if (col && PandasState.columns.includes(col)) {
        const dir = prompt('Sort direction (asc/desc):', 'asc');
        PandasState.fullscreen.sortColumn = col;
        PandasState.fullscreen.sortDirection = dir === 'desc' ? 'desc' : 'asc';
        renderFullscreenTable();
    }
}

function showFilterDialog() {
    const col = prompt('Filter by column:\n' + PandasState.columns.join(', '), PandasState.columns[0]);
    if (col && PandasState.columns.includes(col)) {
        const value = prompt(`Filter "${col}" contains:`);
        if (value) {
            const colIndex = PandasState.columns.indexOf(col);
            PandasState.fullscreen.filteredData = PandasState.data.filter(row =>
                String(row[colIndex]).toLowerCase().includes(value.toLowerCase())
            );
            PandasState.fullscreen.currentPage = 1;
            renderFullscreenTable();
        }
    }
}

function showStatsDialog() {
    if (typeof DataAnalysis !== 'undefined' && PandasState.data.length > 0) {
        // Convert array data to object format for analysis
        const objData = PandasState.data.map(row => {
            const obj = {};
            PandasState.columns.forEach((col, idx) => {
                obj[col] = row[idx];
            });
            return obj;
        });

        const results = DataAnalysis.analyze(objData, PandasState.columns);
        const stats = results.descriptiveStats;

        let message = 'DESCRIPTIVE STATISTICS\n' + '='.repeat(40) + '\n\n';
        Object.entries(stats).forEach(([col, colStats]) => {
            message += `${col}:\n`;
            if (colStats.type === 'numeric') {
                message += `  Mean: ${colStats.mean}, Std: ${colStats.std}\n`;
                message += `  Min: ${colStats.min}, Max: ${colStats.max}\n`;
            } else {
                message += `  Unique: ${colStats.unique}, Top: ${colStats.top}\n`;
            }
            message += '\n';
        });

        alert(message);
    }
}

function exportDataFrame() {
    const format = prompt('Export format (csv/json):', 'csv');
    if (!format) return;

    let content, filename, type;

    if (format.toLowerCase() === 'json') {
        const objData = PandasState.data.map(row => {
            const obj = {};
            PandasState.columns.forEach((col, idx) => {
                obj[col] = row[idx];
            });
            return obj;
        });
        content = JSON.stringify(objData, null, 2);
        filename = 'dataframe.json';
        type = 'application/json';
    } else {
        // CSV
        const header = PandasState.columns.join(',');
        const rows = PandasState.data.map(row => row.join(','));
        content = header + '\n' + rows.join('\n');
        filename = 'dataframe.csv';
        type = 'text/csv';
    }

    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`DataFrame exported as ${filename}`);
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
    if (typeof AppState !== 'undefined' && AppState.activeCellId) {
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
window.PandasState = PandasState;
window.openFullscreenModal = openFullscreenModal;
window.updateDataFrameDisplay = updateDataFrameDisplay;
