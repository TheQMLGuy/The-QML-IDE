/**
 * QML IDE - Main Application Logic
 * Unified Quantum Machine Learning IDE with Jupyter notebook-style interface
 */

// ===========================
// Global State
// ===========================

const AppState = {
    mode: 'ml', // 'ml', 'dl', 'qiskit', 'pennylane', or 'data'
    language: 'python', // 'python' or 'r'
    cells: [],
    activeCellId: null,
    pyodide: null,
    pyodideReady: false,
    webRReady: false,
    syncEnabled: true,
    theme: 'dark',
    cellCounter: 0
};

// ===========================
// Code Snippets
// ===========================

const CODE_SNIPPETS = {
    // AI/ML Snippets
    import_torch: `import numpy as np

# Note: PyTorch runs in simulation mode in browser
# For full PyTorch, use a local Python environment

class SimpleNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = self.relu(x @ self.weights[i] + self.biases[i])
        x = x @ self.weights[-1] + self.biases[-1]
        return x

# Create network
nn = SimpleNN([2, 64, 32, 1])
print("Neural Network created with layers:", nn.layers)`,

    import_sklearn: `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")`,

    mlp_model: `import numpy as np

class MLP:
    """Multi-Layer Perceptron with backpropagation"""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layers = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            net = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.activations.append(self.sigmoid(net))
        return self.activations[-1]
    
    def backward(self, y):
        m = y.shape[0]
        deltas = [self.activations[-1] - y]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = deltas[-1] @ self.weights[i].T * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * (self.activations[i].T @ deltas[i]) / m
            self.biases[i] -= self.lr * np.mean(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y)
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage
mlp = MLP([2, 8, 4, 1], learning_rate=0.5)
print("MLP created:", mlp.layers)`,

    train_loop: `# Training loop example
import numpy as np

def train_model(model, X, y, epochs=100, batch_size=32):
    """Generic training loop"""
    n_samples = len(X)
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            predictions = model.forward(X_batch)
            
            # Compute loss
            loss = np.mean((predictions - y_batch) ** 2)
            epoch_loss += loss
            n_batches += 1
            
            # Backward pass
            model.backward(y_batch)
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return losses

print("Training loop defined!")`,

    data_loader: `import numpy as np

def generate_classification_data(n_samples=500, n_features=2, n_classes=2, noise=0.1):
    """Generate synthetic classification data"""
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # Generate cluster center
        center = np.random.randn(n_features) * 2
        
        # Generate points around center
        points = center + np.random.randn(samples_per_class, n_features) * noise * 3
        X.append(points)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]

# Generate data
X, y = generate_classification_data(n_samples=200, n_classes=3)
print(f"Generated {len(X)} samples with {len(np.unique(y))} classes")
print(f"Feature shape: {X.shape}")`,

    // Quantum Snippets
    import_qiskit: `# Qiskit-style quantum computing (browser simulation)
import numpy as np

class QuantumCircuit:
    """Simple quantum circuit simulator"""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0  # |00...0⟩
        self.gates = []
    
    def h(self, qubit):
        """Hadamard gate"""
        self.gates.append(('H', qubit))
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_gate(H, qubit)
        return self
    
    def x(self, qubit):
        """Pauli-X (NOT) gate"""
        self.gates.append(('X', qubit))
        X = np.array([[0, 1], [1, 0]])
        self._apply_single_gate(X, qubit)
        return self
    
    def cx(self, control, target):
        """CNOT gate"""
        self.gates.append(('CX', control, target))
        new_state = self.state.copy()
        for i in range(len(self.state)):
            if (i >> (self.n_qubits - 1 - control)) & 1:
                j = i ^ (1 << (self.n_qubits - 1 - target))
                new_state[i], new_state[j] = self.state[j], self.state[i]
        self.state = new_state
        return self
    
    def _apply_single_gate(self, gate, qubit):
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
            j = i ^ (bit << (self.n_qubits - 1 - qubit))
            new_state[i] += gate[bit, 0] * self.state[j]
            new_state[i] += gate[bit, 1] * self.state[i if bit == 1 else j ^ (1 << (self.n_qubits - 1 - qubit))]
        self.state = new_state
    
    def measure(self):
        probs = np.abs(self.state) ** 2
        return {format(i, f'0{self.n_qubits}b'): p for i, p in enumerate(probs) if p > 1e-10}

# Create and run circuit
qc = QuantumCircuit(2)
qc.h(0).cx(0, 1)
print("Bell State created!")
print("Probabilities:", qc.measure())`,

    import_pennylane: `# PennyLane-style quantum computing (browser simulation)
import numpy as np

class QuantumNode:
    """PennyLane-style quantum node"""
    
    def __init__(self, n_wires):
        self.n_wires = n_wires
        self.operations = []
    
    def RX(self, theta, wires):
        self.operations.append(('RX', theta, wires))
        return self
    
    def RY(self, theta, wires):
        self.operations.append(('RY', theta, wires))
        return self
    
    def RZ(self, theta, wires):
        self.operations.append(('RZ', theta, wires))
        return self
    
    def CNOT(self, wires):
        self.operations.append(('CNOT', wires))
        return self
    
    def execute(self, params=None):
        """Execute the circuit and return expectation values"""
        state = np.zeros(2**self.n_wires, dtype=complex)
        state[0] = 1.0
        
        # Apply operations (simplified)
        for op in self.operations:
            if op[0] in ['RX', 'RY', 'RZ']:
                theta = op[1]
                # Simplified rotation
                pass
        
        probs = np.abs(state) ** 2
        return probs
    
    def draw(self):
        """Draw the circuit"""
        lines = [f"q{i}: ──" for i in range(self.n_wires)]
        for op in self.operations:
            gate = op[0]
            if gate in ['RX', 'RY', 'RZ']:
                wire = op[2]
                lines[wire] += f"─{gate}({op[1]:.2f})──"
            elif gate == 'CNOT':
                lines[op[1][0]] += "─●──"
                lines[op[1][1]] += "─X──"
        
        return "\\n".join(lines)

# Create quantum node
qnode = QuantumNode(3)
qnode.RX(np.pi/4, 0).RY(np.pi/2, 1).CNOT([0, 1])
print("Quantum Node:")
print(qnode.draw())`,

    bell_state: `import numpy as np

def create_bell_state(state_type='phi_plus'):
    """
    Create one of the four Bell states
    |Φ+⟩ = (|00⟩ + |11⟩) / √2
    |Φ-⟩ = (|00⟩ - |11⟩) / √2  
    |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    |Ψ-⟩ = (|01⟩ - |10⟩) / √2
    """
    sqrt2 = np.sqrt(2)
    
    states = {
        'phi_plus':  np.array([1, 0, 0, 1]) / sqrt2,
        'phi_minus': np.array([1, 0, 0, -1]) / sqrt2,
        'psi_plus':  np.array([0, 1, 1, 0]) / sqrt2,
        'psi_minus': np.array([0, 1, -1, 0]) / sqrt2
    }
    
    return states.get(state_type, states['phi_plus'])

# Create all Bell states
for name in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
    state = create_bell_state(name)
    probs = np.abs(state) ** 2
    print(f"|{name}⟩: {['|00⟩', '|01⟩', '|10⟩', '|11⟩']} = {probs}")
    print(f"  Non-zero: {[(f'|{i:02b}⟩', f'{p:.2f}') for i, p in enumerate(probs) if p > 0.01]}")`,

    qft: `import numpy as np

def qft_matrix(n_qubits):
    """Generate the Quantum Fourier Transform matrix"""
    N = 2 ** n_qubits
    omega = np.exp(2j * np.pi / N)
    
    # Create QFT matrix
    qft = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            qft[i, j] = omega ** (i * j)
    
    return qft / np.sqrt(N)

def apply_qft(state):
    """Apply QFT to a quantum state"""
    n_qubits = int(np.log2(len(state)))
    qft = qft_matrix(n_qubits)
    return qft @ state

# Example: Apply QFT to |01⟩
initial_state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
final_state = apply_qft(initial_state)

print("Initial state |01⟩:")
print(f"  Amplitudes: {initial_state}")

print("\\nAfter QFT:")
for i, amp in enumerate(final_state):
    if np.abs(amp) > 1e-10:
        print(f"  |{i:02b}⟩: {amp:.3f} (prob: {np.abs(amp)**2:.3f})")`,

    vqe: `import numpy as np

def vqe_ansatz(params, n_qubits=2):
    """
    Variational Quantum Eigensolver ansatz
    Creates a parameterized quantum state
    """
    # Start with |00...0⟩
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Apply parameterized rotations
    def ry_gate(theta):
        c, s = np.cos(theta/2), np.sin(theta/2)
        return np.array([[c, -s], [s, c]])
    
    # Simplified: Apply RY to each qubit
    for i, theta in enumerate(params[:n_qubits]):
        # Single qubit rotation (simplified)
        pass
    
    return state

def compute_energy(params, hamiltonian):
    """Compute expectation value of Hamiltonian"""
    state = vqe_ansatz(params, n_qubits=2)
    energy = np.real(state.conj() @ hamiltonian @ state)
    return energy

# Define simple Hamiltonian (H2 molecule approximation)
H = np.array([
    [-1.0, 0, 0, 0.5],
    [0, 0.5, -0.5, 0],
    [0, -0.5, 0.5, 0],
    [0.5, 0, 0, -1.0]
])

# VQE optimization
from scipy.optimize import minimize

result = minimize(
    lambda p: compute_energy(p, H),
    x0=np.random.randn(4) * 0.1,
    method='COBYLA'
)

print(f"VQE Ground State Energy: {result.fun:.4f}")
print(f"Optimal Parameters: {result.x}")`
};

// ===========================
// Initialization
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    console.log('Initializing QML IDE...');

    // Initialize Workflow Engine first (for phase-based flow)
    if (typeof WorkflowEngine !== 'undefined') {
        WorkflowEngine.init();
        console.log('Workflow Engine started - Phase-based flow active');

        // Don't initialize IDE components yet - they'll be initialized when entering Phase 4
        // Initialize Python runtime in background
        await initializePyodide();
        return;
    }

    // Fallback: Direct IDE mode (no workflow)
    console.log('Direct IDE mode - no workflow');

    // Setup event listeners
    setupModeToggle();
    setupHeaderButtons();
    setupConsoleToggle();
    setupSnippetButtons();
    setupLearningRateSlider();

    // Initialize Tab Manager (will create initial cell)
    if (typeof TabManager !== 'undefined') {
        TabManager.init();
    } else {
        // Fallback: Create initial cell if no TabManager
        addCell();
    }

    // Initialize Python runtime
    await initializePyodide();

    // Initialize ML tools (sklearn)
    if (typeof initializeMLTools === 'function') {
        initializeMLTools();
    }

    // Initialize DL tools (PyTorch)
    if (typeof initializeAITools === 'function') {
        initializeAITools();
    }

    // Initialize Qiskit tools
    if (typeof initializeQiskitTools === 'function') {
        initializeQiskitTools();
    }

    // Initialize PennyLane tools
    if (typeof initializePennylaneTools === 'function') {
        initializePennylaneTools();
    }

    // Initialize Pandas tools
    if (typeof initializePandasTools === 'function') {
        initializePandasTools();
    }

    // Initialize Data Analysis tools
    if (typeof initializeDataTools === 'function') {
        initializeDataTools();
    }

    console.log('QML IDE initialized!');
}

// ===========================
// Pyodide Initialization
// ===========================

async function initializePyodide() {
    const statusEl = document.getElementById('runtimeStatus');
    const statusText = statusEl?.querySelector('.status-text');

    try {
        if (statusText) statusText.textContent = 'Loading Python Runtime...';

        AppState.pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
        });

        if (statusText) statusText.textContent = 'Installing NumPy & SciPy...';
        await AppState.pyodide.loadPackage(['numpy', 'scipy']);

        if (statusText) statusText.textContent = 'Installing Pandas...';
        try {
            await AppState.pyodide.loadPackage(['pandas']);
            console.log('Pandas loaded successfully');
        } catch (e) {
            console.warn('Pandas not available:', e.message);
        }

        if (statusText) statusText.textContent = 'Installing scikit-learn...';
        try {
            await AppState.pyodide.loadPackage(['scikit-learn']);
            console.log('scikit-learn loaded successfully');
        } catch (e) {
            console.warn('scikit-learn not available:', e.message);
        }

        AppState.pyodideReady = true;
        if (statusEl) {
            statusEl.classList.remove('loading');
            statusEl.classList.add('ready');
        }
        if (statusText) statusText.textContent = 'Python Ready';

        console.log('Pyodide initialized with packages: numpy, scipy, pandas, scikit-learn');

    } catch (error) {
        console.error('Failed to initialize Pyodide:', error);
        if (statusEl) {
            statusEl.classList.remove('loading');
            statusEl.classList.add('error');
        }
        if (statusText) statusText.textContent = 'Runtime Error';
    }
}

// ===========================
// Mode Toggle
// ===========================

function setupModeToggle() {
    const mlBtn = document.getElementById('mlModeBtn');
    const dlBtn = document.getElementById('dlModeBtn');
    const qiskitBtn = document.getElementById('qiskitModeBtn');
    const pennylaneBtn = document.getElementById('pennylaneModeBtn');
    const dataBtn = document.getElementById('dataModeBtn');

    mlBtn.addEventListener('click', () => switchMode('ml'));
    dlBtn.addEventListener('click', () => switchMode('dl'));
    qiskitBtn.addEventListener('click', () => switchMode('qiskit'));
    pennylaneBtn.addEventListener('click', () => switchMode('pennylane'));
    dataBtn.addEventListener('click', () => switchMode('data'));

    // Pandas mode
    const pandasBtn = document.getElementById('pandasModeBtn');
    if (pandasBtn) {
        pandasBtn.addEventListener('click', () => switchMode('pandas'));
    }

    // Mode cycle button
    const cycleBtn = document.getElementById('modeCycleBtn');
    if (cycleBtn) {
        cycleBtn.addEventListener('click', cycleMode);
    }

    // Tab key to cycle modes
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            cycleMode();
        }
    });
}

// Mode order for cycling
const MODE_ORDER = ['ml', 'dl', 'qiskit', 'pennylane', 'pandas', 'data'];

function cycleMode() {
    const currentIdx = MODE_ORDER.indexOf(AppState.mode);
    const nextIdx = (currentIdx + 1) % MODE_ORDER.length;
    switchMode(MODE_ORDER[nextIdx]);
}

function switchMode(mode) {
    AppState.mode = mode;

    // Update toggle buttons
    const buttons = {
        ml: document.getElementById('mlModeBtn'),
        dl: document.getElementById('dlModeBtn'),
        qiskit: document.getElementById('qiskitModeBtn'),
        pennylane: document.getElementById('pennylaneModeBtn'),
        pandas: document.getElementById('pandasModeBtn'),
        data: document.getElementById('dataModeBtn')
    };

    Object.entries(buttons).forEach(([key, btn]) => {
        if (btn) btn.classList.toggle('active', mode === key);
    });

    // Update sidebars
    const sidebars = {
        ml: document.getElementById('mlSidebar'),
        dl: document.getElementById('dlSidebar'),
        qiskit: document.getElementById('qiskitSidebar'),
        pennylane: document.getElementById('pennylaneSidebar'),
        pandas: document.getElementById('pandasSidebar'),
        data: document.getElementById('dataSidebar')
    };

    Object.entries(sidebars).forEach(([key, sidebar]) => {
        if (sidebar) sidebar.classList.toggle('hidden', mode !== key);
    });

    // Update cell styling
    document.querySelectorAll('.code-cell').forEach(cell => {
        cell.classList.remove('ml-mode', 'dl-mode', 'qiskit-mode', 'pennylane-mode', 'pandas-mode', 'data-mode');
        cell.classList.add(`${mode}-mode`);
    });

    console.log(`Switched to ${mode} mode`);
}

// ===========================
// Header Buttons
// ===========================

function setupHeaderButtons() {
    document.getElementById('runAllBtn').addEventListener('click', runAllCells);
    document.getElementById('addCellBtn').addEventListener('click', () => addCell());
    document.getElementById('addCellInline').addEventListener('click', () => addCell());
    document.getElementById('saveBtn').addEventListener('click', saveNotebook);
    document.getElementById('loadBtn').addEventListener('click', () => document.getElementById('loadInput').click());
    document.getElementById('loadInput').addEventListener('change', loadNotebook);
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
}

function toggleTheme() {
    AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', AppState.theme);
    document.getElementById('themeIcon').textContent = AppState.theme === 'dark' ? '🌙' : '☀️';
}

// ===========================
// Console Toggle
// ===========================

function setupConsoleToggle() {
    const header = document.getElementById('consoleHeader');
    const console = document.getElementById('outputConsole');
    const clearBtn = document.getElementById('clearOutputBtn');

    header.addEventListener('click', () => {
        console.classList.toggle('collapsed');
    });

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        document.getElementById('outputArea').innerHTML = '';
    });
}

// ===========================
// Cell Management
// ===========================

function addCell(initialCode = '') {
    const cellId = ++AppState.cellCounter;
    const container = document.getElementById('notebookContainer');

    const cellEl = document.createElement('div');
    cellEl.className = 'code-cell';
    cellEl.id = `cell-${cellId}`;
    cellEl.classList.add(`${AppState.mode}-mode`);

    const modeLabels = {
        ml: 'ML',
        dl: 'DL',
        qiskit: 'Qiskit',
        pennylane: 'PL',
        pandas: 'Pandas',
        data: 'Data'
    };
    const modeLabel = modeLabels[AppState.mode] || 'ML';

    cellEl.innerHTML = `
        <div class="cell-header">
            <div class="cell-number">
                <span>In [${cellId}]</span>
                <span class="cell-type-badge ${AppState.mode}">${modeLabel}</span>
            </div>
            <div class="cell-actions">
                <button class="cell-btn run" title="Run Cell (Shift+Enter)">▶ Run</button>
                <button class="cell-btn delete" title="Delete Cell">✕</button>
            </div>
        </div>
        <div class="cell-editor">
            <textarea id="editor-${cellId}">${initialCode}</textarea>
        </div>
        <div class="cell-output" id="output-${cellId}"></div>
    `;

    container.appendChild(cellEl);

    // Initialize CodeMirror
    const textarea = document.getElementById(`editor-${cellId}`);
    const editor = CodeMirror.fromTextArea(textarea, {
        mode: 'python',
        theme: 'material-darker',
        lineNumbers: true,
        matchBrackets: true,
        autoCloseBrackets: true,
        indentUnit: 4,
        tabSize: 4,
        indentWithTabs: false,
        lineWrapping: true,
        extraKeys: {
            'Shift-Enter': () => runCell(cellId),
            'Ctrl-Enter': () => runCell(cellId)
        }
    });

    // Store cell data
    AppState.cells.push({
        id: cellId,
        editor: editor,
        mode: AppState.mode
    });

    // Register cell with active tab
    if (typeof TabManager !== 'undefined') {
        TabManager.addCellToActiveTab(cellId);
    }

    // Setup cell events
    cellEl.querySelector('.cell-btn.run').addEventListener('click', () => runCell(cellId));
    cellEl.querySelector('.cell-btn.delete').addEventListener('click', () => deleteCell(cellId));

    // Focus handling
    editor.on('focus', () => {
        document.querySelectorAll('.code-cell').forEach(c => c.classList.remove('focused'));
        cellEl.classList.add('focused');
        AppState.activeCellId = cellId;
    });

    // Focus the new cell
    editor.focus();

    return cellId;
}

function deleteCell(cellId) {
    const cellEl = document.getElementById(`cell-${cellId}`);
    if (cellEl) {
        cellEl.remove();
        AppState.cells = AppState.cells.filter(c => c.id !== cellId);
    }

    // Add new cell if no cells left
    if (AppState.cells.length === 0) {
        addCell();
    }
}

async function runCell(cellId) {
    const cellData = AppState.cells.find(c => c.id === cellId);
    if (!cellData) return;

    const code = cellData.editor.getValue();
    const outputEl = document.getElementById(`output-${cellId}`);

    if (!AppState.pyodideReady) {
        outputEl.className = 'cell-output error';
        outputEl.textContent = 'Python runtime not ready. Please wait...';
        return;
    }

    try {
        outputEl.className = 'cell-output';
        outputEl.textContent = 'Running...';

        // Capture stdout
        AppState.pyodide.runPython(`
import sys
from io import StringIO
_stdout_capture = StringIO()
sys.stdout = _stdout_capture
        `);

        // Run the code
        const result = await AppState.pyodide.runPythonAsync(code);

        // Get captured output
        const stdout = AppState.pyodide.runPython(`
sys.stdout = sys.__stdout__
_stdout_capture.getvalue()
        `);

        // Display output
        let output = stdout || '';
        if (result !== undefined && result !== null) {
            output += (output ? '\n' : '') + String(result);
        }

        outputEl.className = 'cell-output success';
        outputEl.textContent = output || 'Done';

        // Also log to console
        logToConsole(output, 'success');

    } catch (error) {
        outputEl.className = 'cell-output error';
        outputEl.textContent = error.message || String(error);
        logToConsole(error.message || String(error), 'error');
    }
}

async function runAllCells() {
    for (const cell of AppState.cells) {
        await runCell(cell.id);
    }
}

// ===========================
// Console Logging
// ===========================

function logToConsole(message, type = 'info') {
    const outputArea = document.getElementById('outputArea');
    const line = document.createElement('div');
    line.className = `output-line output-${type}`;
    line.textContent = message;
    outputArea.appendChild(line);
    outputArea.scrollTop = outputArea.scrollHeight;
}

// ===========================
// Snippet Buttons
// ===========================

function setupSnippetButtons() {
    document.querySelectorAll('.snippet-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const snippetKey = btn.dataset.snippet;
            const code = CODE_SNIPPETS[snippetKey];
            if (code) {
                insertSnippet(code);
            }
        });
    });
}

function insertSnippet(code) {
    // Insert into active cell or create new one
    if (AppState.activeCellId) {
        const cellData = AppState.cells.find(c => c.id === AppState.activeCellId);
        if (cellData) {
            const currentCode = cellData.editor.getValue();
            cellData.editor.setValue(currentCode + (currentCode ? '\n\n' : '') + code);
            cellData.editor.focus();
            return;
        }
    }

    // Create new cell with snippet
    addCell(code);
}

// ===========================
// Learning Rate Slider
// ===========================

function setupLearningRateSlider() {
    const slider = document.getElementById('learningRate');
    const value = document.getElementById('lrValue');

    if (slider && value) {
        slider.addEventListener('input', () => {
            value.textContent = parseFloat(slider.value).toFixed(4);
        });
    }
}

// ===========================
// Save/Load Notebook
// ===========================

function saveNotebook() {
    const notebook = {
        version: '1.0',
        mode: AppState.mode,
        cells: AppState.cells.map(c => ({
            id: c.id,
            mode: c.mode,
            code: c.editor.getValue()
        }))
    };

    const blob = new Blob([JSON.stringify(notebook, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'qml_notebook.json';
    a.click();

    URL.revokeObjectURL(url);
    logToConsole('Notebook saved!', 'success');
}

function loadNotebook(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const notebook = JSON.parse(e.target.result);

            // Clear existing cells
            AppState.cells.forEach(c => deleteCell(c.id));
            AppState.cells = [];

            // Switch to saved mode
            switchMode(notebook.mode || 'ai');

            // Recreate cells
            notebook.cells.forEach(cellData => {
                addCell(cellData.code);
            });

            logToConsole('Notebook loaded!', 'success');

        } catch (error) {
            logToConsole('Failed to load notebook: ' + error.message, 'error');
        }
    };
    reader.readAsText(file);

    // Reset input
    event.target.value = '';
}

// ===========================
// Export for other modules
// ===========================

window.QMLApp = {
    state: AppState,
    addCell,
    runCell,
    switchMode,
    insertSnippet,
    logToConsole
};
