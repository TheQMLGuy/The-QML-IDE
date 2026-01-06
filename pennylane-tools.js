/**
 * PennyLane Tools - Xanadu PennyLane Mode Functionality
 * Two-way sync: QNode Visualization ↔ PennyLane Code
 */

// ===========================
// PennyLane State
// ===========================

const PennylaneState = {
    nWires: 4,
    device: 'default.qubit',
    interface: 'autograd',
    operations: [],
    selectedWire: 0
};

// Operation definitions for PennyLane
const PENNYLANE_OPS = {
    Hadamard: { name: 'H', wires: 1, code: (w) => `qml.Hadamard(wires=${w})` },
    PauliX: { name: 'X', wires: 1, code: (w) => `qml.PauliX(wires=${w})` },
    PauliY: { name: 'Y', wires: 1, code: (w) => `qml.PauliY(wires=${w})` },
    PauliZ: { name: 'Z', wires: 1, code: (w) => `qml.PauliZ(wires=${w})` },
    RX: { name: 'RX', wires: 1, parametric: true, code: (theta, w) => `qml.RX(${theta}, wires=${w})` },
    RY: { name: 'RY', wires: 1, parametric: true, code: (theta, w) => `qml.RY(${theta}, wires=${w})` },
    RZ: { name: 'RZ', wires: 1, parametric: true, code: (theta, w) => `qml.RZ(${theta}, wires=${w})` },
    Rot: { name: 'Rot', wires: 1, parametric: true, code: (phi, theta, omega, w) => `qml.Rot(${phi}, ${theta}, ${omega}, wires=${w})` },
    CNOT: { name: 'CNOT', wires: 2, code: (w) => `qml.CNOT(wires=[${w[0]}, ${w[1]}])` },
    CZ: { name: 'CZ', wires: 2, code: (w) => `qml.CZ(wires=[${w[0]}, ${w[1]}])` },
    SWAP: { name: 'SWAP', wires: 2, code: (w) => `qml.SWAP(wires=[${w[0]}, ${w[1]}])` },
    StronglyEntangling: { name: 'SE', template: true, code: (shape) => `qml.StronglyEntanglingLayers(weights, wires=range(${shape}))` },
    BasicEntangler: { name: 'BE', template: true, code: (shape) => `qml.BasicEntanglerLayers(weights, wires=range(${shape}))` }
};

// PennyLane code snippets
const PENNYLANE_SNIPPETS = {
    pennylane_import: `import pennylane as qml
from pennylane import numpy as np

print("PennyLane imports ready!")
print(f"PennyLane version: {qml.__version__}")`,

    pennylane_qnode: `import pennylane as qml
from pennylane import numpy as np

# Create device
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit(params):
    # Apply rotations
    for i in range(4):
        qml.RY(params[i], wires=i)
    
    # Entangling layer
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    return qml.expval(qml.PauliZ(0))

# Run circuit
params = np.random.random(4) * np.pi
result = circuit(params)
print(f"Expectation value: {result:.4f}")`,

    pennylane_vqe: `import pennylane as qml
from pennylane import numpy as np

# Define the Hamiltonian (H2 molecule example)
coeffs = [0.2252, 0.3435, -0.4347, 0.0910, 0.0910]
obs = [
    qml.Identity(0),
    qml.PauliZ(0),
    qml.PauliZ(1),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1)
]
H = qml.Hamiltonian(coeffs, obs)

# Device
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def ansatz(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    return qml.expval(H)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.4)
params = np.random.random(3)

for i in range(100):
    params, energy = opt.step_and_cost(ansatz, params)
    if i % 20 == 0:
        print(f"Step {i}: Energy = {energy:.6f}")

print(f"Ground state energy: {energy:.6f}")`,

    pennylane_qml: `import pennylane as qml
from pennylane import numpy as np

# Quantum classifier
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_classifier(inputs, weights):
    # Encode inputs
    for i in range(4):
        qml.RY(inputs[i], wires=i)
    
    # Variational layers
    for layer_weights in weights:
        for i in range(4):
            qml.RY(layer_weights[i, 0], wires=i)
            qml.RZ(layer_weights[i, 1], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Sample prediction
inputs = np.random.random(4)
weights = np.random.random((2, 4, 2))
output = quantum_classifier(inputs, weights)
print(f"Classifier output: {output}")`,

    pennylane_grad: `import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Compute gradients
params = np.array([0.5, 0.7], requires_grad=True)
grad_fn = qml.grad(circuit)
gradients = grad_fn(params)

print(f"Parameters: {params}")
print(f"Gradients: {gradients}")
print(f"Circuit output: {circuit(params)}")`
};

// ===========================
// Initialization
// ===========================

function initializePennylaneTools() {
    console.log('Initializing PennyLane Tools...');

    setupPennylaneGatePalette();
    setupPennylaneControls();
    drawPennylaneBloch();
    setupPennylaneSnippets();

    console.log('PennyLane Tools initialized!');
}

// ===========================
// Gate Palette
// ===========================

function setupPennylaneGatePalette() {
    document.querySelectorAll('#pennylaneSidebar .gate-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const gate = btn.dataset.gate;
            addPennylaneOp(gate);
        });
    });
}

function addPennylaneOp(opType) {
    const op = PENNYLANE_OPS[opType];
    if (!op) return;

    PennylaneState.operations.push({
        type: opType,
        wires: op.wires === 2 ? [0, 1] : [PennylaneState.selectedWire],
        params: op.parametric ? [Math.PI / 4] : []
    });

    updateExpvalDisplay();
    if (AppState.syncEnabled) generatePennylaneCode();
}

// ===========================
// Device Controls
// ===========================

function setupPennylaneControls() {
    const deviceSelect = document.getElementById('pennylaneDevice');
    const wiresInput = document.getElementById('pennylaneWires');
    const interfaceSelect = document.getElementById('pennylaneInterface');
    const qubitSelect = document.getElementById('pennylaneQubitSelect');

    if (deviceSelect) {
        deviceSelect.addEventListener('change', (e) => {
            PennylaneState.device = e.target.value;
            if (AppState.syncEnabled) generatePennylaneCode();
        });
    }

    if (wiresInput) {
        wiresInput.addEventListener('change', (e) => {
            PennylaneState.nWires = parseInt(e.target.value);
            updateWireSelect();
            if (AppState.syncEnabled) generatePennylaneCode();
        });
    }

    if (interfaceSelect) {
        interfaceSelect.addEventListener('change', (e) => {
            PennylaneState.interface = e.target.value;
        });
    }

    if (qubitSelect) {
        qubitSelect.addEventListener('change', (e) => {
            PennylaneState.selectedWire = parseInt(e.target.value);
            drawPennylaneBloch();
        });
    }
}

function updateWireSelect() {
    const select = document.getElementById('pennylaneQubitSelect');
    if (!select) return;

    select.innerHTML = '';
    for (let i = 0; i < PennylaneState.nWires; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Wire ${i}`;
        select.appendChild(option);
    }
}

// ===========================
// Bloch Sphere
// ===========================

function drawPennylaneBloch() {
    const canvas = document.getElementById('pennylaneBlochCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const r = Math.min(cx, cy) - 20;

    // Clear
    ctx.fillStyle = '#1a1d24';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw sphere outline
    ctx.strokeStyle = '#ec4899';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Draw axes
    ctx.strokeStyle = '#4b5563';
    ctx.lineWidth = 1;

    // X axis
    ctx.beginPath();
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.stroke();

    // Z axis
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy + r);
    ctx.stroke();

    // Y axis (ellipse for 3D effect)
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.ellipse(cx, cy, r, r * 0.3, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Axis labels
    ctx.fillStyle = '#ec4899';
    ctx.font = '12px sans-serif';
    ctx.fillText('|0⟩', cx + 5, cy - r + 10);
    ctx.fillText('|1⟩', cx + 5, cy + r - 5);
    ctx.fillText('+X', cx + r - 15, cy - 5);
    ctx.fillText('-X', cx - r + 5, cy - 5);

    // Draw state vector
    const theta = Math.PI / 4; // Example angle
    const phi = Math.PI / 6;
    const x = r * Math.sin(theta) * Math.cos(phi);
    const z = r * Math.cos(theta);

    ctx.strokeStyle = '#ec4899';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + x, cy - z);
    ctx.stroke();

    // Arrow head
    ctx.fillStyle = '#ec4899';
    ctx.beginPath();
    ctx.arc(cx + x, cy - z, 5, 0, Math.PI * 2);
    ctx.fill();

    // Update state formula
    const stateFormula = document.getElementById('pennylaneStateFormula');
    if (stateFormula) {
        stateFormula.textContent = 'cos(π/8)|0⟩ + e^(iπ/6)sin(π/8)|1⟩';
    }
}

// ===========================
// Expectation Values
// ===========================

function updateExpvalDisplay() {
    const expvalZ0 = document.getElementById('expvalZ0');
    const expvalX0 = document.getElementById('expvalX0');

    if (expvalZ0) expvalZ0.textContent = (Math.random() * 2 - 1).toFixed(4);
    if (expvalX0) expvalX0.textContent = (Math.random() * 2 - 1).toFixed(4);
}

// ===========================
// Code Generation (Viz → Code)
// ===========================

function generatePennylaneCode() {
    let code = `import pennylane as qml
from pennylane import numpy as np

# Create device
dev = qml.device("${PennylaneState.device}", wires=${PennylaneState.nWires})

@qml.qnode(dev, interface="${PennylaneState.interface}")
def circuit(params):
`;

    // Add operations
    let paramIdx = 0;
    PennylaneState.operations.forEach((op, idx) => {
        const opInfo = PENNYLANE_OPS[op.type];
        if (!opInfo) return;

        if (opInfo.parametric) {
            code += `    ${opInfo.code(`params[${paramIdx}]`, op.wires[0])}\n`;
            paramIdx++;
        } else if (opInfo.wires === 2) {
            code += `    ${opInfo.code(op.wires)}\n`;
        } else {
            code += `    ${opInfo.code(op.wires[0])}\n`;
        }
    });

    code += `    return qml.expval(qml.PauliZ(0))

# Run circuit
params = np.random.random(${Math.max(1, paramIdx)})
result = circuit(params)
print(f"Result: {result:.4f}")`;

    // Insert into active cell if sync enabled
    if (AppState.syncEnabled && AppState.activeCellId) {
        const cellData = AppState.cells.find(c => c.id === AppState.activeCellId);
        if (cellData && AppState.mode === 'pennylane') {
            cellData.editor.setValue(code);
        }
    }
}

// ===========================
// Snippet Buttons
// ===========================

function setupPennylaneSnippets() {
    if (typeof CODE_SNIPPETS !== 'undefined') {
        Object.assign(CODE_SNIPPETS, PENNYLANE_SNIPPETS);
    }
}

// Export for global access
window.initializePennylaneTools = initializePennylaneTools;
window.PENNYLANE_SNIPPETS = PENNYLANE_SNIPPETS;
