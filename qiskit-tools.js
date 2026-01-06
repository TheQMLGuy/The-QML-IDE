/**
 * Qiskit Tools - IBM Qiskit Mode Functionality
 * Two-way sync: Circuit Visualization ↔ Qiskit Code
 */

// ===========================
// Qiskit State
// ===========================

const QiskitState = {
    nQubits: 3,
    nCbits: 3,
    gates: [],
    backend: 'aer_simulator',
    shots: 1024
};

// Gate definitions for Qiskit
const QISKIT_GATES = {
    h: { name: 'H', qubits: 1, code: (q) => `qc.h(${q})` },
    x: { name: 'X', qubits: 1, code: (q) => `qc.x(${q})` },
    y: { name: 'Y', qubits: 1, code: (q) => `qc.y(${q})` },
    z: { name: 'Z', qubits: 1, code: (q) => `qc.z(${q})` },
    s: { name: 'S', qubits: 1, code: (q) => `qc.s(${q})` },
    t: { name: 'T', qubits: 1, code: (q) => `qc.t(${q})` },
    rx: { name: 'Rx', qubits: 1, parametric: true, code: (q, theta) => `qc.rx(${theta}, ${q})` },
    ry: { name: 'Ry', qubits: 1, parametric: true, code: (q, theta) => `qc.ry(${theta}, ${q})` },
    rz: { name: 'Rz', qubits: 1, parametric: true, code: (q, theta) => `qc.rz(${theta}, ${q})` },
    cx: { name: 'CX', qubits: 2, code: (c, t) => `qc.cx(${c}, ${t})` },
    cz: { name: 'CZ', qubits: 2, code: (c, t) => `qc.cz(${c}, ${t})` },
    swap: { name: 'SWAP', qubits: 2, code: (q1, q2) => `qc.swap(${q1}, ${q2})` },
    ccx: { name: 'CCX', qubits: 3, code: (c1, c2, t) => `qc.ccx(${c1}, ${c2}, ${t})` },
    measure: { name: 'M', qubits: 1, code: (q, c) => `qc.measure(${q}, ${c})` },
    barrier: { name: '║', qubits: 'all', code: () => `qc.barrier()` }
};

// Qiskit code snippets
const QISKIT_SNIPPETS = {
    qiskit_import: `from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

print("Qiskit imports ready!")`,

    qiskit_circuit: `from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create circuit
qc = QuantumCircuit(3, 3)

# Add gates
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

# Simulate
simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled, shots=1024).result()
counts = result.get_counts()

print("Counts:", counts)`,

    qiskit_bell: `from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create Bell state circuit
qc = QuantumCircuit(2, 2)
qc.h(0)         # Put qubit 0 in superposition
qc.cx(0, 1)     # Entangle qubits 0 and 1
qc.measure([0, 1], [0, 1])

# Simulate
simulator = AerSimulator()
result = simulator.run(transpile(qc, simulator), shots=1024).result()
counts = result.get_counts()

print("Bell State Results:", counts)
# Expected: {'00': ~512, '11': ~512}`,

    qiskit_qft: `from qiskit import QuantumCircuit
import numpy as np

def qft(n):
    """Create n-qubit QFT circuit"""
    qc = QuantumCircuit(n)
    
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            qc.cp(np.pi/2**(k-j), k, j)
    
    # Add swaps
    for j in range(n//2):
        qc.swap(j, n-j-1)
    
    return qc

# Create 3-qubit QFT
qft_circuit = qft(3)
print(qft_circuit.draw())`,

    qiskit_vqe: `from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

def ansatz(params, n_qubits):
    """Create variational ansatz"""
    qc = QuantumCircuit(n_qubits)
    
    # Layer of RY rotations
    for i in range(n_qubits):
        qc.ry(params[i], i)
    
    # Entanglement layer
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
    
    return qc

# Create ansatz with random parameters
params = np.random.random(3) * np.pi
circuit = ansatz(params, 3)
print(circuit.draw())`
};

// ===========================
// Initialization
// ===========================

function initializeQiskitTools() {
    console.log('Initializing Qiskit Tools...');

    setupQiskitGatePalette();
    setupQiskitControls();
    drawQiskitCircuit();
    setupQiskitSnippets();

    console.log('Qiskit Tools initialized!');
}

// ===========================
// Gate Palette
// ===========================

function setupQiskitGatePalette() {
    document.querySelectorAll('#qiskitSidebar .gate-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const gate = btn.dataset.gate;
            addQiskitGate(gate);
        });
    });
}

function addQiskitGate(gateType) {
    const gate = QISKIT_GATES[gateType];
    if (!gate) return;

    // For simplicity, add gate to qubit 0 (or 0,1 for 2-qubit gates)
    QiskitState.gates.push({
        type: gateType,
        qubits: gateType === 'cx' ? [0, 1] : gateType === 'ccx' ? [0, 1, 2] : [0],
        params: gate.parametric ? [Math.PI / 4] : []
    });

    drawQiskitCircuit();
    if (AppState.syncEnabled) generateQiskitCode();
}

// ===========================
// Qubit/Cbit Controls
// ===========================

function setupQiskitControls() {
    const qubitInput = document.getElementById('qiskitQubitCount');
    const cbitInput = document.getElementById('qiskitCbitCount');
    const backendSelect = document.getElementById('qiskitBackend');
    const shotsInput = document.getElementById('qiskitShots');

    if (qubitInput) {
        qubitInput.addEventListener('change', (e) => {
            QiskitState.nQubits = parseInt(e.target.value);
            drawQiskitCircuit();
            if (AppState.syncEnabled) generateQiskitCode();
        });
    }

    if (cbitInput) {
        cbitInput.addEventListener('change', (e) => {
            QiskitState.nCbits = parseInt(e.target.value);
            if (AppState.syncEnabled) generateQiskitCode();
        });
    }

    if (backendSelect) {
        backendSelect.addEventListener('change', (e) => {
            QiskitState.backend = e.target.value;
        });
    }

    if (shotsInput) {
        shotsInput.addEventListener('change', (e) => {
            QiskitState.shots = parseInt(e.target.value);
        });
    }
}

// ===========================
// Circuit Drawing
// ===========================

function drawQiskitCircuit() {
    const svg = document.getElementById('qiskitCircuitSvg');
    if (!svg) return;

    const width = svg.getAttribute('width');
    const height = svg.getAttribute('height');
    const nQubits = QiskitState.nQubits;
    const lineSpacing = height / (nQubits + 1);

    // Clear SVG
    svg.innerHTML = '';

    // Draw qubit lines
    for (let i = 0; i < nQubits; i++) {
        const y = (i + 1) * lineSpacing;

        // Qubit label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 10);
        label.setAttribute('y', y + 4);
        label.setAttribute('fill', '#a0a0a0');
        label.setAttribute('font-size', '12');
        label.textContent = `q${i}`;
        svg.appendChild(label);

        // Wire
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', 30);
        line.setAttribute('y1', y);
        line.setAttribute('x2', width - 10);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', '#06b6d4');
        line.setAttribute('stroke-width', 1.5);
        svg.appendChild(line);
    }

    // Draw gates
    let gateX = 60;
    QiskitState.gates.forEach((gate, idx) => {
        const gateInfo = QISKIT_GATES[gate.type];
        if (!gateInfo) return;

        const qIdx = gate.qubits[0];
        const y = (qIdx + 1) * lineSpacing;

        if (gateInfo.qubits === 1) {
            // Single qubit gate
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', gateX - 12);
            rect.setAttribute('y', y - 12);
            rect.setAttribute('width', 24);
            rect.setAttribute('height', 24);
            rect.setAttribute('rx', 4);
            rect.setAttribute('fill', '#1a1d24');
            rect.setAttribute('stroke', '#06b6d4');
            rect.setAttribute('stroke-width', 2);
            svg.appendChild(rect);

            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', gateX);
            text.setAttribute('y', y + 4);
            text.setAttribute('fill', '#06b6d4');
            text.setAttribute('font-size', '12');
            text.setAttribute('text-anchor', 'middle');
            text.textContent = gateInfo.name;
            svg.appendChild(text);
        } else if (gateInfo.qubits === 2) {
            // Two qubit gate (CNOT, etc.)
            const q1 = gate.qubits[0];
            const q2 = gate.qubits[1];
            const y1 = (q1 + 1) * lineSpacing;
            const y2 = (q2 + 1) * lineSpacing;

            // Vertical line
            const vline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            vline.setAttribute('x1', gateX);
            vline.setAttribute('y1', y1);
            vline.setAttribute('x2', gateX);
            vline.setAttribute('y2', y2);
            vline.setAttribute('stroke', '#06b6d4');
            vline.setAttribute('stroke-width', 2);
            svg.appendChild(vline);

            // Control dot
            const ctrl = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            ctrl.setAttribute('cx', gateX);
            ctrl.setAttribute('cy', y1);
            ctrl.setAttribute('r', 5);
            ctrl.setAttribute('fill', '#06b6d4');
            svg.appendChild(ctrl);

            // Target
            const target = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            target.setAttribute('cx', gateX);
            target.setAttribute('cy', y2);
            target.setAttribute('r', 10);
            target.setAttribute('fill', 'none');
            target.setAttribute('stroke', '#06b6d4');
            target.setAttribute('stroke-width', 2);
            svg.appendChild(target);

            // X for target
            const xline1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            xline1.setAttribute('x1', gateX);
            xline1.setAttribute('y1', y2 - 10);
            xline1.setAttribute('x2', gateX);
            xline1.setAttribute('y2', y2 + 10);
            xline1.setAttribute('stroke', '#06b6d4');
            xline1.setAttribute('stroke-width', 2);
            svg.appendChild(xline1);
        }

        gateX += 40;
    });
}

// ===========================
// Code Generation (Viz → Code)
// ===========================

function generateQiskitCode() {
    let code = `from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create circuit
qc = QuantumCircuit(${QiskitState.nQubits}, ${QiskitState.nCbits})

`;

    // Add gates
    QiskitState.gates.forEach(gate => {
        const gateInfo = QISKIT_GATES[gate.type];
        if (!gateInfo) return;

        if (gate.params.length > 0) {
            code += gateInfo.code(...gate.qubits, gate.params[0]) + '\n';
        } else if (gate.qubits.length > 1) {
            code += gateInfo.code(...gate.qubits) + '\n';
        } else {
            code += gateInfo.code(gate.qubits[0]) + '\n';
        }
    });

    code += `
# Measure all qubits
qc.measure_all()

# Simulate
simulator = AerSimulator()
result = simulator.run(transpile(qc, simulator), shots=${QiskitState.shots}).result()
counts = result.get_counts()
print("Results:", counts)`;

    // Insert into active cell if sync enabled
    if (AppState.syncEnabled && AppState.activeCellId) {
        const cellData = AppState.cells.find(c => c.id === AppState.activeCellId);
        if (cellData && AppState.mode === 'qiskit') {
            cellData.editor.setValue(code);
        }
    }
}

// ===========================
// Snippet Buttons
// ===========================

function setupQiskitSnippets() {
    if (typeof CODE_SNIPPETS !== 'undefined') {
        Object.assign(CODE_SNIPPETS, QISKIT_SNIPPETS);
    }
}

// Export for global access
window.initializeQiskitTools = initializeQiskitTools;
window.QISKIT_SNIPPETS = QISKIT_SNIPPETS;
