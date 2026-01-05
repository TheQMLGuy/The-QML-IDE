/**
 * Quantum Tools - Sidebar components for Quantum mode
 * Integrates with existing Qiskit-Pennylane-IDE components
 */

// ===========================
// Circuit Preview
// ===========================

const CircuitPreview = {
    svg: null,
    numQubits: 3,
    gates: [],

    init() {
        this.svg = document.getElementById('circuitSvg');
        if (!this.svg) return;

        const qubitInput = document.getElementById('qubitCount');
        if (qubitInput) {
            qubitInput.addEventListener('change', () => {
                this.numQubits = parseInt(qubitInput.value) || 3;
                this.updateQubitSelect();
                this.render();
            });
        }

        this.setupGateButtons();
        this.render();
    },

    setupGateButtons() {
        document.querySelectorAll('#quantumSidebar .gate-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const gate = btn.dataset.gate;
                if (gate) {
                    this.addGate(gate, 0);
                }
            });
        });
    },

    addGate(type, qubit, params = {}) {
        this.gates.push({ type, qubit, params, time: this.gates.length });
        this.render();
        this.updateVisualization();

        // Log to console
        if (window.QMLApp) {
            window.QMLApp.logToConsole(`Added ${type} gate to qubit ${qubit}`, 'info');
        }
    },

    clearGates() {
        this.gates = [];
        this.render();
        this.updateVisualization();
    },

    updateQubitSelect() {
        const select = document.getElementById('qubitSelect');
        if (!select) return;

        select.innerHTML = '';
        for (let i = 0; i < this.numQubits; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `Qubit ${i}`;
            select.appendChild(option);
        }
    },

    render() {
        if (!this.svg) return;

        const width = 280;
        const height = 20 + this.numQubits * 30;

        this.svg.setAttribute('width', width);
        this.svg.setAttribute('height', height);

        // Clear SVG
        this.svg.innerHTML = '';

        const padding = { left: 40, right: 20, top: 15, bottom: 5 };
        const wireSpacing = 25;

        // Draw qubit wires
        for (let i = 0; i < this.numQubits; i++) {
            const y = padding.top + i * wireSpacing;

            // Wire label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', 10);
            label.setAttribute('y', y + 4);
            label.setAttribute('fill', '#a0a0a0');
            label.setAttribute('font-size', '11');
            label.setAttribute('font-family', 'JetBrains Mono');
            label.textContent = `|q${i}⟩`;
            this.svg.appendChild(label);

            // Wire line
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', padding.left);
            line.setAttribute('y1', y);
            line.setAttribute('x2', width - padding.right);
            line.setAttribute('y2', y);
            line.setAttribute('stroke', '#6b7280');
            line.setAttribute('stroke-width', '1');
            this.svg.appendChild(line);
        }

        // Draw gates
        const gateWidth = 28;
        const gateSpacing = 35;

        // Group gates by time step
        const timeSteps = {};
        this.gates.forEach(gate => {
            if (!timeSteps[gate.time]) {
                timeSteps[gate.time] = [];
            }
            timeSteps[gate.time].push(gate);
        });

        Object.entries(timeSteps).forEach(([time, gates]) => {
            gates.forEach(gate => {
                const x = padding.left + 20 + parseInt(time) * gateSpacing;
                const y = padding.top + gate.qubit * wireSpacing;

                this.renderGate(gate.type, x, y, gate.params);
            });
        });
    },

    renderGate(type, x, y, params = {}) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        // Gate colors
        const colors = {
            H: '#3b82f6',
            X: '#ef4444',
            Y: '#10b981',
            Z: '#8b5cf6',
            S: '#f59e0b',
            T: '#ec4899',
            RX: '#f59e0b',
            RY: '#f59e0b',
            RZ: '#f59e0b',
            CNOT: '#10b981',
            CZ: '#10b981',
            SWAP: '#3b82f6',
            M: '#6b7280'
        };

        const color = colors[type] || '#8b5cf6';

        if (type === 'M') {
            // Measurement symbol
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x - 12);
            rect.setAttribute('y', y - 10);
            rect.setAttribute('width', 24);
            rect.setAttribute('height', 20);
            rect.setAttribute('rx', 3);
            rect.setAttribute('fill', '#252830');
            rect.setAttribute('stroke', color);
            rect.setAttribute('stroke-width', '1.5');
            group.appendChild(rect);

            // Meter arc
            const arc = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            arc.setAttribute('d', `M ${x - 6} ${y + 5} A 8 8 0 0 1 ${x + 6} ${y + 5}`);
            arc.setAttribute('fill', 'none');
            arc.setAttribute('stroke', color);
            arc.setAttribute('stroke-width', '1.5');
            group.appendChild(arc);

            // Needle
            const needle = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            needle.setAttribute('x1', x);
            needle.setAttribute('y1', y + 5);
            needle.setAttribute('x2', x + 4);
            needle.setAttribute('y2', y - 5);
            needle.setAttribute('stroke', color);
            needle.setAttribute('stroke-width', '1.5');
            group.appendChild(needle);
        } else {
            // Standard gate box
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x - 12);
            rect.setAttribute('y', y - 10);
            rect.setAttribute('width', 24);
            rect.setAttribute('height', 20);
            rect.setAttribute('rx', 3);
            rect.setAttribute('fill', '#252830');
            rect.setAttribute('stroke', color);
            rect.setAttribute('stroke-width', '1.5');
            group.appendChild(rect);

            // Gate label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', y + 4);
            text.setAttribute('fill', color);
            text.setAttribute('font-size', '10');
            text.setAttribute('font-weight', '600');
            text.setAttribute('font-family', 'JetBrains Mono');
            text.setAttribute('text-anchor', 'middle');
            text.textContent = type;
            group.appendChild(text);
        }

        this.svg.appendChild(group);
    },

    updateVisualization() {
        // Update Bloch sphere and probabilities if available
        if (window.BlochSphereViz) {
            BlochSphereViz.update();
        }
        ProbabilityDisplay.update(this.gates);
    }
};

// ===========================
// Bloch Sphere Visualization
// ===========================

const BlochSphereViz = {
    canvas: null,
    ctx: null,
    theta: 0,
    phi: 0,
    selectedQubit: 0,
    animating: false,

    init() {
        this.canvas = document.getElementById('blochCanvas');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');

        const select = document.getElementById('qubitSelect');
        if (select) {
            select.addEventListener('change', () => {
                this.selectedQubit = parseInt(select.value);
                this.update();
            });
        }

        this.render();
    },

    update() {
        // Calculate state based on gates
        // For now, just animate to show it's working
        this.animateToState(Math.random() * Math.PI, Math.random() * 2 * Math.PI);
    },

    animateToState(targetTheta, targetPhi) {
        if (this.animating) return;

        this.animating = true;
        const startTheta = this.theta;
        const startPhi = this.phi;
        const duration = 300;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);

            this.theta = startTheta + (targetTheta - startTheta) * eased;
            this.phi = startPhi + (targetPhi - startPhi) * eased;

            this.render();
            this.updateStateFormula();

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                this.animating = false;
            }
        };

        requestAnimationFrame(animate);
    },

    updateStateFormula() {
        const formula = document.getElementById('stateFormula');
        if (!formula) return;

        const alpha = Math.cos(this.theta / 2);
        const beta = Math.sin(this.theta / 2);

        if (beta < 0.01) {
            formula.textContent = '|0⟩';
        } else if (alpha < 0.01) {
            formula.textContent = '|1⟩';
        } else {
            formula.textContent = `${alpha.toFixed(2)}|0⟩ + ${beta.toFixed(2)}e^(i${(this.phi / Math.PI).toFixed(1)}π)|1⟩`;
        }
    },

    render() {
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;
        const cx = width / 2;
        const cy = height / 2;
        const radius = Math.min(width, height) / 2 - 20;

        // Clear
        this.ctx.fillStyle = '#1a1d24';
        this.ctx.fillRect(0, 0, width, height);

        // Draw sphere outline
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 1;

        // Equator
        this.ctx.beginPath();
        this.ctx.ellipse(cx, cy, radius, radius * 0.3, 0, 0, Math.PI * 2);
        this.ctx.stroke();

        // Meridian
        this.ctx.beginPath();
        this.ctx.ellipse(cx, cy, radius * 0.3, radius, 0, 0, Math.PI * 2);
        this.ctx.stroke();

        // Main circle
        this.ctx.strokeStyle = '#6b7280';
        this.ctx.lineWidth = 1.5;
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        this.ctx.stroke();

        // Axes
        this.ctx.strokeStyle = '#4b5563';
        this.ctx.lineWidth = 1;

        // Z axis
        this.ctx.beginPath();
        this.ctx.moveTo(cx, cy - radius);
        this.ctx.lineTo(cx, cy + radius);
        this.ctx.stroke();

        // X axis (projected)
        this.ctx.beginPath();
        this.ctx.moveTo(cx - radius, cy);
        this.ctx.lineTo(cx + radius, cy);
        this.ctx.stroke();

        // Labels
        this.ctx.fillStyle = '#a0a0a0';
        this.ctx.font = '11px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('|0⟩', cx, cy - radius - 8);
        this.ctx.fillText('|1⟩', cx, cy + radius + 14);

        // State vector
        const x = radius * Math.sin(this.theta) * Math.cos(this.phi);
        const y = radius * Math.sin(this.theta) * Math.sin(this.phi);
        const z = radius * Math.cos(this.theta);

        // Project to 2D
        const projX = cx + x * 0.8;
        const projY = cy - z + y * 0.3;

        // Draw state vector
        const gradient = this.ctx.createLinearGradient(cx, cy, projX, projY);
        gradient.addColorStop(0, '#8b5cf6');
        gradient.addColorStop(1, '#3b82f6');

        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(cx, cy);
        this.ctx.lineTo(projX, projY);
        this.ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(projY - cy, projX - cx);
        this.ctx.fillStyle = '#3b82f6';
        this.ctx.beginPath();
        this.ctx.moveTo(projX, projY);
        this.ctx.lineTo(projX - 8 * Math.cos(angle - 0.4), projY - 8 * Math.sin(angle - 0.4));
        this.ctx.lineTo(projX - 8 * Math.cos(angle + 0.4), projY - 8 * Math.sin(angle + 0.4));
        this.ctx.closePath();
        this.ctx.fill();

        // State point
        this.ctx.fillStyle = '#3b82f6';
        this.ctx.shadowColor = '#3b82f6';
        this.ctx.shadowBlur = 10;
        this.ctx.beginPath();
        this.ctx.arc(projX, projY, 5, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
    }
};

// ===========================
// Probability Display
// ===========================

const ProbabilityDisplay = {
    container: null,

    init() {
        this.container = document.getElementById('probabilityBars');
        this.render([1, 0, 0, 0, 0, 0, 0, 0]); // |000⟩
    },

    update(gates) {
        // Simulate simple state evolution
        const numQubits = CircuitPreview.numQubits;
        const numStates = Math.pow(2, numQubits);

        // Start with |00...0⟩
        let probs = new Array(numStates).fill(0);
        probs[0] = 1;

        // Apply gates (simplified simulation)
        gates.forEach(gate => {
            if (gate.type === 'H') {
                // Hadamard creates superposition
                const newProbs = [...probs];
                const bit = 1 << (numQubits - 1 - gate.qubit);
                for (let i = 0; i < numStates; i++) {
                    const j = i ^ bit;
                    if (i < j) {
                        const avg = (probs[i] + probs[j]) / 2;
                        newProbs[i] = avg;
                        newProbs[j] = avg;
                    }
                }
                probs = newProbs;
            } else if (gate.type === 'X') {
                // Pauli-X flips the bit
                const newProbs = [...probs];
                const bit = 1 << (numQubits - 1 - gate.qubit);
                for (let i = 0; i < numStates; i++) {
                    const j = i ^ bit;
                    if (i < j) {
                        [newProbs[i], newProbs[j]] = [probs[j], probs[i]];
                    }
                }
                probs = newProbs;
            }
        });

        this.render(probs);
    },

    render(probabilities) {
        if (!this.container) return;

        const numQubits = Math.log2(probabilities.length);
        this.container.innerHTML = '';

        // Only show states with non-zero probability
        const threshold = 0.001;

        probabilities.forEach((prob, index) => {
            if (prob < threshold) return;

            const item = document.createElement('div');
            item.className = 'prob-bar-item';

            const binaryLabel = index.toString(2).padStart(numQubits, '0');

            item.innerHTML = `
                <span class="prob-label">|${binaryLabel}⟩</span>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: ${prob * 100}%"></div>
                </div>
                <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
            `;

            this.container.appendChild(item);
        });
    }
};

// ===========================
// Initialize
// ===========================

function initializeQuantumTools() {
    console.log('Initializing Quantum Tools...');

    CircuitPreview.init();
    BlochSphereViz.init();
    ProbabilityDisplay.init();

    console.log('Quantum Tools initialized!');
}

// Export
window.QuantumTools = {
    CircuitPreview,
    BlochSphereViz,
    ProbabilityDisplay,
    init: initializeQuantumTools
};
