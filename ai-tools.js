/**
 * AI Tools - Sidebar components for AI/ML mode
 */

// ===========================
// Network Visualization
// ===========================

const NetworkViz = {
    canvas: null,
    ctx: null,
    layers: [2, 64, 32, 1],

    init() {
        this.canvas = document.getElementById('networkCanvas');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.render();

        // Setup layer config button
        const applyBtn = document.getElementById('applyLayers');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.updateLayers());
        }
    },

    updateLayers() {
        const input = document.getElementById('layerConfig');
        if (!input) return;

        const values = input.value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v) && v > 0);
        if (values.length > 0) {
            this.layers = [2, ...values, 1]; // Add input and output layers
            this.render();
        }
    },

    render() {
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        this.ctx.fillStyle = '#1a1d24';
        this.ctx.fillRect(0, 0, width, height);

        const padding = 30;
        const layerSpacing = (width - padding * 2) / (this.layers.length - 1);

        // Calculate max neurons per layer for scaling
        const maxNeurons = Math.max(...this.layers);
        const maxVisibleNeurons = 8;

        // Draw connections first
        this.ctx.strokeStyle = 'rgba(59, 130, 246, 0.15)';
        this.ctx.lineWidth = 0.5;

        for (let l = 0; l < this.layers.length - 1; l++) {
            const x1 = padding + l * layerSpacing;
            const x2 = padding + (l + 1) * layerSpacing;

            const n1 = Math.min(this.layers[l], maxVisibleNeurons);
            const n2 = Math.min(this.layers[l + 1], maxVisibleNeurons);

            const spacing1 = (height - padding * 2) / (n1 + 1);
            const spacing2 = (height - padding * 2) / (n2 + 1);

            for (let i = 0; i < n1; i++) {
                const y1 = padding + (i + 1) * spacing1;
                for (let j = 0; j < n2; j++) {
                    const y2 = padding + (j + 1) * spacing2;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x1, y1);
                    this.ctx.lineTo(x2, y2);
                    this.ctx.stroke();
                }
            }
        }

        // Draw neurons
        for (let l = 0; l < this.layers.length; l++) {
            const x = padding + l * layerSpacing;
            const n = Math.min(this.layers[l], maxVisibleNeurons);
            const spacing = (height - padding * 2) / (n + 1);

            for (let i = 0; i < n; i++) {
                const y = padding + (i + 1) * spacing;

                // Gradient fill
                const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 8);

                if (l === 0) {
                    gradient.addColorStop(0, '#3b82f6');
                    gradient.addColorStop(1, '#1e40af');
                } else if (l === this.layers.length - 1) {
                    gradient.addColorStop(0, '#10b981');
                    gradient.addColorStop(1, '#047857');
                } else {
                    gradient.addColorStop(0, '#8b5cf6');
                    gradient.addColorStop(1, '#5b21b6');
                }

                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 6, 0, Math.PI * 2);
                this.ctx.fill();

                // Glow effect
                this.ctx.shadowColor = l === 0 ? '#3b82f6' : l === this.layers.length - 1 ? '#10b981' : '#8b5cf6';
                this.ctx.shadowBlur = 8;
                this.ctx.fill();
                this.ctx.shadowBlur = 0;
            }

            // Show "..." if truncated
            if (this.layers[l] > maxVisibleNeurons) {
                this.ctx.fillStyle = '#6b7280';
                this.ctx.font = '10px Inter';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(`+${this.layers[l] - maxVisibleNeurons}`, x, height - 10);
            }

            // Layer label
            this.ctx.fillStyle = '#a0a0a0';
            this.ctx.font = '9px Inter';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(this.layers[l], x, 15);
        }
    }
};

// ===========================
// Loss Chart
// ===========================

const LossChart = {
    canvas: null,
    ctx: null,
    losses: [],
    maxPoints: 100,

    init() {
        this.canvas = document.getElementById('lossChart');
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.render();
    },

    addLoss(loss) {
        this.losses.push(loss);
        if (this.losses.length > this.maxPoints) {
            this.losses.shift();
        }
        this.render();
    },

    reset() {
        this.losses = [];
        this.render();
    },

    render() {
        if (!this.ctx) return;

        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear
        this.ctx.fillStyle = '#1a1d24';
        this.ctx.fillRect(0, 0, width, height);

        if (this.losses.length < 2) {
            // Show placeholder
            this.ctx.fillStyle = '#6b7280';
            this.ctx.font = '12px Inter';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Training metrics will appear here', width / 2, height / 2);
            return;
        }

        const padding = 20;
        const plotWidth = width - padding * 2;
        const plotHeight = height - padding * 2;

        const maxLoss = Math.max(...this.losses);
        const minLoss = Math.min(...this.losses);
        const range = maxLoss - minLoss || 1;

        // Draw grid
        this.ctx.strokeStyle = '#374151';
        this.ctx.lineWidth = 0.5;

        for (let i = 0; i <= 4; i++) {
            const y = padding + (i / 4) * plotHeight;
            this.ctx.beginPath();
            this.ctx.moveTo(padding, y);
            this.ctx.lineTo(width - padding, y);
            this.ctx.stroke();
        }

        // Draw loss line
        this.ctx.strokeStyle = '#3b82f6';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        for (let i = 0; i < this.losses.length; i++) {
            const x = padding + (i / (this.losses.length - 1)) * plotWidth;
            const y = padding + ((maxLoss - this.losses[i]) / range) * plotHeight;

            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }

        this.ctx.stroke();

        // Gradient fill under curve
        const gradient = this.ctx.createLinearGradient(0, padding, 0, height - padding);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

        this.ctx.lineTo(width - padding, height - padding);
        this.ctx.lineTo(padding, height - padding);
        this.ctx.closePath();
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
    }
};

// ===========================
// Training Metrics
// ===========================

const TrainingMetrics = {
    update(loss, accuracy) {
        const lossEl = document.getElementById('lossValue');
        const accEl = document.getElementById('accuracyValue');

        if (lossEl) {
            lossEl.textContent = typeof loss === 'number' ? loss.toFixed(4) : loss;
        }

        if (accEl) {
            accEl.textContent = typeof accuracy === 'number' ? `${(accuracy * 100).toFixed(1)}%` : accuracy;
        }
    },

    reset() {
        this.update('-', '-');
    }
};

// ===========================
// Initialize
// ===========================

function initializeAITools() {
    console.log('Initializing AI Tools...');

    NetworkViz.init();
    LossChart.init();
    TrainingMetrics.reset();

    // Demo: Simulate some training data
    simulateDemoTraining();

    console.log('AI Tools initialized!');
}

function simulateDemoTraining() {
    // Simulate loss decrease for demo
    let epoch = 0;
    const maxEpochs = 50;
    let loss = 1.0;

    const interval = setInterval(() => {
        loss *= 0.92 + Math.random() * 0.05;
        LossChart.addLoss(loss);

        const accuracy = 1 - loss * 0.5;
        TrainingMetrics.update(loss, accuracy);

        epoch++;
        if (epoch >= maxEpochs) {
            clearInterval(interval);
        }
    }, 100);
}

// Export
window.AITools = {
    NetworkViz,
    LossChart,
    TrainingMetrics,
    init: initializeAITools
};
