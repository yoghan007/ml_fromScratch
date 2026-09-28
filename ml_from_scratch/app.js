/**
 * Machine Learning from Scratch — Browser JavaScript Engine
 * Vectorized Linear & Logistic Regression with Interactive Canvas Visualization
 */

// Global State
let currentMode = 'linear'; // 'linear' or 'logistic'
let points = []; // Array of {x, y, label} in normalized coordinates [-5, 5]
let model = {
    w1: 0,
    w2: 0,
    b: 0,
    costHistory: []
};

// Canvas Contexts
const mainCanvas = document.getElementById('mainCanvas');
const ctx = mainCanvas.getContext('2d');
const lossCanvas = document.getElementById('lossCanvas');
const lossCtx = lossCanvas.getContext('2d');

// Initialization
window.addEventListener('load', () => {
    setupCanvas();
    generateRandomData();
});

function setupCanvas() {
    // Canvas click listeners
    mainCanvas.addEventListener('mousedown', (e) => {
        const rect = mainCanvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        const norm = canvasToNorm(canvasX, canvasY);
        
        let label = 0;
        if (currentMode === 'logistic') {
            // Right click or Shift+click adds Class 1 (Red), Left click adds Class 0 (Blue)
            label = (e.button === 2 || e.shiftKey) ? 1 : 0;
        }

        points.push({ x: norm.x, y: norm.y, label: label });
        updateSampleCount();
        renderCanvas();
    });

    // Prevent context menu on right click
    mainCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

// Coordinate Conversions: Canvas [0..700, 0..480] <-> Math [-5..5, -5..5]
function normToCanvas(x, y) {
    const cx = (x + 5) / 10 * mainCanvas.width;
    const cy = (5 - y) / 10 * mainCanvas.height;
    return { x: cx, y: cy };
}

function canvasToNorm(cx, cy) {
    const x = (cx / mainCanvas.width) * 10 - 5;
    const y = 5 - (cy / mainCanvas.height) * 10;
    return { x: parseFloat(x.toFixed(3)), y: parseFloat(y.toFixed(3)) };
}

// Helper: Update UI text badges
function updateVal(id, val) {
    document.getElementById(id).textContent = parseFloat(val).toFixed(id.includes('lr') || id.includes('l2') || id.includes('thresh') ? 2 : 0);
}

function updateSampleCount() {
    document.getElementById('sampleCount').textContent = `${points.length} Data Points`;
}

// Switch between Linear and Logistic Modes
function switchMode(mode) {
    currentMode = mode;
    document.getElementById('tab-linear').classList.toggle('active', mode === 'linear');
    document.getElementById('tab-logistic').classList.toggle('active', mode === 'logistic');
    
    document.getElementById('canvas-title').textContent = mode === 'linear' 
        ? 'Data Points Canvas (Linear Regression)' 
        : 'Data Points Canvas (Logistic Regression Binary Classification)';

    document.getElementById('canvas-instruction').textContent = mode === 'linear'
        ? 'Click on canvas to add data points'
        : 'Left-Click: Class 0 (Blue) | Shift+Click or Right-Click: Class 1 (Red)';

    document.getElementById('solverGroup').style.display = mode === 'linear' ? 'flex' : 'none';
    document.getElementById('thresholdGroup').style.display = mode === 'logistic' ? 'flex' : 'none';
    document.getElementById('weight2Card').style.display = mode === 'logistic' ? 'flex' : 'none';
    
    document.getElementById('lossLabel').textContent = mode === 'linear' ? 'MSE Cost J' : 'Log Loss J';
    document.getElementById('scoreLabel').textContent = mode === 'linear' ? 'Score (R²)' : 'Accuracy (%)';

    clearCanvas();
    generateRandomData();
}

function clearCanvas() {
    points = [];
    model = { w1: 0, w2: 0, b: 0, costHistory: [] };
    updateSampleCount();
    resetMetrics();
    renderCanvas();
    renderLossChart();
}

function resetMetrics() {
    document.getElementById('metricWeight').textContent = '0.0000';
    document.getElementById('metricWeight2').textContent = '0.0000';
    document.getElementById('metricBias').textContent = '0.0000';
    document.getElementById('metricCost').textContent = '0.0000';
    document.getElementById('metricScore').textContent = '0.0000';
}

function generateRandomData() {
    points = [];
    const n = 40;
    if (currentMode === 'linear') {
        const trueW = (Math.random() * 1.5 + 0.5) * (Math.random() > 0.5 ? 1 : -1);
        const trueB = (Math.random() * 3 - 1.5);
        for (let i = 0; i < n; i++) {
            const x = (Math.random() * 8) - 4;
            const noise = (Math.random() - 0.5) * 2.0;
            const y = trueW * x + trueB + noise;
            points.push({ x: parseFloat(x.toFixed(2)), y: parseFloat(y.toFixed(2)), label: 0 });
        }
    } else {
        // Logistic random binary clusters
        const center0 = { x: -2, y: -1.5 };
        const center1 = { x: 2, y: 1.5 };
        for (let i = 0; i < n / 2; i++) {
            points.push({
                x: center0.x + (Math.random() - 0.5) * 3,
                y: center0.y + (Math.random() - 0.5) * 3,
                label: 0
            });
            points.push({
                x: center1.x + (Math.random() - 0.5) * 3,
                y: center1.y + (Math.random() - 0.5) * 3,
                label: 1
            });
        }
    }
    updateSampleCount();
    renderCanvas();
}

// Math Utility Functions (Sigmoid & Metrics)
function sigmoid(z) {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
}

// Model Training Logic (Pure JavaScript NumPy equivalent)
function trainModel() {
    if (points.length < 2) {
        alert("Please add at least 2 data points on the canvas first!");
        return;
    }

    const lr = parseFloat(document.getElementById('learningRate').value);
    const epochs = parseInt(document.getElementById('maxEpochs').value);
    const l2 = parseFloat(document.getElementById('l2Reg').value);

    model.costHistory = [];

    if (currentMode === 'linear') {
        const solver = document.getElementById('solverSelect').value;
        if (solver === 'normal_equation') {
            trainLinearNormalEquation(l2);
        } else {
            trainLinearGD(lr, epochs, l2);
        }
    } else {
        trainLogisticGD(lr, epochs, l2);
    }

    renderCanvas();
    renderLossChart();
}

// Step 1 Epoch GD
function stepGD() {
    if (points.length < 2) return;
    const lr = parseFloat(document.getElementById('learningRate').value);
    const l2 = parseFloat(document.getElementById('l2Reg').value);

    if (currentMode === 'linear') {
        trainLinearGD(lr, 1, l2, false);
    } else {
        trainLogisticGD(lr, 1, l2, false);
    }

    renderCanvas();
    renderLossChart();
}

// -------------------------------------------------------------
// LINEAR REGRESSION ENGINE
// -------------------------------------------------------------
function trainLinearGD(lr, epochs, l2, resetWeights = true) {
    const m = points.length;
    if (resetWeights || model.w1 === undefined) {
        model.w1 = 0.0;
        model.b = 0.0;
        model.costHistory = [];
    }

    for (let ep = 0; ep < epochs; ep++) {
        let dw = 0;
        let db = 0;
        let mse = 0;

        for (let i = 0; i < m; i++) {
            const yPred = model.w1 * points[i].x + model.b;
            const err = yPred - points[i].y;
            dw += err * points[i].x;
            db += err;
            mse += err * err;
        }

        dw = (dw / m) + (l2 / m) * model.w1;
        db = (db / m);

        model.w1 -= lr * dw;
        model.b -= lr * db;

        const totalCost = (mse / (2 * m)) + (l2 / (2 * m)) * (model.w1 * model.w1);
        model.costHistory.push(totalCost);
    }

    // Compute R2 Score
    let ssRes = 0, ssTot = 0, yMean = points.reduce((acc, p) => acc + p.y, 0) / m;
    points.forEach(p => {
        const yPred = model.w1 * p.x + model.b;
        ssRes += (p.y - yPred) ** 2;
        ssTot += (p.y - yMean) ** 2;
    });
    const r2 = ssTot === 0 ? 1.0 : 1 - (ssRes / ssTot);

    // Update UI Metrics
    document.getElementById('metricWeight').textContent = model.w1.toFixed(4);
    document.getElementById('metricBias').textContent = model.b.toFixed(4);
    document.getElementById('metricCost').textContent = model.costHistory[model.costHistory.length - 1].toFixed(4);
    document.getElementById('metricScore').textContent = r2.toFixed(4);
}

function trainLinearNormalEquation(l2) {
    const m = points.length;
    let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
    
    for (let p of points) {
        sumX += p.x;
        sumY += p.y;
        sumXX += p.x * p.x;
        sumXY += p.x * p.y;
    }

    // Closed form formula for w & b
    const denom = (m * sumXX - sumX * sumX) + l2 * m;
    if (denom === 0) return;

    model.w1 = (m * sumXY - sumX * sumY) / denom;
    model.b = (sumY - model.w1 * sumX) / m;

    let mse = 0;
    points.forEach(p => {
        const err = (model.w1 * p.x + model.b) - p.y;
        mse += err * err;
    });
    const cost = mse / (2 * m);
    model.costHistory = [cost];

    // Compute R2 Score
    let ssRes = 0, ssTot = 0, yMean = sumY / m;
    points.forEach(p => {
        const yPred = model.w1 * p.x + model.b;
        ssRes += (p.y - yPred) ** 2;
        ssTot += (p.y - yMean) ** 2;
    });
    const r2 = ssTot === 0 ? 1.0 : 1 - (ssRes / ssTot);

    document.getElementById('metricWeight').textContent = model.w1.toFixed(4);
    document.getElementById('metricBias').textContent = model.b.toFixed(4);
    document.getElementById('metricCost').textContent = cost.toFixed(4);
    document.getElementById('metricScore').textContent = r2.toFixed(4);
}

// -------------------------------------------------------------
// LOGISTIC REGRESSION ENGINE
// -------------------------------------------------------------
function trainLogisticGD(lr, epochs, l2, resetWeights = true) {
    const m = points.length;
    if (resetWeights || model.w1 === undefined) {
        model.w1 = 0.0;
        model.w2 = 0.0;
        model.b = 0.0;
        model.costHistory = [];
    }

    const thresh = parseFloat(document.getElementById('decisionThreshold').value);

    for (let ep = 0; ep < epochs; ep++) {
        let dw1 = 0, dw2 = 0, db = 0, bce = 0;

        for (let p of points) {
            const z = model.w1 * p.x + model.w2 * p.y + model.b;
            const a = sigmoid(z);
            const err = a - p.label;

            dw1 += err * p.x;
            dw2 += err * p.y;
            db += err;

            const eps = 1e-15;
            const safeA = Math.max(eps, Math.min(1 - eps, a));
            bce += -(p.label * Math.log(safeA) + (1 - p.label) * Math.log(1 - safeA));
        }

        dw1 = (dw1 / m) + (l2 / m) * model.w1;
        dw2 = (dw2 / m) + (l2 / m) * model.w2;
        db = (db / m);

        model.w1 -= lr * dw1;
        model.w2 -= lr * dw2;
        model.b -= lr * db;

        const totalCost = (bce / m) + (l2 / (2 * m)) * (model.w1 * model.w1 + model.w2 * model.w2);
        model.costHistory.push(totalCost);
    }

    // Compute Accuracy
    let correct = 0;
    points.forEach(p => {
        const z = model.w1 * p.x + model.w2 * p.y + model.b;
        const prob = sigmoid(z);
        const pred = prob >= thresh ? 1 : 0;
        if (pred === p.label) correct++;
    });
    const acc = (correct / m) * 100;

    document.getElementById('metricWeight').textContent = model.w1.toFixed(4);
    document.getElementById('metricWeight2').textContent = model.w2.toFixed(4);
    document.getElementById('metricBias').textContent = model.b.toFixed(4);
    document.getElementById('metricCost').textContent = model.costHistory[model.costHistory.length - 1].toFixed(4);
    document.getElementById('metricScore').textContent = acc.toFixed(1) + '%';
}

// -------------------------------------------------------------
// CANVAS RENDERING
// -------------------------------------------------------------
function renderCanvas() {
    const W = mainCanvas.width;
    const H = mainCanvas.height;
    ctx.clearRect(0, 0, W, H);

    // 1. Draw Grid Lines & Axes
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = -4; i <= 4; i++) {
        const p1 = normToCanvas(i, -5);
        const p2 = normToCanvas(i, 5);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        const p3 = normToCanvas(-5, i);
        const p4 = normToCanvas(5, i);
        ctx.beginPath();
        ctx.moveTo(p3.x, p3.y);
        ctx.lineTo(p4.x, p4.y);
        ctx.stroke();
    }

    // Main Axes
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    const origin = normToCanvas(0, 0);
    ctx.beginPath();
    ctx.moveTo(0, origin.y);
    ctx.lineTo(W, origin.y);
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, H);
    ctx.stroke();

    // 2. Draw Decision Boundary or Fitted Line
    if (currentMode === 'linear') {
        if (model.w1 !== undefined && model.w1 !== 0) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 3;
            ctx.beginPath();
            const pStart = normToCanvas(-5, model.w1 * (-5) + model.b);
            const pEnd = normToCanvas(5, model.w1 * 5 + model.b);
            ctx.moveTo(pStart.x, pStart.y);
            ctx.lineTo(pEnd.x, pEnd.y);
            ctx.stroke();
        }
    } else if (currentMode === 'logistic') {
        // Draw decision boundary line: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2
        if (model.w1 !== undefined && (model.w1 !== 0 || model.w2 !== 0)) {
            // Draw Probability Heatmap Background
            const step = 8;
            for (let cx = 0; cx < W; cx += step) {
                for (let cy = 0; cy < H; cy += step) {
                    const norm = canvasToNorm(cx + step/2, cy + step/2);
                    const z = model.w1 * norm.x + model.w2 * norm.y + model.b;
                    const prob = sigmoid(z);
                    
                    // Interpolate between blue (Class 0) and red (Class 1)
                    ctx.fillStyle = `rgba(${Math.round(239 * prob)}, ${Math.round(68 * prob)}, ${Math.round(246 * (1 - prob))}, 0.25)`;
                    ctx.fillRect(cx, cy, step, step);
                }
            }

            // Draw Boundary Line (P = 0.5)
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 3;
            ctx.setLineDash([6, 6]);
            ctx.beginPath();
            if (Math.abs(model.w2) > 0.0001) {
                const y1 = (-model.w1 * (-5) - model.b) / model.w2;
                const y2 = (-model.w1 * 5 - model.b) / model.w2;
                const p1 = normToCanvas(-5, y1);
                const p2 = normToCanvas(5, y2);
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            } else {
                const xVal = -model.b / model.w1;
                const p1 = normToCanvas(xVal, -5);
                const p2 = normToCanvas(xVal, 5);
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // 3. Draw Data Points
    points.forEach(p => {
        const c = normToCanvas(p.x, p.y);
        ctx.beginPath();
        ctx.arc(c.x, c.y, 7, 0, Math.PI * 2);

        if (currentMode === 'linear') {
            ctx.fillStyle = '#3b82f6';
            ctx.strokeStyle = '#60a5fa';
        } else {
            ctx.fillStyle = p.label === 1 ? '#ef4444' : '#3b82f6';
            ctx.strokeStyle = p.label === 1 ? '#fca5a5' : '#93c5fd';
        }

        ctx.fill();
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

// -------------------------------------------------------------
// LOSS CHART RENDERING
// -------------------------------------------------------------
function renderLossChart() {
    const W = lossCanvas.width;
    const H = lossCanvas.height;
    lossCtx.clearRect(0, 0, W, H);

    if (!model.costHistory || model.costHistory.length === 0) return;

    const costs = model.costHistory;
    const maxCost = Math.max(...costs, 0.01);
    const minCost = Math.min(...costs);

    lossCtx.strokeStyle = currentMode === 'linear' ? '#10b981' : '#8b5cf6';
    lossCtx.lineWidth = 2;
    lossCtx.beginPath();

    costs.forEach((c, idx) => {
        const x = (idx / (costs.length - 1 || 1)) * (W - 20) + 10;
        const y = H - 10 - ((c - minCost) / (maxCost - minCost || 1)) * (H - 20);
        if (idx === 0) lossCtx.moveTo(x, y);
        else lossCtx.lineTo(x, y);
    });

    lossCtx.stroke();
}
