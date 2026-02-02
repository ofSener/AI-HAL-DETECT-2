// LOGIC-HALT

let currentResponses = [];
let currentQuestion = '';
let currentSummary = {};
let groundTruthLabels = {}; // Manuel etiketler

function setQuestion(question) {
    document.getElementById('question-input').value = question;
}

// Tab switching
function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    // Show selected tab
    if (tab === 'generate') {
        document.getElementById('tab-generate').classList.add('active');
        document.getElementById('results-section').classList.remove('hidden-by-tab');
        document.getElementById('qa-result-section').classList.add('hidden');
    } else if (tab === 'check') {
        document.getElementById('tab-check').classList.add('active');
        document.getElementById('results-section').classList.add('hidden-by-tab');
    }
}

// Q&A Check functionality
async function checkQA() {
    const question = document.getElementById('qa-question-input').value.trim();
    const answer = document.getElementById('qa-answer-input').value.trim();

    if (!question) {
        alert('Lutfen bir soru girin');
        return;
    }
    if (!answer) {
        alert('Lutfen kontrol edilecek cevabi girin');
        return;
    }

    document.getElementById('loading-section').classList.remove('hidden');
    document.getElementById('qa-result-section').classList.add('hidden');
    document.getElementById('check-btn').disabled = true;

    try {
        const response = await fetch('/analyze_qa', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer })
        });

        const data = await response.json();

        if (data.error) throw new Error(data.error);

        displayQAResult(data);

    } catch (error) {
        alert('Hata: ' + error.message);
    } finally {
        document.getElementById('loading-section').classList.add('hidden');
        document.getElementById('check-btn').disabled = false;
    }
}

function displayQAResult(data) {
    const result = data.result;
    const metrics = data.metrics;

    // Verdict
    const verdictEl = document.getElementById('qa-verdict');
    if (result.is_hallucination) {
        verdictEl.className = 'qa-verdict hallucination';
        verdictEl.innerHTML = `
            <span class="verdict-icon">⚠️</span>
            <span class="verdict-text">HALUSINASYON TESPIT EDILDI</span>
            <p class="verdict-detail">${result.verdict}</p>
        `;
    } else {
        verdictEl.className = 'qa-verdict safe';
        verdictEl.innerHTML = `
            <span class="verdict-icon">✓</span>
            <span class="verdict-text">GUVENILIR CEVAP</span>
            <p class="verdict-detail">${result.verdict}</p>
        `;
    }

    // Metrics
    document.getElementById('qa-risk').textContent = (result.risk_score * 100).toFixed(2) + '%';
    document.getElementById('qa-consistency').textContent = (metrics.consistency * 100).toFixed(2) + '%';
    document.getElementById('qa-entropy').textContent = (metrics.entropy * 100).toFixed(2) + '%';
    document.getElementById('qa-ncd').textContent = (metrics.ncd * 100).toFixed(2) + '%';

    // Explanation
    document.getElementById('qa-explanation').textContent = result.explanation;

    // Variants
    const variantsEl = document.getElementById('qa-variants-list');
    variantsEl.innerHTML = '';
    data.variants.forEach(v => {
        const variantDiv = document.createElement('div');
        variantDiv.className = 'qa-variant-item';
        variantDiv.innerHTML = `
            <span class="variant-header">Varyant #${v.id} (T=${v.temperature})</span>
            <p class="variant-text">${escapeHtml(v.text_preview)}</p>
        `;
        variantsEl.appendChild(variantDiv);
    });

    document.getElementById('qa-result-section').classList.remove('hidden');
}

async function analyzeQuestion() {
    const question = document.getElementById('question-input').value.trim();
    const numResponses = parseInt(document.getElementById('num-responses').value);

    if (!question) {
        alert('Lutfen bir soru girin');
        return;
    }

    document.getElementById('loading-section').classList.remove('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('analyze-btn').disabled = true;

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, num_responses: numResponses })
        });

        const data = await response.json();

        if (data.error) throw new Error(data.error);

        currentResponses = data.responses;
        currentQuestion = question;
        currentSummary = data.summary;

        displayResults(data);

    } catch (error) {
        alert('Hata: ' + error.message);
    } finally {
        document.getElementById('loading-section').classList.add('hidden');
        document.getElementById('analyze-btn').disabled = false;
    }
}

function displayResults(data) {
    document.getElementById('total-responses').textContent = data.summary.total_responses;
    document.getElementById('safe-count').textContent = data.summary.safe_count || 0;
    document.getElementById('hallucination-count').textContent = data.summary.hallucinations_detected;
    document.getElementById('avg-risk').textContent = (data.summary.average_risk * 100).toFixed(2) + '%';

    const container = document.getElementById('responses-container');
    container.innerHTML = '';

    // Reset ground truth labels
    groundTruthLabels = {};

    data.responses.forEach(response => {
        container.appendChild(createResponseCard(response));
        // Default: sistem tahmini
        groundTruthLabels[response.id] = response.is_hallucination;
    });

    document.getElementById('results-section').classList.remove('hidden');

    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector('.filter-btn').classList.add('active');
}

function createResponseCard(response) {
    // 2-Category system: safe, hallucination
    const category = response.category || (response.is_hallucination ? 'hallucination' : 'safe');
    const statusClass = category;
    const statusTexts = {
        'safe': 'Guvenli',
        'hallucination': 'Halusinasyon'
    };
    const statusText = statusTexts[category] || 'Bilinmiyor';

    const riskPercent = (response.risk_score * 100).toFixed(2);
    const riskClass = response.risk_score < 0.4 ? 'low' : response.risk_score < 0.7 ? 'medium' : 'high';

    const card = document.createElement('div');
    card.className = `response-card ${statusClass}`;
    card.dataset.status = statusClass;
    card.dataset.id = response.id;

    const isHallucination = category === 'hallucination';
    card.innerHTML = `
        <div class="response-header">
            <div class="response-id">
                <span class="response-number">#${response.id}</span>
                <span class="response-temp">T: ${response.temperature}</span>
            </div>
            <div class="response-status">
                <span>${statusText}</span>
            </div>
        </div>
        <div class="response-text">${escapeHtml(response.text)}</div>
        <div class="response-metrics">
            <div class="metric risk-bar-container">
                <span class="metric-label">Risk</span>
                <div class="risk-bar">
                    <div class="risk-bar-fill ${riskClass}" style="width: ${riskPercent}%"></div>
                </div>
                <span class="metric-value">${riskPercent}%</span>
            </div>
            ${response.metrics.consistency !== undefined ? `
            <div class="metric">
                <span class="metric-label">Tutarlilik</span>
                <span class="metric-value">${(response.metrics.consistency * 100).toFixed(2)}%</span>
            </div>` : ''}
            ${response.metrics.entropy !== undefined ? `
            <div class="metric">
                <span class="metric-label">Entropi</span>
                <span class="metric-value">${(response.metrics.entropy * 100).toFixed(2)}% (${response.metrics.entropy_raw || 0} bits)</span>
            </div>` : ''}
            ${response.metrics.ncd !== undefined ? `
            <div class="metric">
                <span class="metric-label">NCD</span>
                <span class="metric-value">${(response.metrics.ncd * 100).toFixed(2)}%</span>
            </div>` : ''}
        </div>
        <div class="label-section">
            <span class="label-title">Gercek Durum:</span>
            <div class="label-buttons">
                <button class="label-btn ${isHallucination ? 'active' : ''}" data-id="${response.id}" data-label="true" onclick="setLabel(${response.id}, true)">Halusinasyon</button>
                <button class="label-btn ${!isHallucination ? 'active' : ''}" data-id="${response.id}" data-label="false" onclick="setLabel(${response.id}, false)">Dogru</button>
            </div>
        </div>
    `;

    return card;
}

// Manuel etiket ayarlama
function setLabel(responseId, isHallucination) {
    groundTruthLabels[responseId] = isHallucination;

    // Update button states
    const card = document.querySelector(`.response-card[data-id="${responseId}"]`);
    if (card) {
        const buttons = card.querySelectorAll('.label-btn');
        buttons.forEach(btn => {
            const btnLabel = btn.dataset.label === 'true';
            btn.classList.toggle('active', btnLabel === isHallucination);
        });
    }
}

function filterResponses(filter) {
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    document.querySelectorAll('.response-card').forEach(card => {
        if (filter === 'all') {
            card.style.display = 'block';
        } else {
            card.style.display = card.dataset.status === filter ? 'block' : 'none';
        }
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function exportResults() {
    if (!currentResponses.length) {
        alert('Sonuc yok');
        return;
    }

    let output = `LOGIC-HALT Sonuclari\n\n`;
    output += `Soru: ${currentQuestion}\n`;
    output += `Tarih: ${new Date().toLocaleString()}\n\n`;
    output += `Toplam: ${currentSummary.total_responses}\n`;
    output += `Guvenli: ${currentSummary.safe_count || 0}\n`;
    output += `Halusinasyon: ${currentSummary.hallucinations_detected}\n`;
    output += `Ortalama Risk: ${(currentSummary.average_risk * 100).toFixed(2)}%\n\n`;

    // Tablo basligi
    output += `| # | Karar | Risk | C | E | E(bits) | NCD |\n`;
    output += `|---|-------|------|------|------|---------|-----|\n`;

    currentResponses.forEach(r => {
        const karar = r.is_hallucination ? 'HALUSINASYON' : 'GUVENLI';
        const risk = (r.risk_score * 100).toFixed(2) + '%';
        const c = r.metrics.consistency !== undefined ? (r.metrics.consistency * 100).toFixed(2) + '%' : '-';
        const e = r.metrics.entropy !== undefined ? (r.metrics.entropy * 100).toFixed(2) + '%' : '-';
        const eBits = r.metrics.entropy_raw !== undefined ? r.metrics.entropy_raw.toFixed(2) : '-';
        const ncd = r.metrics.ncd !== undefined ? (r.metrics.ncd * 100).toFixed(2) + '%' : '-';

        output += `| ${r.id} | ${karar} | ${risk} | ${c} | ${e} | ${eBits} | ${ncd} |\n`;
    });

    output += `\n--- Detayli Cevaplar ---\n\n`;

    currentResponses.forEach(r => {
        output += `[${r.id}] ${r.is_hallucination ? 'HALUSINASYON' : 'GUVENLI'}\n`;
        output += `Risk: ${(r.risk_score * 100).toFixed(2)}%`;
        if (r.metrics.consistency !== undefined) output += ` | C: ${(r.metrics.consistency * 100).toFixed(2)}%`;
        if (r.metrics.entropy !== undefined) output += ` | E: ${(r.metrics.entropy * 100).toFixed(2)}%`;
        if (r.metrics.ncd !== undefined) output += ` | NCD: ${(r.metrics.ncd * 100).toFixed(2)}%`;
        output += `\n${r.text}\n\n`;
    });

    navigator.clipboard.writeText(output).then(() => {
        const status = document.getElementById('export-status');
        status.textContent = 'Kopyalandi';
        status.classList.add('show');
        setTimeout(() => status.classList.remove('show'), 2000);
    });
}

// Etiketleri kaydet
async function saveLabels() {
    if (!currentResponses.length) {
        alert('Sonuc yok');
        return;
    }

    const data = {
        question: currentQuestion,
        timestamp: new Date().toISOString(),
        summary: currentSummary,
        responses: currentResponses.map(r => ({
            id: r.id,
            text: r.text,
            temperature: r.temperature,
            predicted: r.is_hallucination,
            ground_truth: groundTruthLabels[r.id],
            risk_score: r.risk_score,
            metrics: r.metrics
        }))
    };

    try {
        const response = await fetch('/save_labels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            const status = document.getElementById('export-status');
            status.textContent = `Kaydedildi! (${result.total_records} kayit)`;
            status.classList.add('show');
            setTimeout(() => status.classList.remove('show'), 3000);
        } else {
            alert('Kaydetme hatasi: ' + result.error);
        }
    } catch (error) {
        alert('Hata: ' + error.message);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const textarea = document.getElementById('question-input');
    if (textarea) {
        textarea.addEventListener('keydown', e => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                analyzeQuestion();
            }
        });
    }
});
