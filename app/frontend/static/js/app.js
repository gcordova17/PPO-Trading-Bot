let currentPage = 'training';
let currentTrainingTask = null;
let currentBacktestTask = null;
let pollingInterval = null;
let backTestPollingInterval = null;

const API_URL = '';

document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    
    initTrainingForm();
    initBacktestForm();
    
    loadModels();
});

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const pageName = this.getAttribute('data-page');
            
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            showPage(pageName);
        });
    });
}

function showPage(pageName) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('d-none');
    });
    
    document.getElementById(`${pageName}-page`).classList.remove('d-none');
    
    currentPage = pageName;
    
    if (pageName === 'models') {
        loadModels();
    }
}

function initTrainingForm() {
    const form = document.getElementById('training-form');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = {};
        
        for (const [key, value] of formData.entries()) {
            if (key === 'use_sde') {
                data[key] = true;
            } else if (key === 'learning_rate' || key === 'transaction_cost_pct' || 
                      key === 'reward_scaling' || key === 'gamma' || 
                      key === 'gae_lambda' || key === 'clip_range') {
                data[key] = parseFloat(value);
            } else if (key === 'initial_balance' || key === 'window_size' || 
                      key === 'total_timesteps' || key === 'n_steps' || 
                      key === 'batch_size') {
                data[key] = parseInt(value);
            } else {
                data[key] = value;
            }
        }
        
        if (!data.use_sde) {
            data.use_sde = false;
        }
        
        startTraining(data);
    });
}

function startTraining(data) {
    document.getElementById('training-status').innerHTML = '<p>Starting training...</p>';
    document.getElementById('training-progress').classList.remove('d-none');
    document.getElementById('training-results').classList.add('d-none');
    
    fetch(`${API_URL}/train`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        currentTrainingTask = data.task_id;
        
        document.getElementById('training-status').innerHTML = `
            <p>Training task started with ID: ${data.task_id}</p>
            <p>Status: ${data.status}</p>
            <p>Message: ${data.message}</p>
        `;
        
        startPollingTrainingStatus();
    })
    .catch(error => {
        console.error('Error starting training:', error);
        document.getElementById('training-status').innerHTML = `
            <p class="text-danger">Error starting training: ${error.message}</p>
        `;
    });
}

function startPollingTrainingStatus() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(() => {
        if (currentTrainingTask) {
            checkTrainingStatus(currentTrainingTask);
        }
    }, 5000); // Poll every 5 seconds
}

function checkTrainingStatus(taskId) {
    fetch(`${API_URL}/tasks/${taskId}`)
        .then(response => response.json())
        .then(data => {
            const progressBar = document.querySelector('#training-progress .progress-bar');
            const progressText = document.getElementById('progress-text');
            const progress = Math.round(data.progress * 100);
            
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;
            
            document.getElementById('training-status').innerHTML = `
                <p>Training task ID: ${data.task_id}</p>
                <p>Status: ${data.status}</p>
                <p>Message: ${data.message}</p>
            `;
            
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(pollingInterval);
                
                if (data.status === 'completed' && data.metrics) {
                    showTrainingResults(data);
                }
                
                loadModels();
            }
        })
        .catch(error => {
            console.error('Error checking training status:', error);
        });
}

function showTrainingResults(data) {
    document.getElementById('training-results').classList.remove('d-none');
    
    const portfolioMetrics = document.getElementById('portfolio-metrics');
    portfolioMetrics.innerHTML = '';
    
    for (const [key, value] of Object.entries(data.metrics.portfolio)) {
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const formattedValue = formatMetricValue(key, value);
        const valueClass = getValueClass(key, value);
        
        portfolioMetrics.innerHTML += `
            <li>${formattedKey}: <span class="metric-value ${valueClass}">${formattedValue}</span></li>
        `;
    }
    
    const marketComparison = document.getElementById('market-comparison');
    marketComparison.innerHTML = '';
    
    for (const [key, value] of Object.entries(data.metrics.comparison)) {
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const formattedValue = formatMetricValue(key, value);
        const valueClass = getValueClass(key, value, true);
        
        marketComparison.innerHTML += `
            <li>${formattedKey}: <span class="metric-value ${valueClass}">${formattedValue}</span></li>
        `;
    }
    
    if (data.plots && data.plots.length > 0) {
        const plotUrl = `${API_URL}/plots/${data.plots[0].split('/').pop()}`;
        document.getElementById('performance-chart').src = plotUrl;
    }
}

function initBacktestForm() {
    const form = document.getElementById('backtest-form');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = {};
        
        for (const [key, value] of formData.entries()) {
            if (key === 'initial_balance') {
                data[key] = parseInt(value);
            } else if (key === 'transaction_cost_pct') {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }
        
        startBacktest(data);
    });
}

function startBacktest(data) {
    document.getElementById('backtest-status').innerHTML = '<p>Starting backtest...</p>';
    document.getElementById('backtest-progress').classList.remove('d-none');
    document.getElementById('backtest-results').classList.add('d-none');
    
    fetch(`${API_URL}/backtest`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        currentBacktestTask = data.task_id;
        
        document.getElementById('backtest-status').innerHTML = `
            <p>Backtest task started with ID: ${data.task_id}</p>
            <p>Status: ${data.status}</p>
            <p>Message: ${data.message}</p>
        `;
        
        startPollingBacktestStatus();
    })
    .catch(error => {
        console.error('Error starting backtest:', error);
        document.getElementById('backtest-status').innerHTML = `
            <p class="text-danger">Error starting backtest: ${error.message}</p>
        `;
    });
}

function startPollingBacktestStatus() {
    if (backTestPollingInterval) {
        clearInterval(backTestPollingInterval);
    }
    
    backTestPollingInterval = setInterval(() => {
        if (currentBacktestTask) {
            checkBacktestStatus(currentBacktestTask);
        }
    }, 2000); // Poll every 2 seconds
}

function checkBacktestStatus(taskId) {
    fetch(`${API_URL}/tasks/${taskId}`)
        .then(response => response.json())
        .then(data => {
            const progressBar = document.querySelector('#backtest-progress .progress-bar');
            const progressText = document.getElementById('backtest-progress-text');
            const progress = Math.round(data.progress * 100);
            
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;
            
            document.getElementById('backtest-status').innerHTML = `
                <p>Backtest task ID: ${data.task_id}</p>
                <p>Status: ${data.status}</p>
                <p>Message: ${data.message}</p>
            `;
            
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(backTestPollingInterval);
                
                if (data.status === 'completed' && data.metrics) {
                    showBacktestResults(data);
                }
            }
        })
        .catch(error => {
            console.error('Error checking backtest status:', error);
        });
}

function showBacktestResults(data) {
    document.getElementById('backtest-results').classList.remove('d-none');
    
    const portfolioMetrics = document.getElementById('backtest-portfolio-metrics');
    portfolioMetrics.innerHTML = '';
    
    for (const [key, value] of Object.entries(data.metrics.portfolio)) {
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const formattedValue = formatMetricValue(key, value);
        const valueClass = getValueClass(key, value);
        
        portfolioMetrics.innerHTML += `
            <li>${formattedKey}: <span class="metric-value ${valueClass}">${formattedValue}</span></li>
        `;
    }
    
    const marketComparison = document.getElementById('backtest-market-comparison');
    marketComparison.innerHTML = '';
    
    for (const [key, value] of Object.entries(data.metrics.comparison)) {
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const formattedValue = formatMetricValue(key, value);
        const valueClass = getValueClass(key, value, true);
        
        marketComparison.innerHTML += `
            <li>${formattedKey}: <span class="metric-value ${valueClass}">${formattedValue}</span></li>
        `;
    }
    
    if (data.plots && data.plots.length > 0) {
        const plotUrl = `${API_URL}/plots/${data.plots[0].split('/').pop()}`;
        document.getElementById('backtest-chart').src = plotUrl;
    }
}

function loadModels() {
    fetch(`${API_URL}/models`)
        .then(response => response.json())
        .then(models => {
            const tableBody = document.getElementById('models-table-body');
            tableBody.innerHTML = '';
            
            if (models.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="4" class="text-center">No models available</td>
                    </tr>
                `;
            } else {
                models.forEach(model => {
                    tableBody.innerHTML += `
                        <tr>
                            <td>${model.id}</td>
                            <td>${model.ticker}</td>
                            <td>${formatDate(model.created_at)}</td>
                            <td>
                                <button class="btn btn-sm btn-primary backtest-btn" data-model-id="${model.id}" data-ticker="${model.ticker}">Backtest</button>
                            </td>
                        </tr>
                    `;
                });
            }
            
            const modelSelect = document.getElementById('backtest-model');
            modelSelect.innerHTML = '<option value="">Select a model</option>';
            
            models.forEach(model => {
                modelSelect.innerHTML += `
                    <option value="${model.id}">${model.ticker} (${formatDate(model.created_at)})</option>
                `;
            });
            
            document.querySelectorAll('.backtest-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const modelId = this.getAttribute('data-model-id');
                    const ticker = this.getAttribute('data-ticker');
                    
                    showPage('backtest');
                    
                    document.getElementById('backtest-model').value = modelId;
                    document.getElementById('backtest-ticker').value = ticker;
                });
            });
        })
        .catch(error => {
            console.error('Error loading models:', error);
        });
}

function formatMetricValue(key, value) {
    if (key.includes('return') || key.includes('drawdown')) {
        return `${(value * 100).toFixed(2)}%`;
    } else {
        return value.toFixed(4);
    }
}

function getValueClass(key, value, isComparison = false) {
    if (key.includes('drawdown')) {
        return value > -0.1 ? 'positive' : 'negative';
    } else if (key.includes('volatility')) {
        return value < 0.2 ? 'positive' : 'negative';
    } else if (isComparison) {
        return value > 0 ? 'positive' : 'negative';
    } else {
        return value > 0 ? 'positive' : 'negative';
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}
