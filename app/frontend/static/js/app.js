
const API_URL = 'http://localhost:8005';

document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetPage = this.getAttribute('data-page');
            
            navLinks.forEach(link => link.classList.remove('active'));
            this.classList.add('active');
            
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(`${targetPage}-page`).classList.add('active');
            
            if (targetPage === 'models') {
                loadModels();
            } else if (targetPage === 'comparison') {
                loadModelsForComparison();
            }
        });
    });
    
    document.querySelectorAll('[data-page]').forEach(link => {
        if (!link.classList.contains('nav-link')) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                const targetPage = this.getAttribute('data-page');
                
                navLinks.forEach(link => link.classList.remove('active'));
                document.querySelector(`.nav-link[data-page="${targetPage}"]`).classList.add('active');
                
                pages.forEach(page => page.classList.remove('active'));
                document.getElementById(`${targetPage}-page`).classList.add('active');
                
                if (targetPage === 'models') {
                    loadModels();
                } else if (targetPage === 'comparison') {
                    loadModelsForComparison();
                }
            });
        }
    });
    
    const trainingForm = document.getElementById('training-form');
    trainingForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(trainingForm);
        const params = {};
        
        formData.forEach((value, key) => {
            if (key === 'use_gpu') {
                params[key] = true;
            } else if (key === 'transaction_cost_pct') {
                params[key] = parseFloat(value) / 100; // Convert from percentage to decimal
            } else if (key === 'end_date' && value === '') {
                params[key] = null;
            } else if (['initial_balance', 'window_size', 'timesteps', 'batch_size', 'n_epochs'].includes(key)) {
                params[key] = parseInt(value);
            } else if (['learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm', 'reward_scaling'].includes(key)) {
                params[key] = parseFloat(value);
            } else {
                params[key] = value;
            }
        });
        
        if (!formData.has('use_gpu')) {
            params['use_gpu'] = false;
        }
        
        document.getElementById('training-status').style.display = 'block';
        document.getElementById('status-message').textContent = 'Initializing training...';
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-bar').textContent = '0%';
        document.getElementById('training-result').style.display = 'none';
        
        fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            pollTaskStatus(data.task_id, 'training');
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('status-message').textContent = `Error: ${error.message}`;
        });
    });
    
    const backtestForm = document.getElementById('backtest-form');
    backtestForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(backtestForm);
        const params = {};
        
        formData.forEach((value, key) => {
            if (key === 'transaction_cost') {
                params[key] = parseFloat(value) / 100; // Convert from percentage to decimal
            } else if (key === 'initial_balance') {
                params[key] = parseInt(value);
            } else if ((key === 'start_date' || key === 'end_date') && value === '') {
                params[key] = null;
            } else {
                params[key] = value;
            }
        });
        
        document.getElementById('backtest-status').style.display = 'block';
        document.getElementById('backtest-status-message').textContent = 'Initializing backtest...';
        document.getElementById('backtest-progress-bar').style.width = '0%';
        document.getElementById('backtest-progress-bar').textContent = '0%';
        document.getElementById('backtest-result').style.display = 'none';
        
        fetch(`${API_URL}/backtest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            pollTaskStatus(data.task_id, 'backtest');
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('backtest-status-message').textContent = `Error: ${error.message}`;
        });
    });
    
    const comparisonForm = document.getElementById('comparison-form');
    comparisonForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(comparisonForm);
        const params = {};
        
        formData.forEach((value, key) => {
            if (key === 'end_date' && value === '') {
                params[key] = null;
            } else if (key === 'model_id' && value === '') {
                params[key] = null;
            } else {
                params[key] = value;
            }
        });
        
        document.getElementById('comparison-result').style.display = 'block';
        document.getElementById('comparison-plot').src = '';
        document.getElementById('comparison-plot').alt = 'Loading...';
        
        document.getElementById('ppo-indicator-metrics').querySelector('tbody').innerHTML = '';
        document.getElementById('ppo-rl-metrics').querySelector('tbody').innerHTML = '';
        document.getElementById('comparison-market-metrics').querySelector('tbody').innerHTML = '';
        
        const queryString = Object.keys(params)
            .filter(key => params[key] !== null)
            .map(key => `${key}=${encodeURIComponent(params[key])}`)
            .join('&');
        
        fetch(`${API_URL}/compare?${queryString}`)
            .then(response => response.json())
            .then(data => {
                document.getElementById('comparison-plot').src = `${API_URL}${data.plot_url}`;
                document.getElementById('comparison-plot').alt = 'Comparison Plot';
                
                const ppoIndicatorMetricsTable = document.getElementById('ppo-indicator-metrics').querySelector('tbody');
                ppoIndicatorMetricsTable.innerHTML = '';
                
                Object.entries(data.ppo_indicator_metrics).forEach(([key, value]) => {
                    const row = document.createElement('tr');
                    
                    const keyCell = document.createElement('td');
                    keyCell.textContent = formatMetricName(key);
                    
                    const valueCell = document.createElement('td');
                    valueCell.textContent = formatMetricValue(key, value);
                    
                    row.appendChild(keyCell);
                    row.appendChild(valueCell);
                    
                    ppoIndicatorMetricsTable.appendChild(row);
                });
                
                const marketMetricsTable = document.getElementById('comparison-market-metrics').querySelector('tbody');
                marketMetricsTable.innerHTML = '';
                
                Object.entries(data.market_metrics).forEach(([key, value]) => {
                    const row = document.createElement('tr');
                    
                    const keyCell = document.createElement('td');
                    keyCell.textContent = formatMetricName(key);
                    
                    const valueCell = document.createElement('td');
                    valueCell.textContent = formatMetricValue(key, value);
                    
                    row.appendChild(keyCell);
                    row.appendChild(valueCell);
                    
                    marketMetricsTable.appendChild(row);
                });
                
                const ppoRlMetricsTable = document.getElementById('ppo-rl-metrics').querySelector('tbody');
                ppoRlMetricsTable.innerHTML = '';
                
                if (data.rl_metrics) {
                    Object.entries(data.rl_metrics.portfolio).forEach(([key, value]) => {
                        const row = document.createElement('tr');
                        
                        const keyCell = document.createElement('td');
                        keyCell.textContent = formatMetricName(key);
                        
                        const valueCell = document.createElement('td');
                        valueCell.textContent = formatMetricValue(key, value);
                        
                        row.appendChild(keyCell);
                        row.appendChild(valueCell);
                        
                        ppoRlMetricsTable.appendChild(row);
                    });
                } else {
                    const row = document.createElement('tr');
                    
                    const cell = document.createElement('td');
                    cell.textContent = 'No RL model selected';
                    cell.colSpan = 2;
                    
                    row.appendChild(cell);
                    
                    ppoRlMetricsTable.appendChild(row);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('comparison-plot').alt = `Error: ${error.message}`;
            });
    });
});

function loadModels() {
    const modelsList = document.getElementById('models-list');
    modelsList.innerHTML = '<p>Loading models...</p>';
    
    fetch(`${API_URL}/models`)
        .then(response => response.json())
        .then(models => {
            if (models.length === 0) {
                modelsList.innerHTML = '<p>No models found. <a href="#" data-page="training">Train a model</a> to get started.</p>';
                return;
            }
            
            modelsList.innerHTML = '';
            
            const row = document.createElement('div');
            row.className = 'row';
            
            models.forEach(model => {
                const col = document.createElement('div');
                col.className = 'col-md-4 mb-4';
                
                const card = document.createElement('div');
                card.className = 'card model-card';
                card.setAttribute('data-model-id', model.model_id);
                
                card.innerHTML = `
                    <div class="card-header">
                        <h5>${model.ticker}</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Start Date:</strong> ${model.start_date}</p>
                        <p><strong>End Date:</strong> ${model.end_date || 'Current'}</p>
                        <p><strong>Created:</strong> ${formatDate(model.created_at)}</p>
                        <div class="d-grid gap-2">
                            <button class="btn btn-primary btn-sm view-model-btn">View Details</button>
                        </div>
                    </div>
                `;
                
                col.appendChild(card);
                row.appendChild(col);
                
                card.querySelector('.view-model-btn').addEventListener('click', function() {
                    viewModel(model.model_id);
                });
            });
            
            modelsList.appendChild(row);
            
            loadModelsForComparison();
        })
        .catch(error => {
            console.error('Error:', error);
            modelsList.innerHTML = `<p>Error loading models: ${error.message}</p>`;
        });
}

function loadModelsForComparison() {
    const modelSelect = document.getElementById('comparison-model-id');
    
    const firstOption = modelSelect.options[0];
    modelSelect.innerHTML = '';
    modelSelect.appendChild(firstOption);
    
    fetch(`${API_URL}/models`)
        .then(response => response.json())
        .then(models => {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_id;
                option.textContent = `${model.ticker} (${model.start_date} to ${model.end_date || 'Current'})`;
                modelSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function viewModel(modelId) {
    document.getElementById('model-details').style.display = 'block';
    
    document.getElementById('backtest-model-id').value = modelId;
    
    document.getElementById('backtest-status').style.display = 'none';
    
    fetch(`${API_URL}/models/${modelId}`)
        .then(response => response.json())
        .then(model => {
            document.getElementById('model-id').textContent = model.model_id;
            document.getElementById('model-ticker').textContent = model.ticker;
            document.getElementById('model-start-date').textContent = model.start_date;
            document.getElementById('model-end-date').textContent = model.end_date || 'Current';
            document.getElementById('model-created-at').textContent = formatDate(model.created_at);
            
            if (model.metrics) {
                const portfolioMetricsTable = document.getElementById('model-portfolio-metrics').querySelector('tbody');
                portfolioMetricsTable.innerHTML = '';
                
                Object.entries(model.metrics.portfolio).forEach(([key, value]) => {
                    const row = document.createElement('tr');
                    
                    const keyCell = document.createElement('td');
                    keyCell.textContent = formatMetricName(key);
                    
                    const valueCell = document.createElement('td');
                    valueCell.textContent = formatMetricValue(key, value);
                    
                    row.appendChild(keyCell);
                    row.appendChild(valueCell);
                    
                    portfolioMetricsTable.appendChild(row);
                });
                
                const marketMetricsTable = document.getElementById('model-market-metrics').querySelector('tbody');
                marketMetricsTable.innerHTML = '';
                
                Object.entries(model.metrics.market).forEach(([key, value]) => {
                    const row = document.createElement('tr');
                    
                    const keyCell = document.createElement('td');
                    keyCell.textContent = formatMetricName(key);
                    
                    const valueCell = document.createElement('td');
                    valueCell.textContent = formatMetricValue(key, value);
                    
                    row.appendChild(keyCell);
                    row.appendChild(valueCell);
                    
                    marketMetricsTable.appendChild(row);
                });
                
                document.getElementById('model-performance-plot').src = `${API_URL}/plots/${modelId}/performance.png`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function pollTaskStatus(taskId, taskType) {
    const statusElement = document.getElementById(taskType === 'training' ? 'status-message' : 'backtest-status-message');
    const progressBar = document.getElementById(taskType === 'training' ? 'progress-bar' : 'backtest-progress-bar');
    const resultElement = document.getElementById(taskType === 'training' ? 'training-result' : 'backtest-result');
    
    const interval = setInterval(() => {
        fetch(`${API_URL}/tasks/${taskId}`)
            .then(response => response.json())
            .then(data => {
                statusElement.textContent = data.message;
                
                const progress = Math.round(data.progress * 100);
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${progress}%`;
                
                if (data.status === 'completed') {
                    clearInterval(interval);
                    
                    resultElement.style.display = 'block';
                    
                    if (taskType === 'training') {
                        updateMetricsTables(data.result.metrics, 'portfolio-metrics', 'market-metrics');
                        document.getElementById('performance-plot').src = `${API_URL}${data.result.plot_url}`;
                        
                        loadModels();
                    } else {
                        updateMetricsTables(data.result.metrics, 'backtest-portfolio-metrics', 'backtest-market-metrics');
                        document.getElementById('backtest-performance-plot').src = `${API_URL}${data.result.plot_url}`;
                    }
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    statusElement.textContent = `Error: ${data.message}`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                clearInterval(interval);
                statusElement.textContent = `Error: ${error.message}`;
            });
    }, 1000);
}

function updateMetricsTables(metrics, portfolioTableId, marketTableId) {
    const portfolioMetricsTable = document.getElementById(portfolioTableId).querySelector('tbody');
    portfolioMetricsTable.innerHTML = '';
    
    Object.entries(metrics.portfolio).forEach(([key, value]) => {
        const row = document.createElement('tr');
        
        const keyCell = document.createElement('td');
        keyCell.textContent = formatMetricName(key);
        
        const valueCell = document.createElement('td');
        valueCell.textContent = formatMetricValue(key, value);
        
        row.appendChild(keyCell);
        row.appendChild(valueCell);
        
        portfolioMetricsTable.appendChild(row);
    });
    
    const marketMetricsTable = document.getElementById(marketTableId).querySelector('tbody');
    marketMetricsTable.innerHTML = '';
    
    Object.entries(metrics.market).forEach(([key, value]) => {
        const row = document.createElement('tr');
        
        const keyCell = document.createElement('td');
        keyCell.textContent = formatMetricName(key);
        
        const valueCell = document.createElement('td');
        valueCell.textContent = formatMetricValue(key, value);
        
        row.appendChild(keyCell);
        row.appendChild(valueCell);
        
        marketMetricsTable.appendChild(row);
    });
}

function formatMetricName(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function formatMetricValue(key, value) {
    if (typeof value === 'number') {
        if (key.includes('return') || key.includes('drawdown') || key.includes('volatility')) {
            return `${(value * 100).toFixed(2)}%`;
        } else if (key.includes('ratio') || key.includes('index')) {
            return value.toFixed(4);
        } else {
            return value.toFixed(2);
        }
    }
    return value;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}
