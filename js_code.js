// Load chart data
fetch('chart_data.json')
    .then(response => response.json())
    .then(wrapper => {
        // Use the data property which contains the actual chart data
        // The _warning property contains the auto-generation warning
        initializeChart(wrapper.data);
    });

function initializeChart(chartData) {
    // Define chart colors
    const colors = [
        '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
        '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
    ];
    
    // Get canvas context
    const ctx = document.getElementById('token-success-chart').getContext('2d');
    
    // Create model checkboxes
    const modelCheckboxes = document.getElementById('model-checkboxes');
    chartData.models.forEach((model, index) => {
        const color = colors[index % colors.length];
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-model="${model}" checked>
                <span class="checkbox-color" style="background-color: ${color};"></span>
                ${model}
            </label>
        `;
        modelCheckboxes.appendChild(checkbox);
    });
    
    // Create language checkboxes
    const languageCheckboxes = document.getElementById('language-checkboxes');
    chartData.languages.forEach(language => {
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-language="${language}" checked>
                ${language}
            </label>
        `;
        languageCheckboxes.appendChild(checkbox);
    });
    
    // Create chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.buckets.map(bucket => bucket.bucket_location_k + 'k'),
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Prompt Token Length (k)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Success Rate (%)'
                    },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            // Don't show tooltips for confidence interval datasets
                            if (context.dataset.label.includes('CI')) {
                                return null;
                            }
                            
                            const modelName = context.dataset.label;
                            const bucketData = chartData.buckets[context.dataIndex];
                            const modelData = bucketData.models[modelName];
                            
                            // Get basic stats
                            const successRate = context.raw;
                            const successful = modelData.overall.successful;
                            const attempts = modelData.overall.attempts;
                            
                            // Get confidence interval (for selected languages)
                            let ciInfo = '';
                            if (document.getElementById('show-confidence-intervals').checked) {
                                // Find if we have data to calculate CIs
                                let langSuccessful = 0;
                                let langAttempts = 0;
                                
                                const selectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
                                    .map(checkbox => checkbox.getAttribute('data-language'));
                                
                                selectedLanguages.forEach(language => {
                                    if (modelData.languages[language]) {
                                        langSuccessful += modelData.languages[language].successful;
                                        langAttempts += modelData.languages[language].attempts;
                                    }
                                });
                                
                                if (langAttempts > 0) {
                                    const [lower, upper] = wilson_score_interval(langSuccessful, langAttempts);
                                    ciInfo = `
95% CI: ${(lower * 100).toFixed(2)}% - ${(upper * 100).toFixed(2)}%`;
                                }
                            }
                            
                            return [
                                `${modelName}: ${successRate?.toFixed(2) || 'N/A'}% (${successful}/${attempts})`,
                                `Token Range: ${bucketData.bucket_range}${ciInfo}`
                            ];
                        }
                    }
                }
            }
        }
    });
    
    // Function to update chart based on selected models and languages
    function updateChart() {
        // Get selected models and languages
        const selectedModels = Array.from(document.querySelectorAll('input[data-model]:checked'))
            .map(checkbox => checkbox.getAttribute('data-model'));
        
        const selectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
            .map(checkbox => checkbox.getAttribute('data-language'));
        
        // Clear current datasets
        chart.data.datasets = [];
        
        // Create datasets for each selected model
        selectedModels.forEach((model, index) => {
            const color = colors[index % colors.length];
            
            // Calculate data points
            const dataPoints = chartData.buckets.map(bucket => {
                const modelData = bucket.models[model];
                
                // Filter by selected languages
                let successful = 0;
                let attempts = 0;
                
                if (selectedLanguages.length === 0) {
                    // No languages selected, show empty chart (consistent with model selection behavior)
                    return null;
                } else {
                    // Use only selected languages
                    selectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            successful += modelData.languages[language].successful;
                            attempts += modelData.languages[language].attempts;
                        }
                    });
                }
                
                // Calculate success rate
                return attempts > 0 ? (successful / attempts * 100) : null;
            });
            
            // Calculate confidence interval data points if languages are selected
            let lowerBoundPoints = null;
            let upperBoundPoints = null;
            
            if (selectedLanguages.length > 0) {
                lowerBoundPoints = chartData.buckets.map(bucket => {
                    const modelData = bucket.models[model];
                    
                    // Use only selected languages
                    let langSuccessful = 0;
                    let langAttempts = 0;
                    
                    selectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            langSuccessful += modelData.languages[language].successful;
                            langAttempts += modelData.languages[language].attempts;
                        }
                    });
                    
                    if (langAttempts > 0) {
                        // Recalculate Wilson interval for the combined languages
                        const [lower, upper] = wilson_score_interval(langSuccessful, langAttempts);
                        return lower * 100; // Convert to percentage
                    }
                    return null;
                });
                
                upperBoundPoints = chartData.buckets.map(bucket => {
                    const modelData = bucket.models[model];
                    
                    // Use only selected languages
                    let langSuccessful = 0;
                    let langAttempts = 0;
                    
                    selectedLanguages.forEach(language => {
                        if (modelData.languages[language]) {
                            langSuccessful += modelData.languages[language].successful;
                            langAttempts += modelData.languages[language].attempts;
                        }
                    });
                    
                    if (langAttempts > 0) {
                        // Recalculate Wilson interval for the combined languages
                        const [lower, upper] = wilson_score_interval(langSuccessful, langAttempts);
                        return upper * 100; // Convert to percentage
                    }
                    return null;
                });
            }
            
            // Add main dataset
            chart.data.datasets.push({
                label: model,
                data: dataPoints,
                borderColor: color,
                backgroundColor: color + '33',
                fill: false,
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6
            });
            
            // Add confidence interval datasets if enabled
            const showConfidenceIntervals = document.getElementById('show-confidence-intervals').checked;
            
            if (showConfidenceIntervals && selectedLanguages.length > 0) {
                // Add confidence interval area as background
                chart.data.datasets.push({
                    label: `${model} (95% CI)`,
                    data: upperBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: color + '22', // Very transparent version of the line color
                    pointRadius: 0,
                    tension: 0.1,
                    fill: {
                        target: '+1', // Fill to the dataset below (which will be the lower bound)
                        above: color + '22'
                    }
                });
                
                // Add lower bound line (this will be filled to from the dataset above)
                chart.data.datasets.push({
                    label: `${model} (95% CI lower)`,
                    data: lowerBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false,
                    hidden: true // Hide this dataset from the legend
                });
            }
        });
        
        // Update chart
        chart.update();
    }
    
    // Add event listeners to model and language checkboxes
    document.querySelectorAll('input[data-model], input[data-language]').forEach(checkbox => {
        checkbox.addEventListener('change', updateChart);
    });
    
    // Add event listener to confidence interval checkbox
    document.getElementById('show-confidence-intervals').addEventListener('change', updateChart);
    
    // Initial chart update
    updateChart();
}
