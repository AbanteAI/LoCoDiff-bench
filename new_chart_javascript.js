// Load chart data
fetch('chart_data.json')
    .then(response => response.json())
    .then(wrapper => {
        // Use the data property which contains the actual chart data
        // The _warning property contains the auto-generation warning
        initializeChart(wrapper.data);
    });

// JavaScript implementation of Wilson score interval for confidence intervals
function wilson_score_interval(successful, attempts, z = 1.96) {
    if (attempts === 0) return [0.0, 0.0];
    
    // Observed proportion
    const p_hat = successful / attempts;
    
    // Wilson score calculation
    const denominator = 1 + (z * z / attempts);
    const center = (p_hat + (z * z / (2 * attempts))) / denominator;
    const interval = z * Math.sqrt((p_hat * (1 - p_hat) + (z * z / (4 * attempts))) / attempts) / denominator;
    
    const lower_bound = Math.max(0.0, center - interval);
    const upper_bound = Math.min(1.0, center + interval);
    
    return [lower_bound, upper_bound];
}

function initializeChart(chartData) {
    // Define chart colors
    const colors = [
        '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
        '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
    ];
    
    // Create global variables for chart state that need to be accessed by callbacks
    let currentSelectedLanguages = [];
    let currentSelectedModels = [];
    let currentBucketCount = chartData.default_bucket_count || 5;
    let buckets = []; // Will be calculated dynamically
    
    // Get canvas context
    const ctx = document.getElementById('token-success-chart').getContext('2d');
    
    // Create model checkboxes
    const modelCheckboxes = document.getElementById('model-checkboxes');
    chartData.models.forEach((model) => {
        // Use display name if available, otherwise use original model name
        const displayName = chartData.model_display_names && chartData.model_display_names[model] 
            ? chartData.model_display_names[model] 
            : model;
        
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox-item';
        checkbox.innerHTML = `
            <label>
                <input type="checkbox" data-model="${model}" checked>
                ${displayName}
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
    
    // Setup bucket count slider
    const bucketCountSlider = document.getElementById('bucket-count');
    const bucketCountDisplay = document.getElementById('bucket-count-display');
    
    bucketCountSlider.value = currentBucketCount;
    bucketCountDisplay.textContent = currentBucketCount;
    
    bucketCountSlider.addEventListener('input', function() {
        currentBucketCount = parseInt(this.value);
        bucketCountDisplay.textContent = currentBucketCount;
        calculateBuckets();
        updateChart();
    });
    
    // Create chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: chartData.max_tokens_k,
                    ticks: {
                        callback: function(value) {
                            return value + 'k';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Prompt Token Length'
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
                title: {
                    display: true,
                    text: 'LoCoDiff: Natural Long Context Code Bench',
                    font: {
                        size: 18
                    },
                    padding: {
                        top: 10,
                        bottom: 30
                    }
                },
                legend: {
                    labels: {
                        filter: (legendItem, data) => {
                            const dataset = data.datasets[legendItem.datasetIndex];
                            return dataset && dataset.display !== false;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (!context || !context.dataset || !context.dataset.label) {
                                return null;
                            }
                            
                            if (context.dataset.label.includes('CI')) {
                                return null;
                            }
                            
                            try {
                                const displayName = context.dataset.label;
                                const originalModel = context.dataset.originalModel || displayName;
                                const xValue = context.raw.x; // Token count in thousands
                                
                                // Find the bucket with matching token count
                                const bucketData = buckets.find(bucket => 
                                    Math.abs((bucket.avgTokens / 1000) - xValue) < 0.01
                                );
                                
                                if (!bucketData) {
                                    return [`${displayName}`];
                                }
                                
                                const modelStats = bucketData.modelStats[originalModel];
                                if (!modelStats) {
                                    return [`${displayName}: No data available`];
                                }
                                
                                const successRate = context.raw.y;
                                const successful = modelStats.successful;
                                const attempts = modelStats.attempts;
                                const caseCount = bucketData.caseCount;
                                
                                // Get confidence interval
                                let ciInfo = '';
                                const ciElement = document.getElementById('show-confidence-intervals');
                                if (ciElement && ciElement.checked && attempts > 0) {
                                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                                    ciInfo = `\n95% CI: ${(lower * 100).toFixed(2)}% - ${(upper * 100).toFixed(2)}%`;
                                }
                                
                                // Calculate untested cases
                                const untestedCases = caseCount - attempts;
                                let untestedInfo = '';
                                if (untestedCases > 0) {
                                    untestedInfo = `\nModel did not return result for ${untestedCases} case${untestedCases > 1 ? 's' : ''} in this bucket`;
                                }
                                
                                return [
                                    `${displayName}: ${successRate.toFixed(2)}% (${successful}/${caseCount})`,
                                    `Token Range: ${bucketData.minTokens/1000}k-${bucketData.maxTokens/1000}k (avg: ${(bucketData.avgTokens/1000).toFixed(1)}k)
Cases: ${bucketData.caseCount}${untestedInfo}${ciInfo}`
                                ];
                            } catch (error) {
                                console.error('Error in tooltip callback:', error);
                                return ['Error displaying tooltip'];
                            }
                        }
                    }
                }
            }
        }
    });
    
    // Function to filter cases based on selected languages
    function getFilteredCases() {
        if (currentSelectedLanguages.length === 0) {
            return [];
        }
        
        return chartData.cases.filter(caseData => 
            currentSelectedLanguages.includes(caseData.language)
        );
    }
    
    // Function to calculate buckets based on filtered cases
    function calculateBuckets() {
        const filteredCases = getFilteredCases();
        if (filteredCases.length === 0) {
            buckets = [];
            return;
        }
        
        // Sort cases by token count (should already be sorted, but just to be safe)
        filteredCases.sort((a, b) => a.token_count - b.token_count);
        
        // Calculate cases per bucket
        const casesPerBucket = Math.floor(filteredCases.length / currentBucketCount);
        const extraCases = filteredCases.length % currentBucketCount;
        
        // Create new buckets array
        buckets = [];
        
        let caseIndex = 0;
        for (let i = 0; i < currentBucketCount; i++) {
            // Calculate how many cases go in this bucket
            // Last bucket gets the extra cases if count isn't evenly divisible
            const bucketSize = (i === currentBucketCount - 1) 
                ? casesPerBucket + extraCases 
                : casesPerBucket;
            
            if (bucketSize === 0) continue;
            
            // Get cases for this bucket
            const bucketCases = filteredCases.slice(caseIndex, caseIndex + bucketSize);
            caseIndex += bucketSize;
            
            // Calculate token stats
            const minTokens = bucketCases[0].token_count;
            const maxTokens = bucketCases[bucketCases.length - 1].token_count;
            const avgTokens = bucketCases.reduce((sum, c) => sum + c.token_count, 0) / bucketCases.length;
            
            // Initialize bucket data
            const bucket = {
                minTokens,
                maxTokens,
                avgTokens,
                caseCount: bucketCases.length,
                cases: bucketCases,
                modelStats: {}
            };
            
            // Calculate model statistics for this bucket
            chartData.models.forEach(model => {
                const modelStats = {
                    successful: 0,
                    attempts: 0
                };
                
                bucketCases.forEach(caseData => {
                    const result = caseData.results[model];
                    if (result.attempted) {
                        modelStats.attempts++;
                        if (result.success) {
                            modelStats.successful++;
                        }
                    }
                });
                
                bucket.modelStats[model] = modelStats;
            });
            
            buckets.push(bucket);
        }
    }
    
    // Function to update chart based on selected models, languages, and bucket count
    function updateChart() {
        // Get selected models and languages
        currentSelectedModels = Array.from(document.querySelectorAll('input[data-model]:checked'))
            .map(checkbox => checkbox.getAttribute('data-model'));
        
        currentSelectedLanguages = Array.from(document.querySelectorAll('input[data-language]:checked'))
            .map(checkbox => checkbox.getAttribute('data-language'));
        
        // Recalculate buckets based on selected languages and bucket count
        calculateBuckets();
        
        // Clear current datasets
        chart.data.datasets = [];
        
        // If no buckets (e.g., no languages selected), don't update further
        if (buckets.length === 0) {
            chart.update();
            return;
        }
        
        // Update x-axis scale based on actual data
        const minToken = buckets[0].minTokens / 1000;
        const maxToken = buckets[buckets.length - 1].maxTokens / 1000;
        
        chart.options.scales.x.min = Math.floor(minToken);
        chart.options.scales.x.max = Math.ceil(maxToken);
        
        // Create datasets for each selected model
        currentSelectedModels.forEach((model, index) => {
            const color = colors[index % colors.length];
            
            // Calculate data points for this model
            const dataPoints = buckets.map(bucket => {
                const modelStats = bucket.modelStats[model];
                const successful = modelStats.successful;
                const caseCount = bucket.caseCount;
                
                // Calculate success rate
                const successRate = (caseCount > 0) ? (successful / caseCount) * 100 : 0;
                
                return {
                    x: bucket.avgTokens / 1000, // Convert to k
                    y: successRate
                };
            });
            
            // Calculate confidence interval data points
            let lowerBoundPoints = null;
            let upperBoundPoints = null;
            
            if (document.getElementById('show-confidence-intervals').checked) {
                lowerBoundPoints = buckets.map(bucket => {
                    const modelStats = bucket.modelStats[model];
                    const successful = modelStats.successful;
                    const caseCount = bucket.caseCount;
                    
                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                    
                    return {
                        x: bucket.avgTokens / 1000,
                        y: lower * 100
                    };
                });
                
                upperBoundPoints = buckets.map(bucket => {
                    const modelStats = bucket.modelStats[model];
                    const successful = modelStats.successful;
                    const caseCount = bucket.caseCount;
                    
                    const [lower, upper] = wilson_score_interval(successful, caseCount);
                    
                    return {
                        x: bucket.avgTokens / 1000,
                        y: upper * 100
                    };
                });
            }
            
            // Get display name if available
            const displayName = chartData.model_display_names && chartData.model_display_names[model] 
                ? chartData.model_display_names[model] 
                : model;
                
            // Add main dataset
            chart.data.datasets.push({
                label: displayName,
                originalModel: model,
                data: dataPoints,
                borderColor: color,
                backgroundColor: color + '33',
                fill: false,
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6,
                parsing: false
            });
            
            // Add confidence interval datasets if enabled
            if (document.getElementById('show-confidence-intervals').checked) {
                // Add lower bound line (needed for reference by the area dataset)
                const lowerBoundIndex = chart.data.datasets.length;
                chart.data.datasets.push({
                    label: `${displayName} (95% CI lower)`,
                    data: lowerBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false,
                    parsing: false,
                    showLine: false,
                    display: false
                });
                
                // Add confidence interval area
                chart.data.datasets.push({
                    label: `${displayName} (95% CI)`,
                    data: upperBoundPoints,
                    borderColor: 'transparent',
                    backgroundColor: color + '22',
                    pointRadius: 0,
                    tension: 0.1,
                    parsing: false,
                    fill: lowerBoundIndex,
                    display: false
                });
            }
        });
        
        // Update chart
        chart.update();
    }
    
    // Add event listeners
    document.querySelectorAll('input[data-model], input[data-language]').forEach(checkbox => {
        checkbox.addEventListener('change', updateChart);
    });
    
    document.getElementById('show-confidence-intervals').addEventListener('change', updateChart);
    
    // Calculate initial buckets and update chart
    calculateBuckets();
    updateChart();
}
