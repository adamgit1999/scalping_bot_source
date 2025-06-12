// Chart configuration
const chartConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    scrollZoom: true,
    dragmode: 'zoom',
    showTips: true,
    showLink: false,
    plotGlPixelRatio: 2
};

// Initialize main trading chart
let mainChart = null;
let indicators = {};

// Chart data structure
let chartData = {
    candlesticks: [],
    volume: [],
    indicators: {}
};

// Initialize chart
function initChart(containerId, initialData = null) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Create initial layout
    const layout = {
        title: 'Trading Chart',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            color: '#e0e0e0'
        },
        xaxis: {
            rangeslider: { visible: false },
            type: 'date',
            gridcolor: '#2d2d2d',
            zerolinecolor: '#2d2d2d'
        },
        yaxis: {
            domain: [0.2, 1],
            gridcolor: '#2d2d2d',
            zerolinecolor: '#2d2d2d'
        },
        yaxis2: {
            domain: [0, 0.2],
            gridcolor: '#2d2d2d',
            zerolinecolor: '#2d2d2d'
        },
        margin: { t: 30, r: 30, b: 30, l: 30 },
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            orientation: 'h',
            bgcolor: 'rgba(0,0,0,0)',
            bordercolor: 'rgba(0,0,0,0)'
        }
    };

    // Create initial data
    const data = initialData || [
        {
            type: 'candlestick',
            x: [],
            open: [],
            high: [],
            low: [],
            close: [],
            name: 'Price',
            yaxis: 'y'
        },
        {
            type: 'bar',
            x: [],
            y: [],
            name: 'Volume',
            yaxis: 'y2',
            marker: {
                color: 'rgba(0,0,0,0.3)'
            }
        }
    ];

    // Create the chart
    mainChart = Plotly.newPlot(container, data, layout, chartConfig);

    // Add event listeners
    container.on('plotly_relayout', handleChartRelayout);
    container.on('plotly_click', handleChartClick);
}

// Update chart with new data
function updateChart(newData) {
    if (!mainChart) return;

    // Update candlestick data
    if (newData.candlestick) {
        chartData.candlesticks.push(newData.candlestick);
        if (chartData.candlesticks.length > 1000) {
            chartData.candlesticks.shift();
        }
    }

    // Update volume data
    if (newData.volume) {
        chartData.volume.push(newData.volume);
        if (chartData.volume.length > 1000) {
            chartData.volume.shift();
        }
    }

    // Update indicators
    if (newData.indicators) {
        Object.entries(newData.indicators).forEach(([name, data]) => {
            if (!chartData.indicators[name]) {
                chartData.indicators[name] = [];
            }
            chartData.indicators[name].push(data);
            if (chartData.indicators[name].length > 1000) {
                chartData.indicators[name].shift();
            }
        });
    }

    // Prepare data for Plotly
    const plotData = [
        {
            type: 'candlestick',
            x: chartData.candlesticks.map(d => d.time),
            open: chartData.candlesticks.map(d => d.open),
            high: chartData.candlesticks.map(d => d.high),
            low: chartData.candlesticks.map(d => d.low),
            close: chartData.candlesticks.map(d => d.close),
            name: 'Price',
            yaxis: 'y'
        },
        {
            type: 'bar',
            x: chartData.volume.map(d => d.time),
            y: chartData.volume.map(d => d.value),
            name: 'Volume',
            yaxis: 'y2',
            marker: {
                color: chartData.volume.map(d => d.value > 0 ? 'rgba(0,255,0,0.3)' : 'rgba(255,0,0,0.3)')
            }
        }
    ];

    // Add indicator data
    Object.entries(chartData.indicators).forEach(([name, data]) => {
        plotData.push({
            type: 'scatter',
            x: data.map(d => d.time),
            y: data.map(d => d.value),
            name: name,
            yaxis: 'y',
            line: {
                color: indicators[name]?.color || '#ffffff'
            }
        });
    });

    // Update the chart
    Plotly.react(mainChart, plotData);
}

// Add indicator to chart
function addIndicator(name, data, color = '#ffffff') {
    indicators[name] = { color };
    if (data) {
        chartData.indicators[name] = data;
        updateChart({ indicators: { [name]: data[data.length - 1] } });
    }
}

// Remove indicator from chart
function removeIndicator(name) {
    delete indicators[name];
    delete chartData.indicators[name];
    updateChart({});
}

// Handle chart relayout events
function handleChartRelayout(eventData) {
    // Save zoom level and position
    if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
        localStorage.setItem('chartRange', JSON.stringify({
            start: eventData['xaxis.range[0]'],
            end: eventData['xaxis.range[1]']
        }));
    }
}

// Handle chart click events
function handleChartClick(eventData) {
    const point = eventData.points[0];
    if (point) {
        // Emit click event with point data
        const event = new CustomEvent('chartClick', {
            detail: {
                time: point.x,
                value: point.y,
                curveNumber: point.curveNumber,
                pointNumber: point.pointNumber
            }
        });
        document.dispatchEvent(event);
    }
}

// Reset chart zoom
function resetChartZoom() {
    if (!mainChart) return;
    Plotly.relayout(mainChart, {
        'xaxis.autorange': true,
        'yaxis.autorange': true
    });
}

// Export functions
export {
    initChart,
    updateChart,
    addIndicator,
    removeIndicator,
    resetChartZoom
};