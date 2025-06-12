// Export functionality for trade history and performance reports

// Export trade history to CSV
function exportTradeHistoryCSV(trades) {
    if (!trades || trades.length === 0) {
        showError('No trade data available to export');
        return;
    }

    const headers = ['Timestamp', 'Symbol', 'Type', 'Price', 'Quantity', 'Profit', 'Balance'];
    const csvContent = [
        headers.join(','),
        ...trades.map(trade => [
            new Date(trade.timestamp).toISOString(),
            trade.symbol,
            trade.type,
            trade.price,
            trade.quantity,
            trade.profit || '',
            trade.balance
        ].join(','))
    ].join('\n');

    downloadFile(csvContent, 'trade_history.csv', 'text/csv');
}

// Export performance report to PDF
function exportPerformanceReport(data) {
    if (!data) {
        showError('No performance data available to export');
        return;
    }

    // Create HTML content for PDF
    const htmlContent = `
        <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
                    th { background-color: #f5f5f5; }
                    .summary { margin: 20px 0; }
                    .chart { margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>Trading Performance Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total Trades: ${data.totalTrades}</p>
                    <p>Winning Trades: ${data.winningTrades}</p>
                    <p>Losing Trades: ${data.losingTrades}</p>
                    <p>Win Rate: ${(data.winRate * 100).toFixed(2)}%</p>
                    <p>Total Profit: ${data.totalProfit.toFixed(2)}</p>
                    <p>Average Profit per Trade: ${data.avgProfit.toFixed(2)}</p>
                    <p>Largest Win: ${data.largestWin.toFixed(2)}</p>
                    <p>Largest Loss: ${data.largestLoss.toFixed(2)}</p>
                </div>

                <div class="chart">
                    <h2>Equity Curve</h2>
                    <img src="${data.equityCurveImage}" alt="Equity Curve" style="width: 100%;">
                </div>

                <h2>Trade History</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Symbol</th>
                            <th>Type</th>
                            <th>Price</th>
                            <th>Quantity</th>
                            <th>Profit</th>
                            <th>Balance</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.trades.map(trade => `
                            <tr>
                                <td>${new Date(trade.timestamp).toLocaleString()}</td>
                                <td>${trade.symbol}</td>
                                <td>${trade.type}</td>
                                <td>${trade.price}</td>
                                <td>${trade.quantity}</td>
                                <td>${trade.profit || '-'}</td>
                                <td>${trade.balance}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </body>
        </html>
    `;

    // Convert HTML to PDF using html2pdf.js
    const element = document.createElement('div');
    element.innerHTML = htmlContent;
    document.body.appendChild(element);

    const opt = {
        margin: 1,
        filename: 'performance_report.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save().then(() => {
        document.body.removeChild(element);
    });
}

// Helper function to download files
function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}

// Export functions
window.exportReport = {
    exportTradeHistoryCSV,
    exportPerformanceReport
};

// Export Report Module

// Generate PDF Report
async function generatePDFReport(data) {
    try {
        const response = await fetch('/api/export/pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Failed to generate PDF report');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trading-report-${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        return true;
    } catch (error) {
        console.error('Error generating PDF report:', error);
        showNotification('Export Error', 'Failed to generate PDF report', 'error');
        return false;
    }
}

// Generate CSV Report
async function generateCSVReport(data) {
    try {
        const response = await fetch('/api/export/csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Failed to generate CSV report');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trading-report-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        return true;
    } catch (error) {
        console.error('Error generating CSV report:', error);
        showNotification('Export Error', 'Failed to generate CSV report', 'error');
        return false;
    }
}

// Generate Excel Report
async function generateExcelReport(data) {
    try {
        const response = await fetch('/api/export/excel', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Failed to generate Excel report');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trading-report-${new Date().toISOString().split('T')[0]}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        return true;
    } catch (error) {
        console.error('Error generating Excel report:', error);
        showNotification('Export Error', 'Failed to generate Excel report', 'error');
        return false;
    }
}

// Generate HTML Report
async function generateHTMLReport(data) {
    try {
        const response = await fetch('/api/export/html', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Failed to generate HTML report');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trading-report-${new Date().toISOString().split('T')[0]}.html`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        return true;
    } catch (error) {
        console.error('Error generating HTML report:', error);
        showNotification('Export Error', 'Failed to generate HTML report', 'error');
        return false;
    }
}

// Format data for export
function formatExportData(data) {
    return {
        trades: data.trades.map(trade => ({
            timestamp: trade.timestamp,
            symbol: trade.symbol,
            type: trade.type,
            price: trade.price,
            amount: trade.amount,
            profit: trade.profit,
            fee: trade.fee
        })),
        performance: {
            total_profit: data.total_profit,
            win_rate: data.win_rate,
            total_trades: data.total_trades,
            average_profit: data.average_profit,
            max_drawdown: data.max_drawdown,
            sharpe_ratio: data.sharpe_ratio
        },
        settings: {
            strategy: data.strategy,
            timeframe: data.timeframe,
            symbols: data.symbols,
            start_date: data.start_date,
            end_date: data.end_date
        }
    };
}

// Show notification
function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-title">${title}</div>
        <div class="notification-message">${message}</div>
        <button class="notification-close">&times;</button>
    `;

    document.body.appendChild(notification);

    const closeButton = notification.querySelector('.notification-close');
    closeButton.addEventListener('click', () => {
        notification.remove();
    });

    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Export functions
export {
    generatePDFReport,
    generateCSVReport,
    generateExcelReport,
    generateHTMLReport,
    formatExportData
};