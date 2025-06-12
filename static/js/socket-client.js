// Initialize Socket.IO connection
const socket = io();

// Connection status handling
socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
});

// Real-time trade updates
socket.on('trade_update', (data) => {
    updateTradeList(data);
    updatePerformanceMetrics(data);
    showNotification('Trade Update', `New trade executed: ${data.symbol} at ${data.price}`);
});

// Real-time price updates
socket.on('price_update', (data) => {
    updatePriceDisplay(data);
    updateChart(data);
});

// Bot status updates
socket.on('bot_status', (data) => {
    updateBotStatus(data);
    if (data.status === 'error') {
        showNotification('Bot Error', data.message, 'error');
    }
});

// Performance updates
socket.on('performance_update', (data) => {
    updatePerformanceMetrics(data);
});

// Helper functions
function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (statusElement) {
        statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        statusElement.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

function updateTradeList(trade) {
    const tradeList = document.getElementById('trade-list');
    if (tradeList) {
        const tradeElement = createTradeElement(trade);
        tradeList.insertBefore(tradeElement, tradeList.firstChild);
        
        // Remove old trades if list gets too long
        if (tradeList.children.length > 50) {
            tradeList.removeChild(tradeList.lastChild);
        }
    }
}

function createTradeElement(trade) {
    const div = document.createElement('div');
    div.className = `trade-item ${trade.type.toLowerCase()}`;
    div.innerHTML = `
        <div class="trade-symbol">${trade.symbol}</div>
        <div class="trade-price">${formatPrice(trade.price)}</div>
        <div class="trade-amount">${formatAmount(trade.amount)}</div>
        <div class="trade-profit ${trade.profit >= 0 ? 'profit' : 'loss'}">
            ${formatProfit(trade.profit)}
        </div>
        <div class="trade-time">${formatTime(trade.timestamp)}</div>
    `;
    return div;
}

function updatePerformanceMetrics(data) {
    // Update total profit
    const profitElement = document.getElementById('total-profit');
    if (profitElement) {
        profitElement.textContent = formatProfit(data.total_profit);
        profitElement.className = `total-profit ${data.total_profit >= 0 ? 'profit' : 'loss'}`;
    }

    // Update win rate
    const winRateElement = document.getElementById('win-rate');
    if (winRateElement) {
        winRateElement.textContent = `${(data.win_rate * 100).toFixed(1)}%`;
    }

    // Update active trades
    const activeTradesElement = document.getElementById('active-trades');
    if (activeTradesElement) {
        activeTradesElement.textContent = data.active_trades;
    }
}

function updatePriceDisplay(data) {
    const priceElement = document.getElementById('current-price');
    if (priceElement) {
        priceElement.textContent = formatPrice(data.price);
        priceElement.className = `current-price ${data.change >= 0 ? 'up' : 'down'}`;
    }
}

function updateBotStatus(data) {
    const statusElement = document.getElementById('bot-status');
    if (statusElement) {
        statusElement.className = `bot-status ${data.status}`;
        statusElement.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    }

    // Update control buttons
    const startButton = document.getElementById('start-bot');
    const stopButton = document.getElementById('stop-bot');
    if (startButton && stopButton) {
        startButton.disabled = data.status === 'running';
        stopButton.disabled = data.status !== 'running';
    }
}

function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-title">${title}</div>
        <div class="notification-message">${message}</div>
        <button class="notification-close">&times;</button>
    `;

    document.body.appendChild(notification);

    // Add close button functionality
    const closeButton = notification.querySelector('.notification-close');
    closeButton.addEventListener('click', () => {
        notification.remove();
    });

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Utility functions
function formatPrice(price) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(price);
}

function formatAmount(amount) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 4,
        maximumFractionDigits: 4
    }).format(amount);
}

function formatProfit(profit) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        signDisplay: 'always',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(profit);
}

function formatTime(timestamp) {
    return new Date(timestamp).toLocaleTimeString();
}

// Export functions for use in other modules
export {
    socket,
    updateConnectionStatus,
    updateTradeList,
    updatePerformanceMetrics,
    updatePriceDisplay,
    updateBotStatus,
    showNotification
};

