/**
 * Plotly Chart Rendering — Candlestick + SMAs + Volume Spikes
 *
 * Colors match Matplotlib theme:
 *   up=#26A69A, down=#EF5350, spike=#FFD600
 *   SMA30=#2196F3, SMA45=#FF9800, SMA60=#9C27B0
 */

const COLORS = {
    bg: '#1a1a2e',
    cardBg: '#16213e',
    grid: '#2a2a4a',
    text: '#cccccc',
    textMuted: '#999999',
    up: '#26A69A',
    down: '#EF5350',
    spike: '#FFD600',
    sma30: '#2196F3',
    sma45: '#FF9800',
    sma60: '#9C27B0',
};

const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: COLORS.cardBg,
    plot_bgcolor: COLORS.bg,
    font: { color: COLORS.text, size: 11 },
    margin: { l: 60, r: 20, t: 30, b: 40 },
    xaxis: {
        gridcolor: COLORS.grid,
        linecolor: COLORS.grid,
        rangeslider: { visible: false },
    },
    yaxis: {
        gridcolor: COLORS.grid,
        linecolor: COLORS.grid,
    },
};

function loadChart(resultId) {
    fetch(`/api/result/${resultId}/chart-data`)
        .then(r => {
            if (!r.ok) throw new Error('No chart data');
            return r.json();
        })
        .then(data => {
            renderPriceChart(data);
            renderVolumeChart(data);
        })
        .catch(err => {
            document.getElementById('price-chart').innerHTML =
                `<div style="padding:40px;text-align:center;color:${COLORS.textMuted}">
                    No chart data available for this result.
                </div>`;
            document.getElementById('volume-chart').innerHTML = '';
        });
}

function renderPriceChart(data) {
    const traces = [];

    // Candlestick
    traces.push({
        type: 'candlestick',
        x: data.dates,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        increasing: { line: { color: COLORS.up }, fillcolor: COLORS.up },
        decreasing: { line: { color: COLORS.down }, fillcolor: COLORS.down },
        name: 'OHLC',
        showlegend: false,
    });

    // SMA lines
    const smaConfig = [
        { key: 'sma_30', color: COLORS.sma30, name: 'SMA 30' },
        { key: 'sma_45', color: COLORS.sma45, name: 'SMA 45' },
        { key: 'sma_60', color: COLORS.sma60, name: 'SMA 60' },
    ];

    for (const sma of smaConfig) {
        if (data.sma && data.sma[sma.key]) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: data.dates,
                y: data.sma[sma.key],
                line: { color: sma.color, width: 1.5 },
                name: sma.name,
                opacity: 0.85,
            });
        }
    }

    // Volume spike markers on price chart
    if (data.is_spike) {
        const spikeDates = [];
        const spikeHighs = [];
        for (let i = 0; i < data.is_spike.length; i++) {
            if (data.is_spike[i]) {
                spikeDates.push(data.dates[i]);
                spikeHighs.push(data.high[i]);
            }
        }
        if (spikeDates.length > 0) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                x: spikeDates,
                y: spikeHighs,
                marker: {
                    symbol: 'star',
                    size: 10,
                    color: COLORS.spike,
                },
                name: 'Vol Spike',
            });
        }
    }

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            title: 'Price',
        },
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(26,26,46,0.7)',
            font: { size: 10 },
        },
        height: 400,
    };

    Plotly.newPlot('price-chart', traces, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
}

function renderVolumeChart(data) {
    // Color bars: spike=yellow, up=green, down=red
    const colors = [];
    for (let i = 0; i < data.volume.length; i++) {
        if (data.is_spike && data.is_spike[i]) {
            colors.push(COLORS.spike);
        } else if (data.close[i] >= data.open[i]) {
            colors.push(COLORS.up);
        } else {
            colors.push(COLORS.down);
        }
    }

    const traces = [{
        type: 'bar',
        x: data.dates,
        y: data.volume,
        marker: { color: colors },
        name: 'Volume',
        showlegend: false,
    }];

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        yaxis: {
            ...PLOTLY_LAYOUT_BASE.yaxis,
            title: 'Volume',
        },
        height: 200,
        margin: { ...PLOTLY_LAYOUT_BASE.margin, t: 10 },
    };

    Plotly.newPlot('volume-chart', traces, layout, {
        responsive: true,
        displayModeBar: false,
    });
}
