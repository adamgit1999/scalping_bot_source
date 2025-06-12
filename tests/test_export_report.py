import pytest
from export_report import ExportReport
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile

@pytest.fixture
def export_report():
    return ExportReport()

@pytest.fixture
def sample_trades():
    return [
        {
            'id': '12345',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000.0,
            'amount': 0.1,
            'total': 5000.0,
            'fee': 5.0,
            'profit': 100.0,
            'timestamp': datetime.now().isoformat()
        },
        {
            'id': '12346',
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'price': 51000.0,
            'amount': 0.1,
            'total': 5100.0,
            'fee': 5.1,
            'profit': 200.0,
            'timestamp': datetime.now().isoformat()
        }
    ]

@pytest.fixture
def sample_performance():
    return {
        'total_profit': 1000.0,
        'win_rate': 0.65,
        'total_trades': 100,
        'active_trades': 5,
        'timestamp': datetime.now().isoformat()
    }

def test_initialize(export_report):
    """Test export report initialization."""
    assert export_report is not None

def test_export_pdf(export_report, sample_trades, sample_performance):
    """Test PDF export."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, 'test_report.pdf')
        result = export_report.export_pdf(sample_trades, sample_performance, filename)
        
        assert result is True
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0

def test_export_csv(export_report, sample_trades):
    """Test CSV export."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, 'test_report.csv')
        result = export_report.export_csv(sample_trades, filename)
        
        assert result is True
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0
        
        # Verify CSV content
        df = pd.read_csv(filename)
        assert len(df) == len(sample_trades)
        assert all(col in df.columns for col in ['id', 'symbol', 'side', 'price', 'amount', 'total', 'fee', 'profit'])

def test_export_excel(export_report, sample_trades, sample_performance):
    """Test Excel export."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, 'test_report.xlsx')
        result = export_report.export_excel(sample_trades, sample_performance, filename)
        
        assert result is True
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0
        
        # Verify Excel content
        with pd.ExcelFile(filename) as xls:
            assert 'Trades' in xls.sheet_names
            assert 'Performance' in xls.sheet_names
            
            trades_df = pd.read_excel(xls, 'Trades')
            assert len(trades_df) == len(sample_trades)
            
            perf_df = pd.read_excel(xls, 'Performance')
            assert len(perf_df) == 1

def test_export_html(export_report, sample_trades, sample_performance):
    """Test HTML export."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, 'test_report.html')
        result = export_report.export_html(sample_trades, sample_performance, filename)
        
        assert result is True
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0
        
        # Verify HTML content
        with open(filename, 'r') as f:
            content = f.read()
            assert '<html' in content
            assert '<body' in content
            assert '<table' in content
            assert 'BTC/USDT' in content

def test_format_trade_data(export_report, sample_trades):
    """Test trade data formatting."""
    formatted_data = export_report._format_trade_data(sample_trades)
    
    assert isinstance(formatted_data, pd.DataFrame)
    assert len(formatted_data) == len(sample_trades)
    assert all(col in formatted_data.columns for col in ['id', 'symbol', 'side', 'price', 'amount', 'total', 'fee', 'profit'])

def test_format_performance_data(export_report, sample_performance):
    """Test performance data formatting."""
    formatted_data = export_report._format_performance_data(sample_performance)
    
    assert isinstance(formatted_data, pd.DataFrame)
    assert len(formatted_data) == 1
    assert all(col in formatted_data.columns for col in ['total_profit', 'win_rate', 'total_trades', 'active_trades'])

def test_generate_pdf_content(export_report, sample_trades, sample_performance):
    """Test PDF content generation."""
    content = export_report._generate_pdf_content(sample_trades, sample_performance)
    
    assert isinstance(content, list)
    assert len(content) > 0
    assert any('BTC/USDT' in str(item) for item in content)
    assert any('Performance' in str(item) for item in content)

def test_generate_html_content(export_report, sample_trades, sample_performance):
    """Test HTML content generation."""
    content = export_report._generate_html_content(sample_trades, sample_performance)
    
    assert isinstance(content, str)
    assert '<html' in content
    assert '<body' in content
    assert '<table' in content
    assert 'BTC/USDT' in content
    assert 'Performance' in content

def test_edge_cases(export_report):
    """Test edge cases for export functions."""
    # Empty trades list
    empty_trades = []
    empty_performance = {
        'total_profit': 0.0,
        'win_rate': 0.0,
        'total_trades': 0,
        'active_trades': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test PDF export with empty data
        pdf_filename = os.path.join(temp_dir, 'empty_report.pdf')
        result = export_report.export_pdf(empty_trades, empty_performance, pdf_filename)
        assert result is True
        assert os.path.exists(pdf_filename)
        
        # Test CSV export with empty data
        csv_filename = os.path.join(temp_dir, 'empty_report.csv')
        result = export_report.export_csv(empty_trades, csv_filename)
        assert result is True
        assert os.path.exists(csv_filename)
        
        # Test Excel export with empty data
        excel_filename = os.path.join(temp_dir, 'empty_report.xlsx')
        result = export_report.export_excel(empty_trades, empty_performance, excel_filename)
        assert result is True
        assert os.path.exists(excel_filename)
        
        # Test HTML export with empty data
        html_filename = os.path.join(temp_dir, 'empty_report.html')
        result = export_report.export_html(empty_trades, empty_performance, html_filename)
        assert result is True
        assert os.path.exists(html_filename) 