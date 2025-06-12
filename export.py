from flask import Blueprint, request, send_file
import pandas as pd
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os

export_bp = Blueprint('export', __name__)

@export_bp.route('/', methods=['GET'])
def export():
    fmt = request.args.get('format', 'csv')
    # TODO: generate CSV or PDF and send
    return send_file('path/to/generated.file', as_attachment=True)

class ExportManager:
    def __init__(self, trades, equity_curve):
        self.trades = trades
        self.equity_curve = equity_curve
    
    def export_csv(self, filepath):
        """Export trades to CSV file"""
        try:
            # Convert trades to DataFrame
            df = pd.DataFrame(self.trades)
            
            # Add timestamp column
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            
            # Reorder columns
            columns = [
                'timestamp', 'symbol', 'side', 'type',
                'price', 'quantity', 'status', 'pnl'
            ]
            df = df[columns]
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def export_pdf(self, filepath):
        """Export trading report to PDF"""
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create content
            content = []
            
            # Add title
            title = Paragraph(
                "Trading Report",
                styles['Title']
            )
            content.append(title)
            
            # Add timestamp
            timestamp = Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles['Normal']
            )
            content.append(timestamp)
            
            # Add summary statistics
            stats = self._calculate_statistics()
            stats_text = [
                f"Total Trades: {stats['total_trades']}",
                f"Winning Trades: {stats['winning_trades']}",
                f"Losing Trades: {stats['losing_trades']}",
                f"Win Rate: {stats['win_rate']:.2f}%",
                f"Total P&L: {stats['total_pnl']:.2f}",
                f"Average P&L: {stats['avg_pnl']:.2f}",
                f"Largest Win: {stats['largest_win']:.2f}",
                f"Largest Loss: {stats['largest_loss']:.2f}"
            ]
            
            for stat in stats_text:
                content.append(Paragraph(stat, styles['Normal']))
            
            # Add trade history table
            if self.trades:
                # Convert trades to table data
                table_data = [['Time', 'Symbol', 'Side', 'Price', 'Quantity', 'P&L']]
                
                for trade in self.trades:
                    row = [
                        datetime.fromtimestamp(trade['time']/1000).strftime('%Y-%m-%d %H:%M:%S'),
                        trade['symbol'],
                        trade['side'],
                        f"{trade['price']:.8f}",
                        f"{trade['quantity']:.8f}",
                        f"{trade.get('pnl', 0):.2f}"
                    ]
                    table_data.append(row)
                
                # Create table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                content.append(table)
            
            # Build PDF
            doc.build(content)
            return True
            
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return False
    
    def _calculate_statistics(self):
        """Calculate trading statistics"""
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
        
        for trade in self.trades:
            if 'pnl' in trade:
                pnl = trade['pnl']
                stats['total_pnl'] += pnl
                
                if pnl > 0:
                    stats['winning_trades'] += 1
                    stats['largest_win'] = max(stats['largest_win'], pnl)
                else:
                    stats['losing_trades'] += 1
                    stats['largest_loss'] = min(stats['largest_loss'], pnl)
        
        # Calculate additional statistics
        if stats['total_trades'] > 0:
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
            stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades']
        else:
            stats['win_rate'] = 0
            stats['avg_pnl'] = 0
        
        return stats
    
    def export_json(self, filepath):
        """Export trades to JSON file"""
        try:
            # Convert trades to JSON-serializable format
            export_data = {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'statistics': self._calculate_statistics(),
                'export_time': int(datetime.now().timestamp() * 1000)
            }
            
            # Export to JSON
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=4)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

