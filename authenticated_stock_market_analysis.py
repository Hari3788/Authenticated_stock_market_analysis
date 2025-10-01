from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional
import sqlite3
import bcrypt
import jwt
from functools import wraps

app = Flask(__name__)
CORS(app)

# Security Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['JWT_EXPIRATION_HOURS'] = 24

# API Configuration
ALPHA_VANTAGE_API_KEY = "SAU7YTS2D4T2IKKU"
BASE_URL = "https://www.alphavantage.co/query"

# ==================== DATABASE SETUP ====================
def init_db():
    """Initialize SQLite database for users"""
    conn = sqlite3.connect('stock_app.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

# ==================== AUTHENTICATION HELPERS ====================
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_token(user_id: int, username: str) -> str:
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS'])
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token: str) -> Dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return {'valid': True, 'data': payload}
    except jwt.ExpiredSignatureError:
        return {'valid': False, 'error': 'Token expired'}
    except jwt.InvalidTokenError:
        return {'valid': False, 'error': 'Invalid token'}

# ==================== AUTHENTICATION DECORATOR ====================
def token_required(f):
    """Decorator to protect routes - requires valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Format: "Bearer <token>"
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        # Verify token
        result = verify_token(token)
        if not result['valid']:
            return jsonify({'error': result['error']}), 401
        
        # Pass user data to route
        return f(result['data'], *args, **kwargs)
    
    return decorated

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.get_json()
    
    # Validate input
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400
    
    username = data['username']
    email = data['email']
    password = data['password']
    
    # Hash password
    hashed_password = hash_password(password)
    
    try:
        conn = sqlite3.connect('stock_app.db')
        cursor = conn.cursor()
        
        # Insert user
        cursor.execute('''
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Generate token
        token = generate_token(user_id, username)
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {'id': user_id, 'username': username, 'email': email}
        }), 201
        
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400
    
    username = data['username']
    password = data['password']
    
    try:
        conn = sqlite3.connect('stock_app.db')
        cursor = conn.cursor()
        
        # Get user
        cursor.execute('SELECT id, username, email, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user_id, username, email, hashed_password = user
        
        # Verify password
        if not verify_password(password, hashed_password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate token
        token = generate_token(user_id, username)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {'id': user_id, 'username': username, 'email': email}
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify(current_user):
    """Verify if token is valid"""
    return jsonify({
        'valid': True,
        'user': current_user
    }), 200

# ==================== STOCK ANALYZER CLASS ====================
class StockAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': quote.get('01. symbol', ''),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'previous_close': float(quote.get('08. previous close', 0))
                }
            else:
                return {'error': 'Stock not found or API limit reached'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_historical_data(self, symbol: str, period: str = '3month') -> Dict:
        """Get historical stock data"""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                dates = []
                prices = []
                volumes = []
                
                # Get data based on period
                end_date = datetime.now()
                if period == '1week':
                    start_date = end_date - timedelta(days=7)
                elif period == '1month':
                    start_date = end_date - timedelta(days=30)
                elif period == '3month':
                    start_date = end_date - timedelta(days=90)
                elif period == '1year':
                    start_date = end_date - timedelta(days=365)
                else:
                    start_date = end_date - timedelta(days=90)
                
                for date_str, values in time_series.items():
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if date >= start_date:
                        dates.append(date_str)
                        prices.append(float(values['4. close']))
                        volumes.append(int(values['5. volume']))
                
                # Sort by date (oldest first)
                sorted_data = sorted(zip(dates, prices, volumes))
                dates, prices, volumes = zip(*sorted_data) if sorted_data else ([], [], [])
                
                return {
                    'dates': list(dates),
                    'prices': list(prices),
                    'volumes': list(volumes)
                }
            else:
                return {'error': 'Historical data not found'}
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_technical_indicators(self, prices: List[float]) -> Dict:
        """Calculate technical indicators"""
        if len(prices) < 20:
            return {'error': 'Insufficient data for analysis'}
        
        prices_array = np.array(prices)
        
        # Simple Moving Averages
        sma_10 = np.convolve(prices_array, np.ones(10)/10, mode='valid')
        sma_20 = np.convolve(prices_array, np.ones(20)/20, mode='valid')
        
        # RSI Calculation
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gain = np.where(deltas > 0, deltas, 0)
            loss = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
            avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
            
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(prices_array)
        
        return {
            'sma_10': sma_10[-10:].tolist() if len(sma_10) > 0 else [],
            'sma_20': sma_20[-10:].tolist() if len(sma_20) > 0 else [],
            'rsi': rsi[-10:].tolist() if len(rsi) > 0 else [],
            'current_rsi': rsi[-1] if len(rsi) > 0 else 0
        }

# Initialize analyzer
analyzer = StockAnalyzer(ALPHA_VANTAGE_API_KEY)

# Popular stocks for demo
POPULAR_STOCKS = [
    {'symbol': 'AAPL', 'name': 'Apple Inc.'},
    {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
    {'symbol': 'MSFT', 'name': 'Microsoft Corp.'},
    {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
    {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
    {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
    {'symbol': 'NVDA', 'name': 'NVIDIA Corp.'},
    {'symbol': 'NFLX', 'name': 'Netflix Inc.'}
]

# ==================== PROTECTED STOCK ROUTES ====================
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', popular_stocks=POPULAR_STOCKS)

@app.route('/api/quote/<symbol>')
@token_required
def get_quote(current_user, symbol):
    """API endpoint to get stock quote - PROTECTED"""
    quote = analyzer.get_stock_quote(symbol.upper())
    return jsonify(quote)

@app.route('/api/history/<symbol>')
@token_required
def get_history(current_user, symbol):
    """API endpoint to get historical data - PROTECTED"""
    period = request.args.get('period', '3month')
    history = analyzer.get_historical_data(symbol.upper(), period)
    return jsonify(history)

@app.route('/api/analysis/<symbol>')
@token_required
def get_analysis(current_user, symbol):
    """API endpoint to get technical analysis - PROTECTED"""
    period = request.args.get('period', '3month')
    history = analyzer.get_historical_data(symbol.upper(), period)
    
    if 'error' in history:
        return jsonify(history)
    
    if history['prices']:
        indicators = analyzer.calculate_technical_indicators(history['prices'])
        return jsonify({
            'symbol': symbol.upper(),
            'indicators': indicators,
            'latest_price': history['prices'][-1] if history['prices'] else 0,
            'price_change_1d': ((history['prices'][-1] - history['prices'][-2]) / history['prices'][-2] * 100) if len(history['prices']) >= 2 else 0
        })
    else:
        return jsonify({'error': 'No price data available'})

@app.route('/api/search/<query>')
@token_required
def search_stocks(current_user, query):
    """API endpoint to search stocks - PROTECTED"""
    matching_stocks = [
        stock for stock in POPULAR_STOCKS 
        if query.lower() in stock['name'].lower() or query.lower() in stock['symbol'].lower()
    ]
    return jsonify(matching_stocks)

@app.route('/compare')
def compare():
    """Stock comparison page"""
    return render_template('compare.html')

@app.route('/portfolio')
def portfolio():
    """Portfolio tracking page"""
    return render_template('portfolio.html')

# ==================== MAIN ====================
if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Initialize database
    init_db()
    
    print("\n" + "="*50)
    print("🚀 Stock Market Analysis Server with JWT Auth")
    print("="*50)
    print("📊 Dashboard: http://127.0.0.1:5000")
    print("\n🔐 Authentication Endpoints:")
    print("   POST /api/auth/register - Register new user")
    print("   POST /api/auth/login    - Login user")
    print("   GET  /api/auth/verify   - Verify token")
    print("\n📈 Protected Stock Endpoints (require token):")
    print("   GET /api/quote/<symbol>")
    print("   GET /api/history/<symbol>")
    print("   GET /api/analysis/<symbol>")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)