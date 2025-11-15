from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from math import ceil
import os
import requests
import joblib
import json
import numpy as np
import datetime
import calendar
import math
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import redis
from functools import wraps
import hashlib

# Load environment variables
load_dotenv()
API_KEY = os.getenv("EXCHANGE_API_KEY")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# ===== KONFIGURASI DATABASE =====
# Pakai DATABASE_URL dari Railway
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ===== DEFINE MODELS =====
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50))
    password = db.Column(db.String(255))

class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer)
    category = db.Column(db.String(50))
    type = db.Column(db.String(20))  # 'pengeluaran' atau 'pemasukan'
    amount = db.Column(db.Numeric(12, 2))
    currency = db.Column(db.String(10))
    amount_idr = db.Column(db.Numeric(12, 2))
    note = db.Column(db.Text)
    date = db.Column(db.Date)

# Create tables
with app.app_context():
    db.create_all()
    print("✅ Database tables created/verified!")

# ===== LAZY LOAD ML MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")

MODEL_PATH = os.path.join(ML_DIR, "edubadget_gb_multi.joblib")
SCALER_PATH = os.path.join(ML_DIR, "edubadget_income_scaler.joblib")
BASELINE_PATH = os.path.join(ML_DIR, "asia_baseline_frac.json")

_ml_model_cache = None
_scaler_cache = None
_baseline_cache = None

def get_ml_model():
    """Lazy load ML model - hanya load sekali"""
    global _ml_model_cache, _scaler_cache, _baseline_cache
    
    if _ml_model_cache is None:
        try:
            _ml_model_cache = joblib.load(MODEL_PATH)
            _scaler_cache = joblib.load(SCALER_PATH)
            with open(BASELINE_PATH, "r") as f:
                _baseline_cache = json.load(f)
            print("✅ Model ML berhasil dimuat (lazy load).")
        except Exception as e:
            print(f"❌ GAGAL load model ML: {e}")
            return None, None, None
    
    return _ml_model_cache, _scaler_cache, _baseline_cache

# ===== EMOJI HELPER FUNCTION =====
def get_category_emoji(category):
    """Return emoji based on category"""
    emoji_map = {
        'uang bulanan': '💰',
        'gaji': '💵',
        'transfer orang tua': '👨‍👩‍👧',
        'beasiswa': '🎓',
        'freelance': '💻',
        'makanan/minuman': '🍔',
        'makanan': '🍽️',
        'minuman': '🥤',
        'bensin/transportasi': '⛽',
        'transportasi': '🚗',
        'laundry': '👕',
        'hiburan': '🎮',
        'belanja': '🛒',
        'kuliah/buku': '📚',
        'buku': '📖',
        'pakaian': '👔',
        'kos': '🏠',
        'internet/pulsa': '📱',
        'internet': '🌐',
        'pulsa': '📞',
        'kesehatan': '⚕️',
        'lainnya': '📌'
    }
    
    cat_lower = (category or '').lower().strip()
    return emoji_map.get(cat_lower, '💡')

# Register emoji function dengan Jinja
app.jinja_env.globals.update(get_emoji=get_category_emoji)

# ===== SETUP REDIS (Optional - untuk caching) =====
try:
    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
    else:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis Connected")
except:
    REDIS_AVAILABLE = False
    redis_client = None
    print("⚠️ Redis not available - caching disabled")

# -------------------- Helper Functions --------------------

def convert_to_idr(amount, currency):
    """Convert currency to IDR"""
    currency = currency.upper()
    if currency == "IDR":
        return float(amount)
    
    # Cache exchange rate untuk 1 jam
    cache_key = f"exchange_rate:{currency}:IDR"
    if REDIS_AVAILABLE:
        try:
            cached_rate = redis_client.get(cache_key)
            if cached_rate:
                return float(amount) * float(cached_rate)
        except:
            pass
    
    # Fallback rate jika API tidak available
    fallback_rates = {
        'USD': 15000,
        'EUR': 16000,
        'SGD': 11000,
        'MYR': 3200
    }
    
    if currency in fallback_rates:
        return float(amount) * fallback_rates[currency]
    
    # Default fallback
    return float(amount) * 10000

def recommend_budget_ml_cached(income_idr, lifestyle):
    """ML prediction dengan CACHING"""
    cache_key = f"ml_prediction:{int(income_idr)}:{lifestyle}"
    
    if REDIS_AVAILABLE:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                print(f"✓ ML Cache hit: {cache_key}")
                return json.loads(cached)
        except:
            pass
    
    ml_model, scaler, baseline_frac = get_ml_model()
    
    if ml_model is None or scaler is None:
        result = build_fixed_recommendation(income_idr, lifestyle)
    else:
        try:
            import pandas as pd
            X = pd.DataFrame([[income_idr]], columns=['income'])
            scaled_income = scaler.transform(X)
            pred_frac = ml_model.predict(scaled_income)[0]
            
            categories = ["Food", "Transport", "Entertainment", "Laundry", "Savings"]
            results = []
            
            for i, cat in enumerate(categories):
                frac = float(pred_frac[i])
                amount = income_idr * frac
                results.append({
                    "category": cat,
                    "fraction": frac,
                    "amount": amount
                })
            
            lifestyle = lifestyle.lower()
            if lifestyle == "hemat":
                flex_cat = "Emergency Fund"
            elif lifestyle == "moderat":
                flex_cat = "Personal Needs"
            else:
                flex_cat = "Self-Improvement"
            
            results.append({
                "category": flex_cat,
                "fraction": 0.20,
                "amount": income_idr * 0.20
            })
            
            result = results
        except Exception as e:
            print(f"ML Prediction error: {e}")
            result = build_fixed_recommendation(income_idr, lifestyle)
    
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(cache_key, 86400, json.dumps(result, default=str))
        except:
            pass
    
    return result

def build_fixed_recommendation(income_idr, lifestyle_key):
    """Fallback recommendation tanpa ML"""
    pct = {"Food":0.40,"Transport":0.12,"Entertainment":0.08,"Laundry":0.05,"Savings":0.15}
    lk = (lifestyle_key or "moderat").lower()
    
    if lk == "hemat":
        flex_name = "Emergency Fund"
    elif lk in ["moderat","normal"]:
        flex_name = "Personal Needs"
    else:
        flex_name = "Self-Improvement"
    
    res = [{"category":k,"fraction":v,"amount":int(round(income_idr * v))} for k,v in pct.items()]
    res.append({"category":flex_name,"fraction":0.20,"amount":int(round(income_idr * 0.20))})
    return res

def to_dec(v):
    """Convert to Decimal safely"""
    return Decimal(str(v or 0))

def categorize_student(name, student_categories):
    """Categorize transaction by name"""
    n = (name or "").lower()
    for k, keys in student_categories.items():
        for kw in keys:
            if kw in n:
                return k
    return 'lainnya'

def fetch_all_transactions_data():
    """Fetch data menggunakan SQLAlchemy"""
    summary = {}
    today_spending = {}
    weekly_data = {}
    kategori_rows = []
    
    today = datetime.date.today()
    first_day = today.replace(day=1)
    if today.month == 12:
        last_day = today.replace(year=today.year+1, month=1, day=1) - datetime.timedelta(days=1)
    else:
        last_day = today.replace(month=today.month+1, day=1) - datetime.timedelta(days=1)
    
    try:
        # Query 1: Summary
        result = db.session.execute(
            db.text("""
                SELECT type, COALESCE(SUM(amount_idr), 0) as total
                FROM transactions
                WHERE date >= :first_day AND date <= :last_day
                GROUP BY type
            """),
            {'first_day': first_day, 'last_day': last_day}
        )
        
        for row in result:
            summary[row[0]] = to_dec(row[1])
        
        # Query 2: Today's spending
        result = db.session.execute(
            db.text("""
                SELECT category, COALESCE(SUM(amount_idr), 0) as total
                FROM transactions
                WHERE type = 'pengeluaran' AND date = :today
                GROUP BY category
            """),
            {'today': today}
        )
        
        for row in result:
            if row[0]:
                today_spending[row[0]] = to_dec(row[1])
        
        # Query 3: Category totals
        result = db.session.execute(
            db.text("""
                SELECT 
                    category,
                    COALESCE(SUM(amount_idr), 0) as total,
                    COUNT(*) as tx_count,
                    COALESCE(AVG(amount_idr), 0) as avg_tx
                FROM transactions
                WHERE type = 'pengeluaran' 
                  AND date >= :first_day AND date <= :last_day
                GROUP BY category
            """),
            {'first_day': first_day, 'last_day': last_day}
        )
        
        for row in result:
            if row[0]:
                kategori_rows.append((
                    row[0],
                    to_dec(row[1]),
                    int(row[2]) if row[2] else 0,
                    None,
                    to_dec(row[3])
                ))
        
    except Exception as e:
        print(f"❌ Database error in fetch_all_transactions_data: {e}")
    
    return summary, today_spending, weekly_data, kategori_rows

def invalidate_analytics_cache(user_id=None):
    """Invalidate cache setelah ada transaksi baru"""
    if not REDIS_AVAILABLE:
        return
    
    try:
        user_id = user_id or session.get('user_id', 'anonymous')
        today = datetime.date.today()
        cache_key = f"analytics_data:v3:{user_id}:{today}"
        redis_client.delete(cache_key)
        print(f"✓ Cache invalidated: {cache_key}")
    except Exception as e:
        print(f"Invalidate cache error: {e}")

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    try:
        # Total pemasukan
        pemasukan_result = db.session.execute(
            db.text("SELECT COALESCE(SUM(amount_idr), 0) FROM transactions WHERE type='pemasukan'")
        )
        total_masuk = float(pemasukan_result.scalar() or 0)
        
        # Total pengeluaran
        pengeluaran_result = db.session.execute(
            db.text("SELECT COALESCE(SUM(amount_idr), 0) FROM transactions WHERE type='pengeluaran'")
        )
        total_keluar = float(pengeluaran_result.scalar() or 0)
        
        saldo = total_masuk - total_keluar
        
        # Data kategori
        kategori_result = db.session.execute(
            db.text("""
                SELECT category, SUM(amount_idr)
                FROM transactions
                WHERE type='pengeluaran'
                GROUP BY category
            """)
        )
        kategori_data = kategori_result.fetchall()
        
        # Recent transactions
        transactions_result = db.session.execute(
            db.text("SELECT * FROM transactions ORDER BY date DESC LIMIT 5")
        )
        transactions = transactions_result.fetchall()
        
        return render_template(
            "dashboard.html",
            total_masuk=total_masuk,
            total_keluar=total_keluar,
            saldo=saldo,
            transactions=transactions,
            kategori_data=kategori_data,
        )
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/add", methods=["POST"])
def add():
    try:
        category = request.form["category"]
        type_trans = request.form["type"]
        amount = float(request.form["amount"])
        currency = request.form["currency"]
        note = request.form["note"]
        date = request.form["date"]
        
        amount_idr = convert_to_idr(amount, currency)
        
        # Insert menggunakan SQLAlchemy
        db.session.execute(
            db.text("""
                INSERT INTO transactions (category, type, amount, currency, amount_idr, note, date)
                VALUES (:category, :type, :amount, :currency, :amount_idr, :note, :date)
            """),
            {
                'category': category,
                'type': type_trans,
                'amount': amount,
                'currency': currency,
                'amount_idr': amount_idr,
                'note': note,
                'date': date
            }
        )
        db.session.commit()
        
        invalidate_analytics_cache()
        return redirect(url_for("dashboard"))
    except Exception as e:
        db.session.rollback()
        return f"Error: {str(e)}", 500

@app.template_filter()
def rupiah(value):
    try:
        value = float(value)
        return f"{value:,.0f}".replace(",", ".")
    except:
        return value

@app.route("/history")
def history():
    try:
        # Get transactions
        transactions_result = db.session.execute(
            db.text("""
                SELECT id, user_id, category, type, amount, currency, amount_idr, note, date
                FROM transactions 
                ORDER BY date DESC
            """)
        )
        transactions = transactions_result.fetchall()
        
        # Data untuk grafik pengeluaran
        pengeluaran_result = db.session.execute(
            db.text("""
                SELECT date, SUM(amount_idr)
                FROM transactions
                WHERE type='pengeluaran'
                GROUP BY date
                ORDER BY date
            """)
        )
        pengeluaran_data = pengeluaran_result.fetchall()
        dates = [row[0].strftime("%Y-%m-%d") for row in pengeluaran_data]
        values = [float(row[1]) for row in pengeluaran_data]
        
        # Data untuk grafik pemasukan
        pemasukan_result = db.session.execute(
            db.text("""
                SELECT date, SUM(amount_idr)
                FROM transactions
                WHERE type='pemasukan'
                GROUP BY date
                ORDER BY date
            """)
        )
        pemasukan_data = pemasukan_result.fetchall()
        dates_income = [row[0].strftime("%Y-%m-%d") for row in pemasukan_data]
        values_income = [float(row[1]) for row in pemasukan_data]
        
        return render_template(
            "history.html",
            transactions=transactions,
            dates=dates,
            values=values,
            dates_income=dates_income,
            values_income=values_income,
        )
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/delete/<int:id>", methods=["GET"])
def delete(id):
    try:
        db.session.execute(
            db.text("DELETE FROM transactions WHERE id=:id"),
            {'id': id}
        )
        db.session.commit()
        
        invalidate_analytics_cache()
        return redirect(url_for("history"))
    except Exception as e:
        db.session.rollback()
        return f"Error: {str(e)}", 500

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):
    try:
        if request.method == "POST":
            category = request.form["category"]
            type_trans = request.form["type"]
            amount = float(request.form["amount"])
            currency = request.form["currency"]
            note = request.form["note"]
            date = request.form["date"]
            
            amount_idr = convert_to_idr(amount, currency)
            
            db.session.execute(
                db.text("""
                    UPDATE transactions SET category=:category, type=:type, amount=:amount, 
                    currency=:currency, amount_idr=:amount_idr, note=:note, date=:date 
                    WHERE id=:id
                """),
                {
                    'category': category,
                    'type': type_trans,
                    'amount': amount,
                    'currency': currency,
                    'amount_idr': amount_idr,
                    'note': note,
                    'date': date,
                    'id': id
                }
            )
            db.session.commit()
            
            invalidate_analytics_cache()
            return redirect(url_for("history"))
        
        # GET request - show edit form
        transaction_result = db.session.execute(
            db.text("SELECT * FROM transactions WHERE id=:id"),
            {'id': id}
        )
        transaction = transaction_result.fetchone()
        
        return render_template("edit.html", transaction=transaction)
    except Exception as e:
        db.session.rollback()
        return f"Error: {str(e)}", 500

@app.route("/analytics", methods=["GET", "POST"])
def analytics():
    """Simplified analytics untuk deployment"""
    try:
        print("🔍 DEBUG: Masuk ke route analytics")
        
        # Get parameters dengan default values
        target_nabung_input = request.args.get("target") or request.form.get("target", "0")
        lifestyle = (request.args.get("lifestyle") or request.form.get("lifestyle") or "moderat").lower()
        
        print(f"🔍 DEBUG: Parameters - target: {target_nabung_input}, lifestyle: {lifestyle}")
        
        # Fetch data dengan error handling
        try:
            summary, today_spending_raw, weekly_spending, kategori_rows = fetch_all_transactions_data()
            print(f"🔍 DEBUG: Data fetched - summary: {summary}")
        except Exception as fetch_error:
            print(f"❌ ERROR in fetch_all_transactions_data: {fetch_error}")
            # Fallback data
            summary = {"pemasukan": 0, "pengeluaran": 0}
            today_spending_raw = {}
            weekly_spending = {}
            kategori_rows = []
        
        # Basic calculations dengan error handling
        try:
            pemasukan = to_dec(summary.get("pemasukan", 0))
            pengeluaran = to_dec(summary.get("pengeluaran", 0))
            sisa = pemasukan - pengeluaran
            
            print(f"🔍 DEBUG: Calculations - pemasukan: {pemasukan}, pengeluaran: {pengeluaran}, sisa: {sisa}")
        except Exception as calc_error:
            print(f"❌ ERROR in calculations: {calc_error}")
            pemasukan = to_dec(0)
            pengeluaran = to_dec(0)
            sisa = to_dec(0)
        
        # Date calculations
        try:
            today = datetime.date.today()
            days_in_month = calendar.monthrange(today.year, today.month)[1]
            current_day = today.day
            days_remaining = days_in_month - current_day
        except:
            days_in_month = 30
            current_day = 1
            days_remaining = 29
        
        # Target nabung
        try:
            target_nabung = Decimal(str(target_nabung_input or "0"))
        except:
            target_nabung = Decimal("0")
        
        if target_nabung == 0 and pemasukan > 0:
            target_nabung = (pemasukan * Decimal('0.15')).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        # ML Prediction dengan error handling
        try:
            income_idr = float(pemasukan)
        except:
            income_idr = 0.0
        
        try:
            ml_budget = recommend_budget_ml_cached(income_idr, lifestyle)
            print(f"🔍 DEBUG: ML budget generated - items: {len(ml_budget)}")
        except Exception as ml_error:
            print(f"❌ ERROR in ML prediction: {ml_error}")
            ml_budget = build_fixed_recommendation(income_idr, lifestyle)
        
        # === VARIABLES UNTUK TEMPLATE ===
        
        # 1. daily_status
        try:
            # Hitung budget harian
            daily_budget_limit = float((pemasukan - pengeluaran - target_nabung) / Decimal(days_remaining)) if days_remaining > 0 else 0
            
            # Hitung pengeluaran hari ini
            today_expense_result = db.session.execute(
                db.text("SELECT COALESCE(SUM(amount_idr), 0) FROM transactions WHERE type='pengeluaran' AND date = :today"),
                {'today': today}
            )
            today_spent = float(today_expense_result.scalar() or 0)
            
            # Hitung sisa budget hari ini
            daily_remaining = daily_budget_limit - today_spent
            
            # Tentukan status
            if today_spent == 0:
                daily_status_value = 'success'
            elif today_spent <= daily_budget_limit * 0.7:
                daily_status_value = 'success'
            elif today_spent <= daily_budget_limit:
                daily_status_value = 'warning'
            else:
                daily_status_value = 'danger'
            
            # Hitung persentase
            daily_percentage = (today_spent / daily_budget_limit * 100) if daily_budget_limit > 0 else 0
            
            daily_status = {
                'date': today.strftime("%d %B %Y"),
                'budget_limit': daily_budget_limit,
                'spent': today_spent,
                'remaining': daily_remaining,
                'status': daily_status_value,
                'percentage': min(100, daily_percentage)
            }
        except Exception as daily_error:
            print(f"❌ ERROR in daily_status calculation: {daily_error}")
            daily_status = {
                'date': datetime.date.today().strftime("%d %B %Y"),
                'budget_limit': 0,
                'spent': 0,
                'remaining': 0,
                'status': 'success',
                'percentage': 0
            }
        
        # 2. weekly_progress
        try:
            # Hitung minggu ke berapa
            week_num = (current_day - 1) // 7 + 1
            
            # Hitung target mingguan (25% dari pemasukan bulanan)
            weekly_target = float(pemasukan * Decimal('0.25'))
            
            # Hitung pengeluaran minggu ini
            start_of_week = today - datetime.timedelta(days=today.weekday())
            end_of_week = start_of_week + datetime.timedelta(days=6)
            
            weekly_expense_result = db.session.execute(
                db.text("""
                    SELECT COALESCE(SUM(amount_idr), 0) 
                    FROM transactions 
                    WHERE type = 'pengeluaran' 
                    AND date >= :start_date 
                    AND date <= :end_date
                """),
                {'start_date': start_of_week, 'end_date': end_of_week}
            )
            weekly_spending = float(weekly_expense_result.scalar() or 0)
            
            # Hitung sisa budget mingguan
            weekly_remaining = weekly_target - weekly_spending
            
            # Tentukan status
            if weekly_spending == 0:
                weekly_status = 'success'
            elif weekly_spending <= weekly_target * 0.7:
                weekly_status = 'success'
            elif weekly_spending <= weekly_target:
                weekly_status = 'warning'
            else:
                weekly_status = 'danger'
            
            # Hitung persentase
            weekly_percentage = (weekly_spending / weekly_target * 100) if weekly_target > 0 else 0
            
            weekly_progress = {
                'week_num': week_num,
                'target': weekly_target,
                'spending': weekly_spending,
                'remaining': weekly_remaining,
                'status': weekly_status,
                'percentage': min(100, weekly_percentage)
            }
        except Exception as weekly_error:
            print(f"❌ ERROR in weekly_progress calculation: {weekly_error}")
            weekly_progress = {
                'week_num': 1,
                'target': 0,
                'spending': 0,
                'remaining': 0,
                'status': 'success',
                'percentage': 0
            }
        
        # 3. emergency_warning (jika diperlukan)
        emergency_warning = None
        if pemasukan > 0 and pengeluaran > 0:
            spending_ratio = float(pengeluaran / pemasukan)
            if spending_ratio > 0.9:
                daily_limit = float((pemasukan - pengeluaran - target_nabung) / Decimal(days_remaining)) if days_remaining > 0 else 0
                emergency_warning = {
                    'level': 'critical',
                    'message': 'Pengeluaran sudah mencapai 90% dari pemasukan!',
                    'daily_limit': daily_limit
                }
            elif spending_ratio > 0.7:
                daily_limit = float((pemasukan - pengeluaran - target_nabung) / Decimal(days_remaining)) if days_remaining > 0 else 0
                emergency_warning = {
                    'level': 'warning',
                    'message': 'Pengeluaran sudah mencapai 70% dari pemasukan.',
                    'daily_limit': daily_limit
                }
        
        # 4. prediction_message & prediction_status
        prediction_message = None
        prediction_status = None
        if weekly_progress['spending'] > 0:
            weekly_spending_rate = weekly_progress['spending'] / (current_day / 7)
            monthly_projection = weekly_spending_rate * 4
            
            if monthly_projection > float(pemasukan) * 0.9:
                prediction_status = 'Boros'
                prediction_message = 'Pola spending minggu ini menunjukkan potensi over budget bulanan'
            elif monthly_projection > float(pemasukan) * 0.7:
                prediction_status = 'Waspada'
                prediction_message = 'Pola spending masih dalam batas wajar, tapi perlu diperhatikan'
            else:
                prediction_status = 'Aman'
                prediction_message = 'Pola spending minggu ini sehat dan terkendali'
        
        # 5. budget_recommendations & total_saving_potential - FIXED VERSION
        budget_recommendations = []
        total_saving_potential = 0
        
        # Hanya generate budget recommendations jika ada pemasukan
        if float(pemasukan) > 0:
            # Definisikan kategori dan persentase ideal berdasarkan ML budget
            kategori_ideal = {}
            for item in ml_budget:
                category_name = item['category']
                if category_name == 'Food':
                    kategori_ideal['makanan/minuman'] = item['fraction']
                elif category_name == 'Transport':
                    kategori_ideal['transportasi'] = item['fraction']
                elif category_name == 'Entertainment':
                    kategori_ideal['hiburan'] = item['fraction']
                elif category_name == 'Laundry':
                    kategori_ideal['laundry'] = item['fraction']
                elif category_name == 'Savings':
                    # Skip savings karena sudah ada di target nabung
                    continue
                else:
                    # Untuk kategori lainnya dari ML
                    kategori_ideal[category_name.lower()] = item['fraction']
            
            # Tambahkan kategori default jika ML tidak memberikan semua
            if 'makanan/minuman' not in kategori_ideal:
                kategori_ideal['makanan/minuman'] = 0.25
            if 'transportasi' not in kategori_ideal:
                kategori_ideal['transportasi'] = 0.15
            if 'hiburan' not in kategori_ideal:
                kategori_ideal['hiburan'] = 0.10
            if 'laundry' not in kategori_ideal:
                kategori_ideal['laundry'] = 0.05
            
            # Ambil data pengeluaran per kategori bulan ini
            kategori_pengeluaran = {}
            try:
                first_day = today.replace(day=1)
                if today.month == 12:
                    last_day = today.replace(year=today.year+1, month=1, day=1) - datetime.timedelta(days=1)
                else:
                    last_day = today.replace(month=today.month+1, day=1) - datetime.timedelta(days=1)
                
                kategori_result = db.session.execute(
                    db.text("""
                        SELECT category, COALESCE(SUM(amount_idr), 0) as total
                        FROM transactions
                        WHERE type = 'pengeluaran' 
                        AND date >= :first_day AND date <= :last_day
                        GROUP BY category
                    """),
                    {'first_day': first_day, 'last_day': last_day}
                )
                
                for row in kategori_result:
                    kategori_name = row[0].lower()
                    kategori_pengeluaran[kategori_name] = float(row[1])
                    print(f"🔍 DEBUG: Kategori {kategori_name}: Rp {row[1]}")
                    
            except Exception as e:
                print(f"❌ ERROR fetching kategori pengeluaran: {e}")
            
            # Generate recommendations untuk setiap kategori
            for kategori, persentase_ideal in kategori_ideal.items():
                ideal_bulanan = float(pemasukan) * persentase_ideal
                ideal_harian = ideal_bulanan / days_in_month
                ideal_mingguan = ideal_bulanan / 4
                
                # Pengeluaran aktual bulan ini
                pengeluaran_aktual = kategori_pengeluaran.get(kategori, 0)
                pengeluaran_harian_aktual = pengeluaran_aktual / current_day if current_day > 0 else 0
                
                # Hitung sisa budget
                sisa_budget = ideal_bulanan - pengeluaran_aktual
                sisa_harian = sisa_budget / days_remaining if days_remaining > 0 else 0
                
                # Tentukan status
                if pengeluaran_aktual == 0:
                    status = 'success'
                elif pengeluaran_aktual <= ideal_bulanan * 0.7:
                    status = 'success'
                elif pengeluaran_aktual <= ideal_bulanan:
                    status = 'warning'
                else:
                    status = 'over'
                
                # Hitung potensi hemat
                potensi_hemat = max(0, pengeluaran_aktual - ideal_bulanan)
                total_saving_potential += potensi_hemat
                
                # Pengeluaran hari ini untuk kategori ini
                pengeluaran_hari_ini = 0
                try:
                    today_category_result = db.session.execute(
                        db.text("""
                            SELECT COALESCE(SUM(amount_idr), 0)
                            FROM transactions
                            WHERE type = 'pengeluaran' 
                            AND LOWER(category) = :category
                            AND date = :today
                        """),
                        {'category': kategori, 'today': today}
                    )
                    pengeluaran_hari_ini = float(today_category_result.scalar() or 0)
                except Exception as e:
                    print(f"❌ ERROR fetching today's expense for {kategori}: {e}")
                
                # Status hari ini
                selisih_hari_ini = pengeluaran_hari_ini - ideal_harian
                if pengeluaran_hari_ini == 0:
                    status_hari_ini = 'success'
                elif pengeluaran_hari_ini <= ideal_harian:
                    status_hari_ini = 'success'
                elif pengeluaran_hari_ini <= ideal_harian * 1.3:
                    status_hari_ini = 'warning'
                else:
                    status_hari_ini = 'over'
                
                # Tips berdasarkan kategori
                tips = []
                if kategori == 'makanan/minuman':
                    tips = ["Bawa bekal dari kos", "Masak sendiri lebih hemat", "Manfaatkan promo makanan"]
                elif kategori == 'transportasi':
                    tips = ["Gunakan transportasi umum", "Berkelompok untuk bagi ongkos", "Plan rute perjalanan"]
                elif kategori == 'hiburan':
                    tips = ["Cari hiburan gratis", "Manfaatkan diskon mahasiswa", "Batasi hangout mahal"]
                elif kategori == 'laundry':
                    tips = ["Cuci baju sendiri jika memungkinkan", "Kumpulkan baju kotor untuk cuci sekaligus"]
                elif kategori == 'belanja':
                    tips = ["Buat list belanja", "Tunggu diskon akhir bulan", "Prioritaskan kebutuhan"]
                
                # Format nama kategori untuk display
                display_category = kategori.title().replace('/', ' / ')
                
                budget_recommendations.append({
                    'category': display_category,
                    'current_total': pengeluaran_aktual,
                    'current_daily': pengeluaran_harian_aktual,
                    'ideal_monthly': ideal_bulanan,
                    'ideal_daily': ideal_harian,
                    'ideal_weekly': ideal_mingguan,
                    'percentage': persentase_ideal * 100,
                    'remaining_budget': sisa_budget,
                    'remaining_daily': sisa_harian,
                    'status': status,
                    'today_amount': pengeluaran_hari_ini,
                    'today_diff': selisih_hari_ini,
                    'today_status': status_hari_ini,
                    'saving_potential': potensi_hemat,
                    'tips': tips
                })
            
            print(f"🔍 DEBUG: Generated {len(budget_recommendations)} budget recommendations")
        
        # 6. user_recommendation (dari ML)
        user_recommendation = None
        if income_idr > 0:
            # Buat user_recommendation dari ML budget
            lifestyle_descriptions = {
                'hemat': 'Mode hemat fokus pada penghematan maksimal dan prioritas kebutuhan pokok',
                'moderat': 'Mode moderat menyeimbangkan antara kebutuhan dan keinginan dengan bijak', 
                'aktif': 'Mode aktif memberikan ruang lebih untuk pengembangan diri dan sosial'
            }
            
            lifestyle_tips = {
                'hemat': [
                    'Prioritaskan kebutuhan pokok: makan, transport, tagihan',
                    'Batasi makan di luar maksimal 2x seminggu',
                    'Manfaatkan promo dan diskon mahasiswa',
                    'Bawa bekal dari kos untuk hemat uang makan'
                ],
                'moderat': [
                    'Alokasikan 30% untuk hiburan dan sosial',
                    'Tetap prioritaskan tabungan 20% dari pemasukan',
                    'Plan ahead untuk kegiatan bulanan',
                    'Sisihkan dana untuk self-improvement'
                ],
                'aktif': [
                    'Investasi dalam skill development',
                    'Network building dengan teman dan komunitas',
                    'Eksplorasi kegiatan baru yang produktif',
                    'Tetap monitor pengeluaran sosial'
                ]
            }
            
            user_recommendation = {
                'lifestyle': lifestyle,
                'income': income_idr,
                'recommendations': ml_budget,
                'description': lifestyle_descriptions.get(lifestyle, 'Mode pengelolaan keuangan personal'),
                'tips': lifestyle_tips.get(lifestyle, [])
            }
        
        print("🔍 DEBUG: Berhasil memproses semua data, menuju render template")
        print(f"🔍 DEBUG: budget_recommendations count: {len(budget_recommendations)}")
        print(f"🔍 DEBUG: total_saving_potential: {total_saving_potential}")
        print(f"🔍 DEBUG: user_recommendation: {user_recommendation is not None}")
        
        return render_template("analytics.html",
            pemasukan=float(pemasukan),
            pengeluaran=float(pengeluaran),
            sisa=float(sisa),
            target_nabung=float(target_nabung),
            ml_budget=ml_budget,
            income_idr=income_idr,
            lifestyle=lifestyle,
            days_remaining=days_remaining,
            current_day=current_day,
            days_in_month=days_in_month,
            
            # Variabel-variabel baru untuk template
            daily_status=daily_status,
            weekly_progress=weekly_progress,
            emergency_warning=emergency_warning,
            prediction_message=prediction_message,
            prediction_status=prediction_status,
            budget_recommendations=budget_recommendations,
            total_saving_potential=total_saving_potential,
            user_recommendation=user_recommendation
        )
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in analytics route: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error in analytics: {str(e)}", 500
        
@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route('/test-db')
def test_db():
    try:
        db.engine.connect()
        return "✅ MySQL Connected Successfully!"
    except Exception as e:
        return f"❌ Database Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
