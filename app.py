from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
from dotenv import load_dotenv
from math import ceil
from app import app  
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

# Load model .joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")

MODEL_PATH = os.path.join(ML_DIR, "edubadget_gb_multi.joblib")
SCALER_PATH = os.path.join(ML_DIR, "edubadget_income_scaler.joblib")
BASELINE_PATH = os.path.join(ML_DIR, "asia_baseline_frac.json")

# ===== LAZY LOAD ML MODEL (Load sekali saja) =====
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
# Letakkan SEBELUM app = Flask(__name__)

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

load_dotenv()
API_KEY = os.getenv("EXCHANGE_API_KEY")

app = Flask(__name__)
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "edubudget"

mysql = MySQL(app)

# ✅ REGISTER EMOJI FUNCTION SETELAH app DIBUAT
app.jinja_env.globals.update(get_emoji=get_category_emoji)

# ===== SETUP REDIS =====
try:
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
        return amount
    
    # Cache exchange rate untuk 1 jam
    cache_key = f"exchange_rate:{currency}:IDR"
    if REDIS_AVAILABLE:
        try:
            cached_rate = redis_client.get(cache_key)
            if cached_rate:
                return amount * float(cached_rate)
        except:
            pass
    
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{currency}"
    response = requests.get(url).json()
    rate = response["conversion_rates"]["IDR"]
    
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(cache_key, 3600, str(rate))  # Cache 1 jam
        except:
            pass
    
    return amount * rate

def recommend_budget_ml_cached(income_idr, lifestyle):
    """
    ML prediction dengan CACHING - tidak compute ulang untuk income & lifestyle yang sama
    """
    # Generate cache key berdasarkan input
    cache_key = f"ml_prediction:{int(income_idr)}:{lifestyle}"
    
    # Try get from cache
    if REDIS_AVAILABLE:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                print(f"✓ ML Cache hit: {cache_key}")
                return json.loads(cached)
        except:
            pass
    
    # Compute prediction
    ml_model, scaler, baseline_frac = get_ml_model()
    
    if ml_model is None or scaler is None:
        result = build_fixed_recommendation(income_idr, lifestyle)
    else:
        try:
            # ===== FIX WARNING SKLEARN: Buat DataFrame dengan feature names =====
            import pandas as pd
            X = pd.DataFrame([[income_idr]], columns=['income'])
            
            # Normalisasi income
            scaled_income = scaler.transform(X)
            
            # Prediksi fraksi kategori utama (SEKALI SAJA!)
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
            
            # Flexible category (20% fixed)
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
    
    # Save to cache (24 jam)
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

# ===== ULTRA OPTIMIZED: Index-Friendly Queries =====
def fetch_all_transactions_data(cur):
    """
    Ultra-optimized version using index-friendly queries
    Return: (summary, today_spending, weekly_data, kategori_rows)
    """
    import time
    start_time = time.time()
    
    summary = {}
    today_spending = {}
    weekly_data = {}
    kategori_rows = []
    
    # Get current month boundaries (untuk memanfaatkan index pada kolom date)
    today = datetime.date.today()
    first_day = today.replace(day=1)
    if today.month == 12:
        last_day = today.replace(year=today.year+1, month=1, day=1) - datetime.timedelta(days=1)
    else:
        last_day = today.replace(month=today.month+1, day=1) - datetime.timedelta(days=1)
    
    try:
        # Query 1: Summary (pemasukan & pengeluaran) - INDEX OPTIMIZED
        cur.execute("""
            SELECT type, COALESCE(SUM(amount_idr), 0) as total
            FROM transactions
            WHERE date >= %s AND date <= %s
            GROUP BY type
        """, (first_day, last_day))
        
        for row in cur.fetchall():
            summary[row[0]] = to_dec(row[1])
        
        print(f"✓ Query 1 (Summary): {time.time() - start_time:.3f}s")
        
        # Query 2: Today's spending - INDEX OPTIMIZED
        cur.execute("""
            SELECT category, COALESCE(SUM(amount_idr), 0) as total
            FROM transactions
            WHERE type = 'pengeluaran' AND date = %s
            GROUP BY category
        """, (today,))
        
        for row in cur.fetchall():
            if row[0]:
                today_spending[row[0]] = to_dec(row[1])
        
        print(f"✓ Query 2 (Today): {time.time() - start_time:.3f}s")
        
        # Query 3: Weekly spending - SIMPLIFIED
        cur.execute("""
            SELECT 
                CEIL(DATEDIFF(date, %s) / 7) + 1 as week_num,
                COALESCE(SUM(amount_idr), 0) as total,
                COUNT(DISTINCT date) as days_count
            FROM transactions
            WHERE type = 'pengeluaran' 
              AND date >= %s AND date <= %s
            GROUP BY week_num
        """, (first_day, first_day, last_day))
        
        for row in cur.fetchall():
            week_num = int(row[0]) if row[0] else 0
            if week_num > 0 and week_num <= 5:
                weekly_data[week_num] = {
                    'total': to_dec(row[1]),
                    'days': int(row[2]) if row[2] else 0
                }
        
        print(f"✓ Query 3 (Weekly): {time.time() - start_time:.3f}s")
        
        # Query 4: Category totals - INDEX OPTIMIZED
        cur.execute("""
            SELECT 
                category,
                COALESCE(SUM(amount_idr), 0) as total,
                COUNT(*) as tx_count,
                COALESCE(AVG(amount_idr), 0) as avg_tx
            FROM transactions
            WHERE type = 'pengeluaran' 
              AND date >= %s AND date <= %s
            GROUP BY category
        """, (first_day, last_day))
        
        for row in cur.fetchall():
            if row[0]:
                kategori_rows.append((
                    row[0],  # category
                    to_dec(row[1]),  # total amount
                    int(row[2]) if row[2] else 0,  # tx_count
                    None,  # days_with_tx (placeholder)
                    to_dec(row[3])  # avg_tx
                ))
        
        print(f"✓ Query 4 (Categories): {time.time() - start_time:.3f}s")
        print(f"✅ Total fetch time: {time.time() - start_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Database error in fetch_all_transactions_data: {e}")
        import traceback
        traceback.print_exc()
    
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
    cur = mysql.connection.cursor()
    
    cur.execute("SELECT SUM(amount_idr) FROM transactions WHERE type='pemasukan'")
    total_masuk = cur.fetchone()[0] or 0
    
    cur.execute("SELECT SUM(amount_idr) FROM transactions WHERE type='pengeluaran'")
    total_keluar = cur.fetchone()[0] or 0
    
    saldo = total_masuk - total_keluar
    
    cur.execute("""
        SELECT category, SUM(amount_idr)
        FROM transactions
        WHERE type='pengeluaran'
        GROUP BY category
    """)
    kategori_data = cur.fetchall()
    
    cur.execute("SELECT * FROM transactions ORDER BY date DESC LIMIT 5")
    transactions = cur.fetchall()
    
    cur.close()
    
    return render_template(
        "dashboard.html",
        total_masuk=total_masuk,
        total_keluar=total_keluar,
        saldo=saldo,
        transactions=transactions,
        kategori_data=kategori_data,
    )

@app.route("/add", methods=["POST"])
def add():
    category = request.form["category"]
    type_trans = request.form["type"]
    amount = float(request.form["amount"])
    currency = request.form["currency"]
    note = request.form["note"]
    date = request.form["date"]
    
    amount_idr = convert_to_idr(amount, currency)
    
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO transactions (category, type, amount, currency, amount_idr, note, date)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (category, type_trans, amount, currency, amount_idr, note, date))
    mysql.connection.commit()
    cur.close()
    
    invalidate_analytics_cache()
    return redirect(url_for("dashboard"))

@app.template_filter()
def rupiah(value):
    try:
        value = float(value)
        return f"{value:,.0f}".replace(",", ".")
    except:
        return value

@app.route("/history")
def history():
    cur = mysql.connection.cursor()
    
    # ✅ PERBAIKAN: Ambil kolom dengan urutan yang jelas
    cur.execute("""
        SELECT id, user_id, category, type, amount, currency, amount_idr, note, date
        FROM transactions 
        ORDER BY date DESC
    """)
    transactions = cur.fetchall()
    
    # Data untuk grafik pengeluaran
    cur.execute("""
        SELECT date, SUM(amount_idr)
        FROM transactions
        WHERE type='pengeluaran'
        GROUP BY date
        ORDER BY date
    """)
    pengeluaran_data = cur.fetchall()
    dates = [row[0].strftime("%Y-%m-%d") for row in pengeluaran_data]
    values = [float(row[1]) for row in pengeluaran_data]
    
    # Data untuk grafik pemasukan
    cur.execute("""
        SELECT date, SUM(amount_idr)
        FROM transactions
        WHERE type='pemasukan'
        GROUP BY date
        ORDER BY date
    """)
    pemasukan_data = cur.fetchall()
    dates_income = [row[0].strftime("%Y-%m-%d") for row in pemasukan_data]
    values_income = [float(row[1]) for row in pemasukan_data]
    
    cur.close()
    
    return render_template(
        "history.html",
        transactions=transactions,
        dates=dates,
        values=values,
        dates_income=dates_income,
        values_income=values_income,
    )

@app.route("/delete/<int:id>", methods=["GET"])
def delete(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM transactions WHERE id=%s", (id,))
    mysql.connection.commit()
    cur.close()
    
    invalidate_analytics_cache()
    return redirect(url_for("history"))

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):
    cur = mysql.connection.cursor()
    
    if request.method == "POST":
        category = request.form["category"]
        type_trans = request.form["type"]
        amount = float(request.form["amount"])
        currency = request.form["currency"]
        note = request.form["note"]
        date = request.form["date"]
        
        amount_idr = convert_to_idr(amount, currency)
        
        cur.execute("""
            UPDATE transactions SET category=%s, type=%s, amount=%s, currency=%s, amount_idr=%s, note=%s, date=%s 
            WHERE id=%s
        """, (category, type_trans, amount, currency, amount_idr, note, date, id))
        
        mysql.connection.commit()
        cur.close()
        
        invalidate_analytics_cache()
        return redirect(url_for("history"))
    
    cur.execute("SELECT * FROM transactions WHERE id=%s", (id,))
    transaction = cur.fetchone()
    cur.close()
    
    return render_template("edit.html", transaction=transaction)

@app.route("/analytics", methods=["GET", "POST"])
def analytics():
    """
    OPTIMIZED VERSION - No form submission dialog
    """
    import time
    route_start = time.time()
    
    # Get parameters (dari GET atau POST)
    if request.method == "POST":
        target_nabung_input = request.form.get("target")
        lifestyle = request.form.get("lifestyle", "moderat").lower()
        # Redirect to GET untuk menghindari re-submit
        return redirect(url_for('analytics', target=target_nabung_input, lifestyle=lifestyle))
    
    # GET request processing
    target_nabung_input = request.args.get("target")
    lifestyle = (request.args.get("lifestyle") or "moderat").lower()
    
    print("\n" + "="*50)
    print(f"🕐 Analytics route started at {datetime.datetime.now()}")
    print(f"📊 Params: target={target_nabung_input}, lifestyle={lifestyle}")
    
    # Cache key
    user_id = session.get('user_id', 'anonymous')
    today = datetime.date.today()
    cache_params = f"{target_nabung_input}:{lifestyle}"
    cache_key = f"analytics_data:v4:{user_id}:{today}:{hashlib.md5(cache_params.encode()).hexdigest()}"
    
    # Try get from cache
    if REDIS_AVAILABLE:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                print(f"✅ Cache HIT! Returned in {time.time() - route_start:.3f}s")
                print("="*50 + "\n")
                data = json.loads(cached)
                return render_template("analytics.html", **data)
        except Exception as e:
            print(f"⚠️ Cache error: {e}")
    
    print(f"⏳ Cache MISS - Computing analytics...")
    
    # === FETCH DATA ===
    fetch_start = time.time()
    cur = mysql.connection.cursor()
    summary, today_spending_raw, weekly_spending, kategori_rows = fetch_all_transactions_data(cur)
    cur.close()
    print(f"✓ Database fetch: {time.time() - fetch_start:.3f}s")
    
    # === CALCULATIONS ===
    calc_start = time.time()
    
    pemasukan = to_dec(summary.get("pemasukan", 0))
    pengeluaran = to_dec(summary.get("pengeluaran", 0))
    sisa = pemasukan - pengeluaran
    
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    current_day = today.day
    current_week = math.ceil(current_day / 7)
    days_remaining = days_in_month - current_day
    
    today_spending = today_spending_raw
    today_spending_total = sum(today_spending.values()) if today_spending else Decimal('0')
    
    # TARGET NABUNG
    try:
        target_nabung = Decimal(str(target_nabung_input or "0"))
    except:
        target_nabung = Decimal("0")
    
    if target_nabung == 0 and pemasukan > 0:
        target_nabung = (pemasukan * Decimal('0.15')).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    
    print(f"✓ Basic calculations: {time.time() - calc_start:.3f}s")
    
    # === ML PREDICTION ===
    ml_start = time.time()
    try:
        income_idr = float(pemasukan)
    except:
        income_idr = 0.0
    
    ml_budget = None
    try:
        ml_budget = recommend_budget_ml_cached(income_idr, lifestyle)
        for it in ml_budget:
            it["fraction"] = float(it.get("fraction",0))
            it["amount"] = int(round(float(it.get("amount",0))))
    except Exception as e:
        print(f"⚠️ ML error: {e}")
        ml_budget = build_fixed_recommendation(income_idr, lifestyle)
    
    print(f"✓ ML prediction: {time.time() - ml_start:.3f}s")
    
    # === STUDENT CATEGORIES & RECOMMENDATIONS ===
    process_start = time.time()
    
    student_categories = {
        'makanan': ['makanan','minuman','makan','jajan','snack','minum','kopi','ngopi','cafe','food','resto','warteg','kantin'],
        'transport': ['transport','bensin','ojol','grab','gojek','parkir','angkot','motor','taxi','bis'],
        'laundry': ['laundry','cuci'],
        'hiburan': ['hiburan','nonton','game','streaming','spotify','netflix','kongko','hangout','main','bioskop'],
        'belanja': ['belanja','shopping','baju','sepatu','fashion','pakaian'],
        'akademik': ['kuliah','buku','fotokopi','print','alat tulis','praktikum','tugas','ujian'],
        'kos': ['kos','sewa','kontrakan','boarding'],
        'internet': ['internet','pulsa','paket data','wifi','kuota'],
        'kesehatan': ['obat','dokter','vitamin','kesehatan','klinik','medis'],
        'lainnya': []
    }
    
    student_spending = defaultdict(lambda: {'total': Decimal('0'), 'count': 0, 'days_active': 0, 'items': [], 'today': Decimal('0')})
    
    for cat_name, total, tx_count, days_with_tx, avg_tx in kategori_rows:
        amt = to_dec(total)
        sc = categorize_student(cat_name, student_categories)
        sd = student_spending[sc]
        sd['total'] += amt
        sd['count'] += int(tx_count or 0)
        sd['days_active'] = max(sd['days_active'], int(days_with_tx or 1))
        sd['items'].append(cat_name)
        if cat_name in today_spending:
            sd['today'] += today_spending[cat_name]
    
    # Budget ideal
    budget_ideal_mahasiswa = {
        'makanan': Decimal('0.40'),
        'transport': Decimal('0.12'),
        'internet': Decimal('0.07'),
        'akademik': Decimal('0.08'),
        'hiburan': Decimal('0.08'),
        'laundry': Decimal('0.05'),
        'belanja': Decimal('0.05'),
        'tabungan': (target_nabung / pemasukan) if pemasukan > 0 else Decimal('0.15'),
    }
    
    # Predictions
    week_spending_factor = {1: Decimal('1.3'), 2: Decimal('1.1'), 3: Decimal('0.9'), 4: Decimal('0.7')}
    completed_weeks = [w for w in weekly_spending.keys() if w < current_week]
    
    if completed_weeks:
        avg_weekly_past = sum(weekly_spending[w]['total'] for w in completed_weeks) / Decimal(str(len(completed_weeks)))
    else:
        avg_weekly_past = (pengeluaran / Decimal(str(max(1, current_week)))) if pemasukan > 0 else Decimal('0')
    
    remaining_weeks = math.ceil(days_remaining / 7)
    predicted_future_spending = Decimal('0')
    
    for future_week in range(current_week + 1, min(current_week + remaining_weeks + 1, 5)):
        predicted_future_spending += avg_weekly_past * week_spending_factor.get(future_week, Decimal('0.7'))
    
    predicted_total_spending = (pengeluaran + predicted_future_spending).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    predicted_sisa = (pemasukan - predicted_total_spending).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    
    print(f"✓ Predictions & categorization: {time.time() - process_start:.3f}s")
    
    # === BUILD RECOMMENDATIONS ===
    rec_start = time.time()
    
    budget_recommendations = []
    total_saving_potential = Decimal('0')
    
    for student_cat in ['makanan','transport','internet','akademik','hiburan','laundry','belanja']:
        data = student_spending.get(student_cat, None)
        if not data or data['total'] == 0:
            continue
        
        current_total = data['total']
        current_daily = (current_total / Decimal(str(current_day))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        ideal_percentage = budget_ideal_mahasiswa.get(student_cat, Decimal('0.05'))
        ideal_monthly = (pemasukan * ideal_percentage).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        ideal_daily = (ideal_monthly / Decimal(str(days_in_month))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        ideal_weekly = (ideal_daily * Decimal('7')).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        remaining_budget = (ideal_monthly - current_total).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        remaining_daily = (remaining_budget / Decimal(str(max(1, days_remaining)))).quantize(Decimal('1'), rounding=ROUND_HALF_UP) if days_remaining > 0 else Decimal('0')
        
        today_amount = data['today']
        today_status = 'success' if today_amount <= ideal_daily else 'warning' if today_amount <= ideal_daily * Decimal('1.2') else 'danger'
        today_diff = (today_amount - ideal_daily).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        if current_total > ideal_monthly:
            saving_potential = (current_total - ideal_monthly).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            total_saving_potential += saving_potential
            status = 'over'
        elif current_total > ideal_monthly * Decimal('0.9'):
            saving_potential = Decimal('0')
            status = 'warning'
        else:
            saving_potential = Decimal('0')
            status = 'good'
        
        tips = []
        if student_cat == 'makanan' and status in ['over','warning']:
            tips = ["Warteg/kantin kampus: ~Rp 12-15k/porsi","Masak sendiri: ~Rp 8-10k/porsi"]
        elif student_cat == 'transport' and status in ['over','warning']:
            tips = ["Ojol pool", "Jalan kaki/sepeda <2km"]
        
        budget_recommendations.append({
            'category': student_cat.title(),
            'current_total': float(current_total),
            'current_daily': float(current_daily),
            'ideal_monthly': float(ideal_monthly),
            'ideal_daily': float(ideal_daily),
            'ideal_weekly': float(ideal_weekly),
            'remaining_budget': float(remaining_budget),
            'remaining_daily': float(remaining_daily),
            'today_amount': float(today_amount),
            'today_status': today_status,
            'today_diff': float(today_diff),
            'saving_potential': float(saving_potential),
            'status': status,
            'tips': tips,
            'percentage': float(ideal_percentage * 100)
        })
    
    budget_recommendations.sort(key=lambda x: (x['status'] == 'over', x['saving_potential']), reverse=True)
    
    print(f"✓ Recommendations built: {time.time() - rec_start:.3f}s")
    
    # === REMAINING CALCULATIONS ===
    final_start = time.time()
    
    # Weekly progress
    current_week_data = weekly_spending.get(current_week, {'total': Decimal('0'), 'days': 0})
    current_week_spending = current_week_data['total']
    
    if target_nabung > 0:
        total_budget_for_spending = pemasukan - target_nabung
        weeks_in_month = Decimal(str(days_in_month)) / Decimal('7')
        ideal_weekly_budget = (total_budget_for_spending / weeks_in_month).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    else:
        ideal_weekly_budget = (pemasukan * Decimal('0.25')).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    
    weekly_progress = {
        'week_num': current_week,
        'spending': float(current_week_spending),
        'target': float(ideal_weekly_budget),
        'remaining': float(ideal_weekly_budget - current_week_spending),
        'percentage': int((current_week_spending / ideal_weekly_budget * Decimal('100')).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) if ideal_weekly_budget > 0 else 0,
        'status': 'success' if current_week_spending <= ideal_weekly_budget else 'warning' if current_week_spending <= ideal_weekly_budget * Decimal('1.1') else 'danger'
    }
    
    # Daily budget
    if pemasukan > 0:
        if days_remaining > 0 and target_nabung > 0:
            remaining_budget_for_spending = sisa - target_nabung
            daily_budget_limit = (remaining_budget_for_spending / Decimal(str(days_remaining))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        else:
            daily_budget_limit = (sisa / Decimal(str(max(1, days_remaining)))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    else:
        daily_budget_limit = Decimal('0')
    
    daily_status = {
        'date': today.strftime('%A, %d %b %Y'),
        'budget_limit': float(daily_budget_limit),
        'spent': float(today_spending_total),
        'remaining': float(daily_budget_limit - today_spending_total),
        'percentage': int((today_spending_total / daily_budget_limit * Decimal('100')).quantize(Decimal('1'), rounding=ROUND_HALF_UP)) if daily_budget_limit > 0 else 0,
        'status': 'success' if today_spending_total <= daily_budget_limit else 'warning' if today_spending_total <= daily_budget_limit * Decimal('1.2') else 'danger'
    }
    
    # Lifestyle tips
    tips_bank = {
    "hemat": {
        "title": "🟦 Lifestyle: Hemat",
        "desc": "Gaya hidup minimalis yang fokus pada pengeluaran efisien. Cocok untuk mahasiswa yang ingin memaksimalkan tabungan dengan budget terbatas.",
        "tips": [
            "Prioritaskan kebutuhan pokok seperti makanan bergizi dan transportasi",
            "Manfaatkan fasilitas kampus (kantin, perpustakaan, WiFi gratis)",
            "Hindari impulse buying dan selalu buat daftar belanja",
            "Sisihkan minimal 20-25% untuk tabungan dan dana darurat",
            "Cari side hustle ringan untuk tambahan income (freelance, tutor, dll)"
        ]
    },
    "moderat": {
        "title": "🟩 Lifestyle: Moderat",
        "desc": "Gaya hidup seimbang antara kebutuhan dan keinginan. Ideal untuk mahasiswa yang ingin menikmati kehidupan kampus sambil tetap menabung.",
        "tips": [
            "Terapkan aturan 50/30/20: 50% kebutuhan, 30% keinginan, 20% tabungan",
            "Sisihkan tabungan 10-15% di awal bulan sebelum dipakai",
            "Batasi hangout/nongkrong maksimal 2-3x per minggu",
            "Manfaatkan promo dan diskon khusus mahasiswa (streaming, transport, F&B)",
            "Review pengeluaran setiap minggu untuk evaluasi budget",
            "Sisakan dana fleksibel untuk kebutuhan mendadak"
        ]
    },
    "aktif": {
        "title": "🟧 Lifestyle: Aktif",
        "desc": "Gaya hidup dinamis dengan banyak aktivitas sosial dan pengembangan diri. Cocok untuk mahasiswa yang aktif berorganisasi dan networking.",
        "tips": [
            "Alokasikan 10-20% khusus untuk pengembangan diri (kursus, buku, seminar)",
            "Gunakan aplikasi budget tracker untuk monitoring real-time",
            "Cari event/workshop gratis atau berbayar terjangkau di kampus",
            "Batasi spending untuk hangout - pilih tempat yang value for money",
            "Investasi pada skill yang bisa jadi side income (desain, coding, content creation)",
            "Tetap sisihkan minimal 10% untuk tabungan meski aktivitas padat",
            "Join komunitas yang sejalan dengan minat untuk networking efektif"
            ]
        }
    }
    selected_tips = tips_bank.get(lifestyle, tips_bank["moderat"])
    
    # Prediction status
    prediction_status = None
    prediction_message = None
    if pemasukan == 0:
        prediction_status = "Kosong"
        prediction_message = "Belum ada data pemasukan bulan ini."
    else:
        if predicted_sisa < target_nabung * Decimal('0.5'):
            prediction_status = "Boros"
            kekurangan = (target_nabung - predicted_sisa).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            prediction_message = f"Prediksi nabung: Rp {int(predicted_sisa):,} (kurang Rp {int(kekurangan):,})"
        elif predicted_sisa < target_nabung:
            prediction_status = "Waspada"
            kekurangan = (target_nabung - predicted_sisa).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            prediction_message = f"Hampir! Prediksi nabung: Rp {int(predicted_sisa):,} (kurang Rp {int(kekurangan):,})"
        else:
            prediction_status = "Aman"
            lebih = (predicted_sisa - target_nabung).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            prediction_message = f"Mantap! Prediksi nabung: Rp {int(predicted_sisa):,} (lebih Rp {int(lebih):,})"
    
    # Emergency warning
    spending_pace = (pengeluaran / pemasukan * Decimal('100')) if pemasukan > 0 else Decimal('0')
    time_pace = (Decimal(str(current_day)) / Decimal(str(days_in_month)) * Decimal('100'))
    emergency_warning = None
    if spending_pace > time_pace * Decimal('1.5'):
        emergency_warning = {'level':'critical','message':f"Kamu sudah pakai {float(spending_pace):.0f}% budget",'daily_limit':float(daily_budget_limit)}
    elif spending_pace > time_pace * Decimal('1.2'):
        emergency_warning = {'level':'warning','message':f"Spending {float(spending_pace):.0f}% lebih cepat",'daily_limit':float(daily_budget_limit)}
    
    print(f"✓ Final calculations: {time.time() - final_start:.3f}s")
    
    # User recommendation
    user_recommendation = {
        "lifestyle": lifestyle,
        "income": income_idr,
        "recommendations": ml_budget,
        "description": selected_tips["desc"],
        "tips": selected_tips["tips"]
    }
    
    # Build response
    response_data = {
        "pemasukan": float(pemasukan),
        "pengeluaran": float(pengeluaran),
        "sisa": float(sisa),
        "target_nabung": float(target_nabung),
        "budget_recommendations": budget_recommendations,
        "daily_status": daily_status,
        "weekly_progress": weekly_progress,
        "prediction_status": prediction_status,
        "prediction_message": prediction_message,
        "emergency_warning": emergency_warning,
        "current_week": current_week,
        "days_remaining": days_remaining,
        "days_in_month": days_in_month,
        "current_day": current_day,
        "total_saving_potential": float(total_saving_potential),
        "ml_budget": ml_budget,
        "income_idr": income_idr,
        "lifestyle": lifestyle,
        "selected_tips": selected_tips,
        "user_recommendation": user_recommendation
    }
    
    # Cache response
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(cache_key, 300, json.dumps(response_data, default=str))
            print(f"✓ Data cached")
        except Exception as e:
            print(f"⚠️ Cache save error: {e}")
    
    total_time = time.time() - route_start
    print(f"✅ Analytics completed in {total_time:.3f}s")
    print("="*50 + "\n")
    
    return render_template("analytics.html", **response_data)

# ===== API ENDPOINT untuk AJAX (Optional) =====
@app.route("/api/analytics", methods=["POST"])
def api_analytics():
    """
    API endpoint untuk fetch analytics data via AJAX
    Menghindari full page reload
    """
    try:
        target_nabung_input = request.json.get("target")
        lifestyle = request.json.get("lifestyle", "moderat")
        
        # Redirect to analytics route with params
        # Atau bisa copy logic di atas dan return JSON
        
        return jsonify({
            "status": "success",
            "redirect": f"/analytics?target={target_nabung_input}&lifestyle={lifestyle}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/guide")
def guide():
    return render_template("guide.html")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    app.run()

RAILWAY_START_COMMAND = gunicorn app:app
