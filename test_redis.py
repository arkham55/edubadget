import redis

try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    result = redis_client.ping()
    print(f"✅ Redis Connected: {result}")
except Exception as e:
    print(f"❌ Error: {e}")