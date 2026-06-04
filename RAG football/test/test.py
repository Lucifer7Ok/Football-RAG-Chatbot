import urllib.request, json

API_KEY = "AIzaSyBPZ7LdMzbliA013adnr36S_SgreVtUpjo"

url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"
try:
    req = urllib.request.urlopen(url, timeout=10)
    data = json.loads(req.read())
    embed_models = [m["name"] for m in data.get("models", []) if "embed" in m["name"]]
    print(f"✅ Key hợp lệ | Embedding models: {embed_models}")
except urllib.error.HTTPError as e:
    body = json.loads(e.read())
    print(f"❌ HTTP {e.code}: {body['error']['message']}")