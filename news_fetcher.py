import requests

def fetch_live_news(query="startups"):
    API_KEY = "58678627711942188765caa5c43cd62e"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={API_KEY}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        articles = res.json().get('articles', [])
        news_items = [{'title': art['title'], 'url': art['url']} for art in articles]
        return news_items
    except Exception as e:
        print(f"News API error: {e}")
        return []
    