from fastapi import APIRouter
import yfinance as yf

router = APIRouter()

@router.get("/news/live")
def get_live_news(symbol: str = "AAPL"):
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news

        formatted_news = []
        for item in news_items[:10]:  # top 10 news
            formatted_news.append({
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "published": item.get("providerPublishTime"),
            })

        return {"symbol": symbol, "news": formatted_news}

    except Exception as e:
        return {"error": str(e)}
