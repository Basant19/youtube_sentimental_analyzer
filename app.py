import os
import uuid
import logging
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Flask, render_template, request, redirect, url_for, session

from src.utils.scrapper import get_youtube_comments
from src.utils.inference import load_model_and_artifacts, predict_sentiment

# Template/static folder paths
base_dir = os.path.dirname(__file__)
template_dir = os.path.join(base_dir, 'frontend', 'templates')
static_dir = os.path.join(base_dir, 'frontend', 'static')
wordcloud_dir = os.path.join(static_dir, 'wordclouds')
os.makedirs(wordcloud_dir, exist_ok=True)  # ✅ Ensure wordcloud dir exists

# Flask app setup
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'your_secret_key'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Load model and vectorizer
try:
    model, vectorizer = load_model_and_artifacts("my_model")
    logging.info("✅ Model and vectorizer loaded.")
except Exception as e:
    logging.exception("❌ Could not load model/vectorizer.")
    raise e


def generate_wordcloud(text_list):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_list))
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(wordcloud_dir, filename)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, format='png')
        plt.close()

        return filename
    except Exception as e:
        logging.exception("❌ Word cloud generation failed.")
        return None


def map_prediction(value):
    return "Positive" if value == 1 else "Negative" if value == -1 else "Neutral"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("youtube_url")
        try:
            max_comments = int(request.form.get("max_comments", 20))
        except ValueError:
            max_comments = 20

        logging.info(f"🔍 Processing URL: {url} | Max comments: {max_comments}")

        try:
            comments = get_youtube_comments(url)
            if not comments:
                raise ValueError("No comments retrieved from the video.")

            comments = comments[:max_comments]
            logging.info(f"✅ Retrieved {len(comments)} comments.")

            raw_results = predict_sentiment(comments, model, vectorizer)
            results = [(comment, map_prediction(pred)) for comment, pred in raw_results]

            sentiments = [s for _, s in results]
            sentiment_counts = {
                "Positive": sentiments.count("Positive"),
                "Negative": sentiments.count("Negative"),
                "Neutral": sentiments.count("Neutral"),
            }

            wordcloud_filename = generate_wordcloud(comments)

            # Store minimal data in session
            session['results'] = results
            session['sentiment_counts'] = sentiment_counts
            session['wordcloud_filename'] = wordcloud_filename

            return redirect(url_for('results'))

        except Exception as e:
            logging.exception("❌ Error in processing.")
            return render_template("index.html", error=str(e), show_results=False)

    return render_template("index.html", show_results=False)


@app.route("/results")
def results():
    results = session.get('results')
    sentiment_counts = session.get('sentiment_counts')
    wordcloud_filename = session.get('wordcloud_filename')

    if not results:
        return redirect(url_for("index"))

    wordcloud_url = f"/static/wordclouds/{wordcloud_filename}" if wordcloud_filename else None

    return render_template("index.html",
                           results=results,
                           sentiment_counts=sentiment_counts,
                           wordcloud_img_url=wordcloud_url,
                           show_results=True)


@app.route("/clear", methods=["POST"])
def clear_results():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
