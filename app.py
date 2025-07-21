import os
import io
import base64
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Flask, render_template, request

from src.utils.scrapper import get_youtube_comments  # Ensure this doesn't require 'max_results' if not used
from src.utils.inference import load_model_and_artifacts, predict_sentiment

# Set up template and static directory paths
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'static')

# Flask App Initialization
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Model and Vectorizer Once
try:
    model, vectorizer = load_model_and_artifacts("my_model")
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.exception("❌ Failed to load model artifacts.")
    raise e


def generate_wordcloud(text_list):
    """Generate base64 image of word cloud from list of texts."""
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_list))
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logging.exception("❌ Word cloud generation failed.")
        return None


def map_prediction(value):
    """Map numerical predictions to human-readable labels."""
    if value == 1:
        return "Positive"
    elif value == -1:
        return "Negative"
    return "Neutral"


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
            # Step 1: Scrape comments
            comments = get_youtube_comments(url)  # Adjust if you need to pass max_comments
            if not comments:
                raise ValueError("No comments retrieved from the YouTube video.")

            comments = comments[:max_comments]  # Limit to max_comments manually
            logging.info(f"✅ Retrieved {len(comments)} comments.")

            # Step 2: Predict sentiment
            raw_results = predict_sentiment(comments, model, vectorizer)
            results = [(comment, map_prediction(pred)) for comment, pred in raw_results]

            # Step 3: Count sentiment results
            sentiments = [sentiment for _, sentiment in results]
            sentiment_counts = {
                "Positive": sentiments.count("Positive"),
                "Negative": sentiments.count("Negative"),
                "Neutral": sentiments.count("Neutral"),
            }
            logging.info(f"📊 Sentiment counts: {sentiment_counts}")

            # Step 4: Generate word cloud
            wordcloud_img = generate_wordcloud(comments)

            # Step 5: Render result
            return render_template("index.html",
                                   url=url,
                                   results=results,
                                   sentiment_counts=sentiment_counts,
                                   wordcloud_img=wordcloud_img,
                                   show_results=True)

        except Exception as e:
            logging.exception("❌ Error during processing.")
            return render_template("index.html", error=str(e), show_results=False)

    # GET request
    return render_template("index.html", show_results=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use 8080 as default if PORT not set
    app.run(host="0.0.0.0", port=port, debug=True)

