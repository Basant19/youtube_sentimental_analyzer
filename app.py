import os
import io
import base64
import logging
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Flask, render_template, request

# Custom imports from your project
from src.utils.scrapper import get_youtube_comments
from src.utils.inference import load_model_from_registry, predict_sentiment

# Define template and static directory paths explicitly
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'static')

# Initialize Flask with custom template and static folder
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the MLflow model from the registry at startup
try:
    model = load_model_from_registry("my_model")
    logging.info("Model loaded successfully from MLflow registry.")
except Exception as e:
    logging.exception("Failed to load model from MLflow registry.")
    raise e

def generate_wordcloud(text_list):
    """
    Generate a word cloud image (base64-encoded) from a list of comments.
    """
    try:
        logging.debug("Generating word cloud...")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_list))
        buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        logging.debug("Word cloud generated successfully.")
        return img_base64
    except Exception as e:
        logging.exception("Error occurred while generating word cloud.")
        raise e

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle GET and POST requests:
    - GET: Display the form.
    - POST: Process the YouTube URL, predict sentiment, and generate visualizations.
    """
    if request.method == "POST":
        url = request.form.get("youtube_url")
        logging.info(f"Received POST request for URL: {url}")

        try:
            # Step 1: Scrape YouTube comments
            comments = get_youtube_comments(url)
            logging.info(f"Fetched {len(comments)} comments from YouTube.")

            if not comments:
                raise ValueError("No comments were retrieved from the video.")

            # Step 2: Predict sentiment using the loaded model
            sentiments = predict_sentiment(comments, model)
            logging.info("Sentiment prediction completed.")

            # Step 3: Count the sentiment categories
            sentiment_counts = {
                "Positive": sentiments.count("Positive"),
                "Negative": sentiments.count("Negative"),
                "Neutral": sentiments.count("Neutral")
            }
            logging.debug(f"Sentiment counts: {sentiment_counts}")

            # Step 4: Generate word cloud from comments
            wordcloud_img = generate_wordcloud(comments)

            # Step 5: Render the results in the HTML template
            return render_template("index.html",
                                   url=url,
                                   sentiment_counts=sentiment_counts,
                                   wordcloud_img=wordcloud_img,
                                   show_results=True)

        except Exception as e:
            logging.exception("An error occurred during sentiment analysis.")
            return render_template("index.html", error=str(e), show_results=False)

    # If GET request, just show the form
    return render_template("index.html", show_results=False)

if __name__ == "__main__":
    app.run(debug=True)
