from flask import Flask, request, render_template, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the AI model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # ⚡ Make summary much shorter
    summary = summarizer(
        text,
        max_length=40,   # cut down max words (was 80)
        min_length=10,   # allow as short as ~1–2 lines
        do_sample=False
    )
    
    return render_template("index.html", original=text, summary=summary[0]['summary_text'])

if __name__ == "__main__":
    app.run(debug=True)
