import warnings
import pandas as pd
from detoxify import Detoxify
from textblob import TextBlob
import json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 🔹 Load data
df = pd.read_csv("social_feed_metadata.csv")
df['post_text'] = df['post_text'].fillna("").str.lower()

# 🔍 Toxicity analysis using Detoxify
print("Analyzing toxicity...")
df['toxicity_score'] = df['post_text'].apply(lambda x: Detoxify('original').predict(x)['toxicity'])

# 😊 Sentiment analysis (optional, not used in classification but useful for reports)
df['polarity'] = df['post_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 🚫 Unsafe keyword list
unsafe_keywords = ['kill', 'hate', 'nsfw', 'violence', 'gun', 'suicide', 'drugs']
df['contains_unsafe_keywords'] = df['post_text'].apply(lambda x: any(word in x for word in unsafe_keywords))

# 🧠 Classification logic
def classify(row):
    score = row['toxicity_score']
    keywords = row['contains_unsafe_keywords']
    
    if score > 0.7 or keywords:
        return "Unsafe", "High toxicity or flagged keyword"
    elif score >= 0.3:
        return "Neutral", "Moderate toxicity"
    else:
        return "Safe", "Low toxicity"

df[['final_label', 'reason']] = df.apply(lambda row: pd.Series(classify(row)), axis=1)

# 💾 Save moderated feed
df[['post_id', 'post_text', 'toxicity_score', 'final_label', 'reason']].to_csv("moderated_feed.csv", index=False)
print("Saved moderated_feed.csv")

# 📊 Create report summary
report = {
    "total_posts": len(df),
    "flagged_posts": int(len(df[df['final_label'] == "Unsafe"])),
    "percent_unsafe": round(100 * len(df[df['final_label'] == "Unsafe"]) / len(df), 2),
    "common_reasons": df[df['final_label'] == "Unsafe"]['reason'].value_counts().to_dict(),
    "sample_flagged_posts": df[df['final_label'] == "Unsafe"].head(5)[['post_id', 'post_text', 'reason']].to_dict(orient='records'),
    "overall_safety_thoughts": "Moderation required for unsafe content. Consider stricter keyword checks or image analysis if needed."
}

with open("report_summary.json", "w") as f:
    json.dump(report, f, indent=4)

print("Saved report_summary.json")
