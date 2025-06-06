import csv
from collections import defaultdict
from datetime import datetime
import smtplib
from email.message import EmailMessage
import json
import os
from dotenv import load_dotenv
# Load your CSV reviews once, at module import
reviews_data = []

load_dotenv()

gmail_user = os.getenv("GMAIL_USER")
gmail_password = os.getenv("GMAIL_PASSWORD")


def load_reviews_from_csv(file_path="realistic_restaurant_reviews.csv"):
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reviews_data.append(row)

load_reviews_from_csv()

def rating_summary_tool(_input: str) -> str:
    print("Called rating_summary_tool")
    # Simple summary example: average rating and count
    total_rating = 0
    count = 0
    for review in reviews_data:
        try:
            total_rating += int(review["Rating"])
            count += 1
        except:
            pass
    avg_rating = total_rating / count if count else 0

    summary = f"There are {count} reviews with an average rating of {avg_rating:.1f} stars."
    return summary

def search_reviews_tool(query: str) -> str:
    print(f"Called search_reviews_tool with query: {query}")
    results = []
    for review in reviews_data:
        if query.lower() in review["Review"].lower():
            results.append(f"{review['Title']} ({review['Date']}): {review['Review']}")
    if not results:
        return "No reviews found matching your query."
    # Return up to first 3 matches to keep output manageable
    return "\n\n".join(results[:3])

def count_rating_tool(rating_str: str) -> str:
    try:
        rating = int(rating_str.strip())
    except Exception:
        return "Invalid input. Please provide a number like '5'."

    count = sum(1 for review in reviews_data if str(review.get("Rating", "")).strip() == str(rating))
    return f"{count} customers gave a {rating}-star rating."

def top_rated_comments_tool(n: str) -> str:
    try:
        n = int(n)
    except:
        return "Please provide a number like '3' to get top rated comments."
    
    top_reviews = [
        review for review in reviews_data
        if str(review.get("Rating", "")).strip() == "5"
    ]
    top_reviews = top_reviews[:n]
    return "\n\n".join(f"{r['Title']} ({r['Date']}): {r['Review']}" for r in top_reviews) if top_reviews else "No top rated reviews found."

def low_rated_reasons_tool(_input: str = "") -> str:
    keywords = {}
    for review in reviews_data:
        rating = review.get("Rating", "").strip()
        if rating in ["1", "2"]:
            for word in review["Review"].lower().split():
                word = word.strip(".,!?()")
                if word.isalpha():
                    keywords[word] = keywords.get(word, 0) + 1
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
    return f"Common keywords in low-rated reviews: {', '.join(word for word, _ in sorted_keywords)}"

def review_count_by_date_tool(date: str) -> str:
    count = sum(1 for review in reviews_data if review.get("Date") == date)
    return f"{count} reviews were posted on {date}."

def most_mentioned_dish_tool(_input: str = "") -> str:
    word_freq = {}
    for review in reviews_data:
        for word in review["Review"].lower().split():
            word = word.strip(".,!?()")
            if word.isalpha() and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
    if not word_freq:
        return "No mentions found."
    top_word = max(word_freq.items(), key=lambda x: x[1])
    return f"The most mentioned term in reviews is '{top_word[0]}' with {top_word[1]} mentions."

def sentiment_trend_tool(_input: str = "") -> str:
    ratings_by_month = defaultdict(list)
    for review in reviews_data:
        try:
            date_obj = datetime.strptime(review["Date"], "%Y-%m-%d")
            month = date_obj.strftime("%Y-%m")
            rating = int(review["Rating"])
            ratings_by_month[month].append(rating)
        except:
            continue
    sorted_months = sorted(ratings_by_month.keys())
    trend = [
        f"{month}: {sum(ratings_by_month[month]) / len(ratings_by_month[month]):.2f}"
        for month in sorted_months
    ]
    return "Sentiment trend by month:\n" + "\n".join(trend)


def send_mail_tool(input_str: str) -> str:
    """
    Expects input_str as a JSON string with keys:
    {
        "to": "recipient@example.com",
        "subject": "Subject here",
        "body": "Email body here"
    }
    Sends the email via SMTP and returns success or error message.
    """
    try:
        data = json.loads(input_str)
        to_addr = data["to"]
        subject = data.get("subject", "No Subject")
        body = data.get("body", "")

        msg = EmailMessage()
        msg["From"] = gmail_user  # change to your email
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)

        # Setup SMTP server (example using Gmail SMTP)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_password) 
            smtp.send_message(msg)

        return f"Email sent successfully to {to_addr}"
    except Exception as e:
        return f"[SendMail Error] {str(e)}"

def final_answer_tool(text: str) -> str:
    if not text:
        return "No final answer provided."
    return text.strip()
