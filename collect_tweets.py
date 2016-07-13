import tweepy
import sys
import string
app_auth = tweepy.AppAuthHandler('ZvS9I5d0K8VeaZp11T90AnCyR', 'rUa1F0yKSZ0LYk6aurSWVIejZ9YtMBYfpFuawsN7JYngwjbw0F')
app_api = tweepy.API(app_auth, wait_on_rate_limit=True)
tweets = []
names = sys.stdin.read()
names =  names.split("\n")
exclude = set(string.punctuation)

for name in names:
    tweets = []

    row =  name.split(",")
    for page in tweepy.Cursor(app_api.user_timeline, id=row[0], count = 200).pages():
            tweets.extend(page)
    sys.stdout.write(row[0])
    sys.stdout.write(",")
    sys.stdout.write(row[1])
    sys.stdout.write(",")


    for tweet in tweets:
        for hashtag in tweet.entities["hashtags"]:
            sys.stdout.write(hashtag["text"].encode('utf-8'))
            sys.stdout.write(" ")
    sys.stdout.write(";")


