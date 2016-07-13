import tweepy
import string
import csv
import sys
user_auth = tweepy.OAuthHandler('ZvS9I5d0K8VeaZp11T90AnCyR', 'rUa1F0yKSZ0LYk6aurSWVIejZ9YtMBYfpFuawsN7JYngwjbw0F')
user_auth.set_access_token('3421429701-q9tUJqHfgHNhR2qVLAUfQZHKbMfmKOO7dHRekQx', 'oQuBTY35HiT7xt69kifMsdDOCl38GAWnL4zYD5aNAf8SE')

exclude = set(string.punctuation)
sys.stdout.write("screen_name,created_at,text\n")

class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        s = ''.join(ch for ch in status.text.encode('utf-8') if ch not in exclude)

        sys.stdout.write(status.author.screen_name)
        sys.stdout.write(",")
        sys.stdout.write(str(status.created_at)) 
        sys.stdout.write(",")
        sys.stdout.write(s)
        sys.stdout.write("\n")

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = user_auth, listener=myStreamListener)

myStream.filter(track=['election2016'])

