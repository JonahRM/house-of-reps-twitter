import sys
from fuzzywuzzy import fuzz

f = open("house_party.txt")
house = f.read().split("\n")
house_dict = {}

for row in house:
    row = row.split(",")
    house_dict.update({row[0]:row[1]})

f = open("house_tweets.txt")

house_tweets = f.read().split(";")
for row in house_tweets:
    row = row.split(',')
    current_winner_party = ""
    current_winner = ""
    current_winner_score = 0
    current_score = 0
    for name,party in house_dict.iteritems():
        current_score = fuzz.ratio(row[1], name)
        if (current_score > current_winner_score):
            current_winner_score = current_score
            current_winner = name
            current_winner_party = party
    sys.stdout.write(row[0])
    sys.stdout.write(",")
    sys.stdout.write(row[1])
    sys.stdout.write(",")
    sys.stdout.write(current_winner_party)
    sys.stdout.write(",")
    sys.stdout.write(row[2])
    sys.stdout.write(";")









