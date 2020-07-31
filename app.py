from tweetfetch import *
from texttrain import *


topic = input()
num = int(input())

twitter_streamer = TwitterStream('tweet.csv',num)
twitter_streamer.stream_tweets(topic)

t = Traintext()
t.train()
t.textplot(topic)