import tweepy
from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import twitter_config
import json
import pandas as pd
import csv
import re

clean_data=[]
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        consumer_key = twitter_config.consumer_key
        consumer_secret = twitter_config.consumer_secret
        access_token = twitter_config.access_token
        access_secret = twitter_config.access_secret

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        return auth


class TweetAnalyzer():
    def clean_tweet(self,tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([\"])|([R][T])", '', tweet).split())

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet for tweet in tweets], columns=['SentimentText'])
        return df

class TwitterListener(StreamListener):
    def __init__(self, tweets_filename,num):
        self.tweets_filename = tweets_filename
        self.maxcount=num
        with open(self.tweets_filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(["SentimentText"])

    def on_data(self, data):
        if self.maxcount!=0:
            self.maxcount -= 1
            try:
                print(data)
                json_data=json.loads(data)
                send_data=json_data['text']
                tweet_analyzer = TweetAnalyzer()
                ct=tweet_analyzer.clean_tweet(send_data)

                with open(self.tweets_filename, 'a', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    if ct != "n" and ct:
                        writer.writerow([ct])

                return True

            except BaseException as e:
                print("Error on_data %s" % str(e))
                return True



    def on_error(self, status):
        print(status)
        return True

class TwitterStream():
    """ Streaming of live tweets """
    def __init__(self, tweets_filename,num):
        self.twitter_authenticator=TwitterAuthenticator()
        listener = TwitterListener(tweets_filename,num)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        self.stream = Stream(auth, listener)

    def stream_tweets(self, topic):
        '''listener = TwitterListener(tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)'''
        self.stream.filter(languages=['en'], track=[topic],async=True)

