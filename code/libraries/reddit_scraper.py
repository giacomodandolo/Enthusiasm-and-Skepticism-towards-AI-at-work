import pandas as pd
from time import sleep
from datetime import datetime
from tqdm.auto import tqdm
import re
import praw
import praw.exceptions

class RedditScraper:
    def __init__(self, client_id, client_secret, application_name, application_version):
        self.client_id = client_id
        self.client_secret = client_secret
        self.application_name = application_name
        self.application_version = application_version
        self.reddit_client = praw.Reddit(
            client_id = self.client_id,
            client_secret = self.client_secret,
            user_agent = f"{application_name}_{application_version}",
            check_for_async = False
        )

    def retrieve_posts(self, subreddit_name, post_num, time_filter):
        all_posts = []
        subreddit = self.reddit_client.subreddit(subreddit_name)
        posts = subreddit.top(limit=post_num, time_filter=time_filter)

        for post in tqdm(posts, total=post_num, desc='Reddit posts'):
            all_posts.append({
                'subreddit': post.subreddit.display_name,
                'selftext': post.selftext,
                'author_fullname': post.author.name if post.author else 'N/A',
                'title': post.title,
                'upvote_ratio': post.upvote_ratio,
                'ups': post.ups,
                'created': post.created,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'author': str(post.author) if post.author else 'N/A',
                'id': post.id
            })
        
        posts_df = pd.DataFrame(all_posts) # build dataframe
        posts_df.drop_duplicates(subset='id', inplace=True)
        return posts_df
    
    def retrieve_posts_with_keyword(self, subreddit_name, post_num, time_filter, keywords):
        keywords = [kw.lower() for kw in keywords]
        all_posts = []
        subreddit = self.reddit_client.subreddit(subreddit_name)

        posts = subreddit.top(limit=None, time_filter=time_filter)
        for post in tqdm(posts, desc=f'Reddit top posts filtered by {keywords}'):
        
            # Combina titolo e selftext, tutto in minuscolo
            text_to_check = ((post.title or "") + " " + (post.selftext or "")).lower()
            
            # Controlla se almeno una keyword è presente
            if any(re.search(rf'\b{re.escape(kw)}\b', text_to_check) for kw in keywords):
                all_posts.append({
                    'subreddit': post.subreddit.display_name,
                    'selftext': post.selftext,
                    'author_fullname': post.author_fullname if post.author else 'N/A',
                    'title': post.title,
                    'upvote_ratio': post.upvote_ratio,
                    'ups': post.ups,
                    'created': post.created,
                    'created_utc': post.created_utc,
                    'created_datetime': datetime.utcfromtimestamp(post.created_utc),
                    'num_comments': post.num_comments,
                    'author': str(post.author) if post.author else 'N/A',
                    'id': post.id
                })
            
            if post_num is not None and len(all_posts) >= post_num:
                break

        posts_df = pd.DataFrame(all_posts)
        posts_df.drop_duplicates(subset='id', inplace=True)
        return posts_df
    
    def extract_comments(self, post_df, num_posts = None):
        if num_posts is None:
            num_posts = len(post_df)

        comments_list = []
        
        for i in tqdm(range(num_posts), desc='Reddit comments'):
            post_id = post_df.iloc[i]['id']
            submission = self.reddit_client.submission(id=post_id)
            try:
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    if comment.body.strip().lower() == "[deleted]":
                        continue

                    comments_list.extend([{'comment_id': comment.id,
                                        'parent_id': comment.parent_id,
                                        'post_id': post_id,
                                        'comment_body': comment.body,
                                        'created_utc': comment.created_utc,
                                        'created_datetime': datetime.utcfromtimestamp(comment.created_utc)
                                        }])

            except Exception as e:
                print("\n ERROR while processing post:", post_id)
                print("Error type:", type(e).__name__)
                print("Error message:", e)
                print("Skipping this post...\n")
                sleep(2)
                continue

        comments_df = pd.DataFrame(comments_list)
        return comments_df

    def extract_comments_with_keyword(self, post_df, keywords, num_posts = None):
        if num_posts is None:
            num_posts = len(post_df)

        keywords = [kw.lower() for kw in keywords]
        comments_list = []

        for i in tqdm(range(num_posts), desc='Reddit comments'):
            post_id = post_df.iloc[i]['id']
            submission = self.reddit_client.submission(id=post_id)
            try:
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    if not comment.body or comment.body.strip().lower() in ["[deleted]", "[removed]"]:
                        continue

                    comment_text = comment.body.lower() if comment.body else ""
                    if any(kw in comment_text for kw in keywords):
                        comments_list.append({
                            'comment_id': comment.id,
                            'parent_id': comment.parent_id,
                            'post_id': post_id,
                            'comment_body': comment.body,
                            'created_utc': comment.created_utc,
                            'created_datetime': datetime.utcfromtimestamp(comment.created_utc)
                        })
            except Exception as e:
                print("\n ERROR while processing post:", post_id)
                print("Error type:", type(e).__name__)
                print("Error message:", e)
                print("Skipping this post...\n")
                sleep(2)
                continue

        comments_df = pd.DataFrame(comments_list)
        return comments_df