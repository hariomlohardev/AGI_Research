# %%
import pandas as pd
import re
import math
import matplotlib.pyplot as plt

dataset = [
    ("Hey, are we still on for lunch at 1?", "ham"),
    ("CONGRATULATIONS! You have won a $500 gift card. Reply WIN to claim now.", "spam"),
    ("Can you pick up some milk on your way home?", "ham"),
    ("URGENT: Your bank account has been locked. Click here to verify your identity.", "spam"),
    ("I'm running about 10 minutes late, start without me.", "ham"),
    ("Get a FREE custom website today! Limited time offer. Call 1-800-WEB-NOW.", "spam"),
    ("Did you see the game last night? That ending was crazy.", "ham"),
    ("Exclusive offer for our loyal customers: 50% off all luxury watches. Buy now!", "spam"),
    ("Don't forget to attach the report before you send that email.", "ham"),
    ("FINAL NOTICE: Your car warranty is about to expire. Press 1 to speak with an agent.", "spam"),
    ("Sounds good, see you tomorrow morning.", "ham"),
    ("Make $5000 a week working from home! No experience needed. Text MONEY to 8888.", "spam"),
    ("Happy birthday! Hope you have a great day.", "ham"),
    ("You have (1) new voicemail from a private number. Call 0900-LISTEN to retrieve.", "spam"),
    ("Are you free for a quick call this afternoon?", "ham"),
    ("LOAN APPROVED! You are eligible for up to $10,000. Click to deposit now.", "spam"),
    ("Let me know when you get there so I know you're safe.", "ham"),
    ("Win a free vacation to the Bahamas! Just text SUN to 555-5555.", "spam"),
    ("I'll send the draft over by tonight for your review.", "ham"),
    ("Your package could not be delivered. Please pay the $2.99 shipping fee here.", "spam")
]

# %%

class Classifier:
    """ 

    This is an simple Spam classifier \n
    Parameters to init are "dataset" : list (list of dictionary)\n
    Example :\n
    dataset = [\n
        ("Hey, are we still on for lunch at 1?", "ham"),\n
        ("CONGRATULATIONS! You have won a $500 gift card. Reply WIN to claim now.", "spam"),\n
        ]\n
    """
    def __init__(self , dataset : list = None):
        self.dataset = dataset
        self.total_spam = 0
        self.total_spam_word = 0
        self.total_ham_word = 0

        self.spam_words = None
        self.ham_words = None
        self.vocab_size = None

        self.init()
        self.total_ham = len(dataset) - self.total_spam
        self.total_sms =  self.total_spam + self.total_ham

        self.total_word =  self.total_spam_word + self.total_ham_word


    def init(self):
        data = []
        for sms, label in dataset:
            if label == 'spam':
                self.total_spam += 1
            # Basic cleaning: lowercase and remove non-alphanumeric
            words = re.findall(r'\w+', sms.lower())
            for word in words:
                data.append({'word': word, 'label': label})

        df = pd.DataFrame(data)


        df = df.groupby(['word', 'label']).size().reset_index(name='word_count')
        self.vocab_size = len(set(df['word']))


        # Create a DataFrame containing only spam
        df_spam = df[df['label'] == 'spam'].copy()
        # Create a DataFrame containing only ham
        df_ham = df[df['label'] == 'ham'].copy()


        df_ham = df_ham.drop(columns='label')
        self.total_ham_word = sum(df_ham['word_count'])
        self.ham_words = dict(zip(df_ham['word'] ,df_ham['word_count']))
        
        df_spam = df_spam.drop(columns='label')
        self.total_spam_word = sum(df_spam['word_count'])
        self.spam_words = dict(zip(df_spam['word'] ,df_spam['word_count']))
        



    def _spam_prob(self , word:str):
        word_count = self.spam_words.get(word , 0)
        prob = (word_count + 1) / (self.total_spam_word + self.vocab_size)
        return float(prob)

    def _ham_prob(self , word:str):
        word_count = self.ham_words.get(word ,0)
        prob = (word_count + 1) / (self.total_ham_word + self.vocab_size)
        return float(prob)

    def is_spam(self, string :str):
        words = string.lower().strip().split()
        total_prob = 0
        for word in words:
            prob = self._spam_prob(word)
            log_prob = math.log(prob)
            total_prob += log_prob

        prob_is_spam = math.log(self.total_spam / self.total_sms)
        total_prob += prob_is_spam
        return total_prob

    def is_ham(self, string :str):
        words = string.lower().strip().split()
        total_prob = 0
        for word in words:
            prob = self._ham_prob(word)
            log_prob = math.log(float(prob))
            total_prob += log_prob

        prob_is_ham = math.log(self.total_ham / self.total_sms)
        total_prob += prob_is_ham
        return total_prob

    def __call__(self,string :str):
        """
        Simple spam classifier
        Input : string , pass the message as an string
        Output : {
            'prob_spam' : prob_spam,
            'prob_ham' : prob_ham
        }
        """
        prob_spam = self.is_spam(string)
        prob_ham = self.is_ham(string)

        if prob_spam > prob_ham:
            print("ALERT This is a SPAM")
        elif prob_ham > prob_spam:
            print("This message is safe")
        else:
            print("something went wrong Sorry!")

        return {
            'prob_spam' : prob_spam,
            'prob_ham' : prob_ham
        }

        

        

# %%
clas = Classifier(dataset)

# %%
clas('you won cash')

# %%


# %%



