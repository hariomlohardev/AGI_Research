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

def bayes_update(prior, likelihood, evidence):
    probability =  (likelihood * prior)/evidence
    return probability

word_storage = {}
for sms in dataset:
    # print(f" message: {sms[0]} \n Is spam: {True if sms[1]== "spam" else False} \n **************************")
    if sms[1] == 'spam':
        for word in sms[0].split():
            if word.lower().removesuffix('.') in word_storage:
                word = word.lower().removesuffix('.')
                word_storage[word] += 1
            else:
                word = word.lower().removesuffix('.')
                word_storage[word] = 1

total_words = sum(word_storage.values())
print(total_words)
print(word_storage)
probability = {}
for word in word_storage:
    count = word_storage[word]
    prob = count/total_words
    probability[word] = round(prob ,ndigits=2)

print(f"***************\n ")
for word in probability:
    print(f"{word} : {probability[word]}")