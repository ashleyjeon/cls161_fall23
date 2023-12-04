# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# next, we initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()

# initialize our 'english.pickle' function and give it a short name
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# the variable 'message_text' now contains the text we will analyze
message_text = '''Like you, I am getting very frustrated with this process. I am genuinely trying to be as reasonable as possible. I am not trying to "hold up" the deal at the last minute. I'm afraid that I am being asked to take a fairly large leap of faith after this company (I don't mean the two of you -- I mean Enron) has screwed me and the people who work for me.'''
print(message_text) # outputs this message

# calling the polarity_scores method on sid and passing in the message_text 
# outputs a directory with negative, neutral, positive, and compound scores for 
# the input text
scores = sid.polarity_scores(message_text)

# to print out the results of the sentiment analysis 
# here we loop through the keys contained in scores(pos, neu, neg, and compound scores)
# and print the key-value pairs on the screen
for key in sorted(scores):
    print('{0}: {1}, '.format(key, scores[key]), end='')

print('\n')

# Challenge Task:
challenge_text1 = '''Looks great.  I think we should have a least 1 or 2 real time traders in Calgary.'''
scores1 = sid.polarity_scores(challenge_text1)

for key in sorted(scores1):
    print('{0}: {1}, '.format(key, scores1[key]), end='')


# Determining Appropriate Scope for E-mail
# Continue with the same code the previous section, but replace the *message_text* variable with the new e-mail text:
message_text_scope = '''It seems to me we are in the middle of no man's land with respect to the  following:  Opec production speculation, Mid east crisis and renewed  tensions, US elections and what looks like a slowing economy (?), and no real weather anywhere in the world. I think it would be most prudent to play  the markets from a very flat price position and try to day trade more aggressively. I have no intentions of outguessing Mr. Greenspan, the US. electorate, the Opec ministers and their new important roles, The Israeli and Palestinian leaders, and somewhat importantly, Mother Nature.  Given that, and that we cannot afford to lose any more money, and that Var seems to be a problem, let's be as flat as possible. I'm ok with spread risk  (not front to backs, but commodity spreads). The morning meetings are not inspiring, and I don't have a real feel for  everyone's passion with respect to the markets.  As such, I'd like to ask  John N. to run the morning meetings on Mon. and Wed.  Thanks. Jeff'''
print(message_text_scope)

scores_scope = sid.polarity_scores(message_text_scope)

for key in sorted(scores_scope):
    print('{0}: {1}, '.format(key, scores_scope[key]), end='')

