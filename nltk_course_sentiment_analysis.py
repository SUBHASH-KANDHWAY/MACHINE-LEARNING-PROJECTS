

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
nltk.download('vader_lexicon')

my_string = "I read a book. The first three chapters were boring and depressing. Then the next 2 chapters were wonderful. I recommend this book to my friends."
sent = sent_tokenize(my_string)
print(sent)

sa = SentimentIntensityAnalyzer()
for sentence in sent:
  print(sentence)
  ps = sa.polarity_scores(sentence)
  for n in ps:
    print('{0}: {1}, '.format(n, ps[n]), end='')
  print()
  print()