# import nltk
# nltk.download("stopwords")

# from nltk.corpus import stopwords
# print(stopwords.words("english")[:20])  # prints first 20 Hindi stopwords


import stopwordsiso as stopwords

# Check if Hindi stopwords exist
print("hindi" in stopwords.langs())  

# Get Hindi stopwords
hindi_stopwords = stopwords.stopwords("hindi")
print(list(hindi_stopwords)[:20])
