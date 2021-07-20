
import streamlit as st
import tweepy

from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer

import ftfy

from nltk.stem import PorterStemmer

#from spellchecker import SpellChecker
import nltk

from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()
english_stemmer = PorterStemmer()
#english_corrector = SpellChecker()


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

text_processor = TextPreProcessor(
    # terms that will be annotated
    annotate={ "allcaps", "elongated", "repeated",'emphasis'},
    #annotate={ "allcaps", "elongated", "repeated"},
    corrector="twitter", 
    
    unpack_hashtags=False,  # perform word segmentation on hashtags
    unpack_contractions=False,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize
)




my_file = open("positive_words2withtopwordsNEW.txt", "r", encoding="utf-8")
content = my_file.read()
content_listp = content.split("\n")
my_file.close()
#print(content_listp)


my_file = open("negative_words2withtopwordsNEW.txt", "r", encoding="utf-8")
content = my_file.read()
content_listn = content.split("\n")
my_file.close()
#print(content_list)


my_file = open("negative_words2 rated 4 and 5NEW.txt", "r", encoding="utf-8")
content = my_file.read()
content_listdn = content.split("\n")
my_file.close()


#cleanning time

#en_stop = get_stop_words('en') 
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

urlPattern        =  r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*|[^ ]*(.com))"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"


tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()



# These are just common English contractions. There are many edge cases. i.e. University's working on it.
#if we remove punctuation might not be useful - we correct contraction
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", 
                    "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 
                    "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 
                    "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", 
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                    "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", 
                    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", 
                    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 
                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
                    "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", 
                    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
                    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
                    "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
                    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}



from gensim.parsing.preprocessing import remove_stopwords
def replace_word(text):
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*|[^ ]*(\.com)|[^ ]*(\.my)|(http...))"
    userPattern = r"@[^\s]+"
    htagPattern = r"#[^\s]+"
    alphaPattern = r"[^a-zA-Z]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"


    # Fix Problem with Unicode Text
    text = ftfy.fix_text(text)

    # Replace all URls with ''
    #text = re.sub(urlPattern, '', text)

    # Replace @USERNAME to ''.
    #text = re.sub(userPattern, '', text)

    # Replace #HASHTAG to ''.
    #text = re.sub(htagPattern, '', text)


    # Convert everything to lower case
    text = text.lower()

    # Expand abbreviation
    # TBC


    for i in range(len(text)):
        for key, value in contraction_dict.items():
              if text[i] == key:
                text[i] = value
                

                
    #text = remove_stopwords(text)
    x = 0
    for i in text:      
        
        for key, value in contraction_dict.items():
              
              if i == key:
                text[x] = ''
        x = x + 1
    
    # Replace all non alphabets.
    text = re.sub(alphaPattern, ' ', text)
    
    # Replace 3 or more consecutive letters by 2 letter.
    #text = re.sub(sequencePattern, seqReplacePattern, text)
    
    # Replace malay short form
    #for sf in malayshortform:
        #text = re.sub(r"\b" + sf + r"\b",  malayshortform[sf], text)


    # Tokenize the words
    #tokens = token.tokenize(text)

    #for w in tokens:
       #word_list.append(w)
    # Replace 3 or more consecutive letters by 2 letter.
    text = re.sub(sequencePattern, seqReplacePattern, text)



    #CONVERT DOUBLE SPACE TO SINGLE SPACE
    text = text.replace("  ", " ")
    text = text.replace("   ", " ")
    text = text.replace("    ", " ")
    

    
    # Add space at beginning and end
    text = " " + text + " "
    
    #IF THE ROW IS EMPTY LIST IT AS NAN
    if text == '' or text == ' ':
        text = np.nan
    
        
    return text 





consumerKey = 'd7RJDeV6M1TdKnXXdY29Zud5O'
consumerSecret = '8LV35luiAco2mBnQ1W6erOnA8cbMwVgxblfHjP5zk5dmAXGwd6'
accessToken = '2206645458-9qlftwQ5eiovob7GCp21VrAoFRXi7AJLGt5ts3O'
accessTokenSecret = 'Oc9ZKbHSL0reJhZYcU0Vk9UERbVvsTwerIfDUTwiRNGYf'



#Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    
# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret) 
    
# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit = True)















#plt.style.use('fivethirtyeight')










def app():


    st.title("Depression Analyzer")



    


    raw_text = st.text_area("Enter twitter username (without @)")





    Analyzer_choice = st.selectbox("Select the Activities",  ["Show Recent Tweets","Generate WordCloud" ,"Visualize result","Depression state across time"])


    if st.button("Analyze"):

        
        if Analyzer_choice == "Show Recent Tweets":

            st.success("Fetching last 100 Tweets")

            
            def get_data():
    
                posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")

                df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
                df2 = pd.DataFrame([tweet.created_at for tweet in posts], columns=['created_at'])
                df['date'] = df2['created_at']
                
                df['Clean_Text'] = ""
                
                for i in range(len(df)):
                        df['Clean_Text'][i] = replace_word(df['Tweets'][i])
                
                
                
                
                df['score'] = ""
                
                df['no_of_positive_words'] = ""
                df['no_of_negative_words'] = ""
                df['definitenegativescore'] = ""
                
                negativescore = 0
                positivescore = 0
                definitenegativescore = 0
                
                for i in range(len(df)):
                    negativescore = 0
                    positivescore = 0
                    definitenegativescore = 0
                    for key in content_listn:
                        key = " " + key
                        if key in df['Clean_Text'][i]:
                            negativescore  += 1
                            
                    for key1 in content_listp:
                        key1 = " " + key1
                        if key1 in df['Clean_Text'][i]:
                            positivescore  += 1
                    for key3 in content_listdn:
                        key3 = " " + key3
                        if key3 in df['Clean_Text'][i]:
                            definitenegativescore += 1
                    if positivescore > 0 or negativescore > 0 or definitenegativescore > 0:
                        if (negativescore > positivescore) or definitenegativescore > 0:
                            df['score'][i] = 'depressed'
                        else:
                            df['score'][i] = 'not depressed'
                    if positivescore <= 0 and negativescore <= 0:
                        df['score'][i] = "N/A"
                    if df['score'][i] != 'depressed' and df['score'][i] != 'not depressed' and df['score'][i] != 'N/A':
                        df['score'][i] = "N/A"
                    df['no_of_positive_words'][i] = positivescore
                    df['no_of_negative_words'][i] = negativescore + definitenegativescore
                    
                    #df['definitenegativescore'][i] = definitenegativescore
    
                return df
            df=get_data()
            df = df.drop(columns=['definitenegativescore'])
            st.write(df)


        elif Analyzer_choice=="Generate WordCloud":

            st.success("Generating Word Cloud")

            def gen_wordcloud():

                posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")


                # Create a dataframe with a column called Tweets
                df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
                
                # word cloud visualization
                allWords = ' '.join([twts for twts in df['Tweets']])
                wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
                plt.imshow(wordCloud, interpolation="bilinear")
                plt.axis('off')
                plt.savefig('WC.jpg')
                img= Image.open("WC.jpg") 
                return img

            img=gen_wordcloud()

            st.image(img)



        elif Analyzer_choice=="Visualize result":



            
            def Plot_Analysis():

                st.success("Generating Visualisation")

                


                posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")

                df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
                df2 = pd.DataFrame([tweet.created_at for tweet in posts], columns=['created_at'])
                df['date'] = df2['created_at']
                
                df['Clean_Text'] = ""
                
                for i in range(len(df)):
                        df['Clean_Text'][i] = replace_word(df['Tweets'][i])
                
                
                
                
                df['score'] = ""
                
                df['positive'] = ""
                df['negative'] = ""
                df['definitenegativescore'] = ""
                
                negativescore = 0
                positivescore = 0
                definitenegativescore = 0
                
                for i in range(len(df)):
                    negativescore = 0
                    positivescore = 0
                    definitenegativescore = 0
                    for key in content_listn:
                        key = " " + key
                        if key in df['Clean_Text'][i]:
                            negativescore  += 1
                            
                    for key1 in content_listp:
                        key1 = " " + key1
                        if key1 in df['Clean_Text'][i]:
                            positivescore  += 1
                    for key3 in content_listdn:
                        key3 = " " + key3
                        if key3 in df['Clean_Text'][i]:
                            definitenegativescore += 1
                    if positivescore > 0 or negativescore > 0 or definitenegativescore > 0:
                        if (negativescore > positivescore) or definitenegativescore > 0:
                            df['score'][i] = 1
                        else:
                            df['score'][i] = 0
                    # if positivescore <= 0 and negativescore <= 0 and neutralscore <= 0:
                    #     df['score'][i] = "N/A"
                    # if df['score'][i] != 'depressed' and df['score'][i] != 'not depressed' and df['score'][i] != 'N/A':
                    #     df['score'][i] = "N/A"
                    df['positive'][i] = positivescore
                    df['negative'][i] = negativescore + definitenegativescore
                 #df = df.drop(columns=['definitenegativescore'])
                 #df['definitenegativescore'][i] = definitenegativescore

                def getAnalysis(score):
                  if score == 1:
                    return 'Positive'
                  elif score == 0:
                    return 'Negative'
                  else:
                    return 'N/A'
                    
                df['Type of tweets'] = df['score'].apply(getAnalysis)
                return df

            
            df= Plot_Analysis()



            st.write(sns.countplot(x=df["Type of tweets"],data=df))

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(use_container_width=True)

        else:
            def Plot_Analysis():

                st.success("Generating Visualisation")

                


                posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")

                df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
                df2 = pd.DataFrame([tweet.created_at for tweet in posts], columns=['created_at'])
                df['date'] = df2['created_at']
                
                df['Clean_Text'] = ""
                
                for i in range(len(df)):
                        df['Clean_Text'][i] = replace_word(df['Tweets'][i])
                
                
                
                
                df['score'] = ""
                
                df['positive'] = ""
                df['negative'] = ""
                df['definitenegativescore'] = ""
                
                negativescore = 0
                positivescore = 0
                definitenegativescore = 0
                
                for i in range(len(df)):
                    negativescore = 0
                    positivescore = 0
                    definitenegativescore = 0
                    for key in content_listn:
                        key = " " + key
                        if key in df['Clean_Text'][i]:
                            negativescore  += 1
                            
                    for key1 in content_listp:
                        key1 = " " + key1
                        if key1 in df['Clean_Text'][i]:
                            positivescore  += 1
                    for key3 in content_listdn:
                        key3 = " " + key3
                        if key3 in df['Clean_Text'][i]:
                            definitenegativescore += 1
                    if positivescore > 0 or negativescore > 0 or definitenegativescore > 0:
                        if (negativescore > positivescore) or definitenegativescore > 0:
                            df['score'][i] = 0
                        else:
                            df['score'][i] = 1
                    # if positivescore <= 0 and negativescore <= 0 and neutralscore <= 0:
                    #     df['score'][i] = "N/A"
                    # if df['score'][i] != 'depressed' and df['score'][i] != 'not depressed' and df['score'][i] != 'N/A':
                    #     df['score'][i] = "N/A"
                    df['positive'][i] = positivescore
                    df['negative'][i] = negativescore + definitenegativescore
                 #df = df.drop(columns=['definitenegativescore'])
                 #df['definitenegativescore'][i] = definitenegativescore

            #df = df[date,score,positive,negative]
                return df

            
            df= Plot_Analysis()
            

            
            df = df.drop(columns=['Tweets',"Clean_Text","definitenegativescore"])
            df = df.rename(columns={'date':'index'}).set_index('index')
            st.line_chart(df)

            #df = df.set_index('date')
            #ax = df.plot(secondary_y='Sentiment')






    st.subheader(' ------------------------Created By :  Bernice Yeow ---------------------- :sunglasses:')


            

                


























if __name__ == "__main__":
    app()