import streamlit as st 
from nltk.sentiment import  SentimentIntensityAnalyzer 
import pandas as pd 
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt  
import nltk   
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore") 

nltk.download('vader_lexicon') 

api_key=st.text_input("Enter your Groq API key",type="password") 
if api_key:
  LLm=ChatGroq( 
      api_key=api_key,
      model="llama3-8b-8192",
      temperature=0.5
  )
else:
  st.warning("Please enter your Groq API key to use the model.")







sia=SentimentIntensityAnalyzer() 
st.title("🧠 VADER Sentiment Analysis (Positive, Negative, Neutral)")
st.markdown("Enter any text below and get a sentiment rating using NLTK's VADER.")

dataset=st.file_uploader("upload your file",type=["CSV"])

if dataset:
  df=pd.read_csv(dataset)
  if "text" not in df.columns:  
    st.error("CSV must contain a 'text' column") 

  else:
    df['values']=df['text'].apply(lambda x: sia.polarity_scores(str(x))) 
    df['compound']=df["values"].apply(lambda x: x["compound"])
    df['sentiment']=df['compound'].apply(lambda x: 'Positive' if x>=0.05 else ('Negative' if x<=-0.05 else 'Neutral'))



    st.dataframe(df[['text','compound','sentiment']].head())

 
    #visulzation

    sentiment_values=df['sentiment'].value_counts()
    fig1,plot=plt.subplots()
    plot.pie(sentiment_values,labels=sentiment_values.index,autopct='%1.1f%%',colors=sns.color_palette("pastel"))
    plot.axis("equal")
    st.pyplot(fig1)
    st.success("Analysis Complete")
    if st.checkbox("show full Analysis"): 
        st.dataframe(df[['text','compound','sentiment']])
        st.balloons() 


    if st.button("ask you Gemini about this data"):
        prompt = f"""This is the sentiment distribution from VADER analysis:
    {sentiment_values.to_string()}
    Can you analyze and give insights about the overall sentiment of this dataset?"""

        response = LLm.invoke(prompt)
        st.write(response.content)
