import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='Analisis Sentimen Review Hotel Menggunakan LSTM by Fazano Fikri El Huda')

data_path = ("review_hotel.csv")
@st.cache(persist=True)
def load_data():
    data = pd.read_csv(data_path)
    return data
data = load_data()

df = data.copy()
df["cat"] = df["category"]
df['cat'] = df['cat'].replace(1,"Positif")
df['cat'] = df['cat'].replace(0,"Negatif")

st.title('Analisis Sentimen Review Hotel Menggunakan LSTM')
st.markdown('Dataset : https://github.com/rakkaalhazimi/Data-NLP-Bahasa-Indonesia/blob/main/review_hotel.csv')
st.write(data.head(5))

st.sidebar.title("Eksplorasi review hotel")

#grafik
st.sidebar.subheader("Jumlah sentimen")
select = st.sidebar.selectbox('Pilih Visualisasi',['Histogram','PieChart'])

sentiment_count = df['cat'].value_counts(dropna=False).sort_index()
sentiment_count = pd.DataFrame({'Sentimen':sentiment_count.index,'Jumlah':sentiment_count.values})

if st.sidebar.checkbox('Show',False,key='0'):
    st.subheader("Visualisasi jumlah sentimen pada review hotel")
    if select=='Histogram':
        fig = px.bar(sentiment_count,x='Sentimen',y='Jumlah',color='Jumlah',height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count,values='Jumlah',names='Sentimen')
        st.plotly_chart(fig)

#wordcloud
st.sidebar.subheader("WordCloud review")
word_sentiment = st.sidebar.radio("Pilih sentimen", tuple(pd.unique(df["cat"])))
if st.sidebar.checkbox("Show", False, key="6"):
    st.subheader(f"Word Cloud review pada sentimen {word_sentiment.capitalize()}")
    df = df[df["cat"]==word_sentiment]
    words = " ".join(df["review_text"])
    wordcloud = WordCloud(background_color="white", width=800, height=640).generate(words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

#prediksi
model = tf.keras.models.load_model("model")
st.subheader('Review hotel Anda')
input_review = st.text_area('Coba tulis review hotel yang pernah Anda inap dibawah')
if input_review:
    prediksi = model.predict([input_review])
    if prediksi.squeeze()>0.5:
        st.write('Review Anda positif :yum:')
    else:
        st.write('Review Anda negatif :neutral_face:')
else:
    pass
