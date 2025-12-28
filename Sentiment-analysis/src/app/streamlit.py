import requests
import streamlit as st




API_URL = "http://127.0.0.1:8000"


    
col1, col2 = st.columns([3, 15], vertical_alignment="center")

with col1:
    st.markdown("[![Amazon Logo](https://www.pngall.com/wp-content/uploads/15/Amazon-Logo-White.png)](https://www.amazon.fr/)")

with col2:
    st.title("Amazon Sentiment Analysis")





st.subheader("✍️ Print your review...")
review_text = st.text_area("qbc", height=150, placeholder="I liked the product", label_visibility="hidden")

if st.button('Analyze', key=1):

    response = requests.post(f"{API_URL}/predict-sentiment", params={"review": review_text})

    if response.status_code == 200:
        prediction = response.json()
        sentiment = prediction
        st.success(f"{sentiment}")
    else:
        st.error(f"Erreur {response.status_code} : {response.text}")



# st.subheader("📂 Upload your reviews...")
# message = """Formats acceptés : .csv/.xlsx
# \nLe fichier doit contenir deux colonnes "Text" et "Sentiment". """

# file = st.file_uploader("abv", help=message, label_visibility="hidden")





# if st.button('Analyze', key=2):
    
#     response = requests.post(f"{API_URL}/predict-sentiment")

#     if response.status_code == 200:
#         prediction = response.json()
#         sentiment = prediction
#         st.success(f"{sentiment}")
#     else:
#         st.error(f"Erreur {response.status_code} : {response.text}")



# if file is not None:
#     st.success("Fichier téléchargé avec succès !")

