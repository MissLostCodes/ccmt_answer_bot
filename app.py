#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[49]:


import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import streamlit_analytics




# In[16]:


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ## load pdfs 

# In[11]:


# Load all PDFs
brochure_docs = PyPDFLoader("ccmt_info.pdf").load()
flowchart_docs = PyPDFLoader("ccmt_flowcharts_2.pdf").load()
fee_docs = PyPDFLoader("fee_table.pdf").load()

# Add metadata (tags)
for doc in fee_docs:
    doc.metadata["source"] = "fee_clean"   

for doc in brochure_docs:
    doc.metadata["source"] = "brochure"

for doc in flowchart_docs:
    doc.metadata["source"] = "flowchart"


# ### chunking

# In[12]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(brochure_docs + flowchart_docs + fee_docs)


# In[14]:


print("Type of variable:", type(chunks))
print()
print("Type of each object inside the list:", type(chunks[0]))
print()
print("Total number of documents inside list:", len(chunks))
print()



# In[18]:


import re

def clean_text(text):
    # Remove invalid surrogate characters
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    # Optional: Remove emojis or other symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

# Apply to chunks
for doc in chunks:
    doc.page_content = clean_text(doc.page_content)


# In[19]:


embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=GOOGLE_API_KEY)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)
vectordb.persist()


# In[22]:


retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# In[ ]:





# ## rag chain 

# In[24]:


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# In[29]:


chat_model=ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY , model='gemini-2.0-flash-exp')
parser = StrOutputParser()


# ### retriever: A vector retriever (e.g., FAISS or Chroma) — returns the most relevant k documents based on similarity to the question.
# ### format_docs: A function to convert those document objects into a single formatted string (so it can go into the prompt).

# In[28]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[30]:


rag_chain = {"context": retriever | format_docs , "question": RunnablePassthrough()} | prompt_template | chat_model | parser


# ## Streamlit UI
# 

# In[50]:


# Set page config at the very beginning
st.set_page_config(page_title="CCMT Chatbot", layout="wide")

# ----------- Unique Visit Logging Logic -------------
COUNT_FILE = "user_count.txt"

# Create the file if it doesn't exist
if not os.path.exists(COUNT_FILE):
    with open(COUNT_FILE, "w") as f:
        f.write("")

# Only add one star per session
if "counted" not in st.session_state:
    with open(COUNT_FILE, "a") as f:
        f.write("*")
    st.session_state.counted = True

# Read total user count
with open(COUNT_FILE, "r") as f:
    stars = f.read()
    total_users = stars.count("*")

# ----------- Main Chatbot UI ------------------------
st.title("🎓 CCMT Counselling Chatbot")
st.markdown("Ask anything about CCMT rules, rounds, fees, etc...")

# Background styling (optional)
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Show unique user visits in the sidebar
st.sidebar.markdown(f"🌟 **Unique Visits:** {total_users}")

# User Query Input
query = st.text_input("💬 Ask your question:")

# Submit button
if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Thinking..."):
            result = rag_chain.invoke(query)
            st.markdown("**Answer:**")
            st.write(result)
    else:
        st.warning("Please enter a question first.")


# In[ ]:





# In[47]:





# In[ ]:




