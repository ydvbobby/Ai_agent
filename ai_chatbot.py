import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings   
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain import hub
import requests

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate





#define model
llm = HuggingFaceEndpoint(
   repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
   task="text-generation",
)

model = ChatHuggingFace(llm=llm)


# Sidebar static menu
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["PDF Reader", "Simple Chatbot", "YouTube Summarizer"]
)










# Page content
if page == "PDF Reader":
    st.title("📄 PDF Reader")
    st.write("Upload and read PDF files here...")
    
    from langchain.document_loaders import PyPDFLoader
    import tempfile
    
    file = st.file_uploader("Upload Pdf", type="pdf")
    if file:
           with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
              temp_file.write(file.read())
              temp_file_path = temp_file.name
            
           loader = PyPDFLoader(temp_file_path)
           documents = loader.load()
           
         
           text_document = "\n".join(document.page_content for document in documents)
           
     
           
           
           summarize = st.button("summarize pdf")
           query = st.text_input("enter query")
           
           if summarize:
                  st.write('This feature  is under maintainance')
           if query:
                  
                  
                         
                  #text splitting
                  splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 150)
                  chunks = splitter.create_documents([text_document])
                  
                  #embedding 
                  embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
                  
                  #create a vector store
                  vector_store1 = FAISS.from_documents(chunks, embeddings)
            
                  #define retreivers
                  retriever = vector_store1.as_retriever(search_type="similarity", search_kwargs={"k":3})
               
               
                  def get_context(retreived_data):
                     context = " ".join(docs.page_content for docs in retreived_data)
                     return context
                     
                  
                  
                  prompt = PromptTemplate(
                     template="""
                         You are ai assistant who the answer to the questions based on the provided chunks as context.
                         If the context is not enough just say I don't know
                     
                         context: {context}\n
                         question: {question}""",
                     
                         input_variables=["context", "question"]
                     
                     )
                  
                  model = model
                  parser = StrOutputParser()
                  
                  parallel_chain = RunnableParallel({
                      "context": retriever|RunnableLambda(get_context),
                      "question": RunnablePassthrough()
                   })
                         
                  main_chain = parallel_chain | prompt | model | parser
                         
                  result = main_chain.invoke(query)
                  st.write(result)
                    
                  
                  
                  
                  
                   
           
            
    
     
    
    
    
    
    

elif page == "Simple Chatbot":
    
    load_dotenv()

    #define tool
    @tool
    def google_search(query: str) -> str:

        """Use this tool to search the web using Google via SerpAPI"""

        search = GoogleSearch({
           "q": query,
           "api_key": os.getenv("SERPAPI_API_KEY")
        })

        results = search.get_dict()

        if "error" in results:
            return "Search failed. Try again later."

        # ✅ Convert top 3 results to plain readable string
        output = "🔎 Top Google Search Results:\n\n"
        for idx, res in enumerate(results.get("organic_results", [])[:3], start=1):
           title = res.get("title", "No title")
           link = res.get("link", "No link")
           snippet = res.get("snippet", "No summary available")
           output += f"{idx}. {title}\n{snippet}\n{link}\n\n"

        return output.strip()

    @tool
    def get_weather(city:str):
        """ This function gets current weather report of any given city name as a string."""

        city = city = str(city).strip().strip('"').strip("'").replace('\n', '').replace('\r', '')
        print("city name =(" ,city, ")")


        API_key = os.getenv("WEATHER_API_KEY")
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}&units=metric'
        response = requests.get(url)
        return response.json()

    # Configure Gemini
    prompt = hub.pull('hwchase17/react')
    model = model

   
    if 'chat_history' not in st.session_state:
       st.session_state.chat_history = ChatMessageHistory()

# create agent
    agent = create_react_agent(
       llm = model, 
       tools=[google_search,get_weather],
       prompt=prompt
    )

    agent_executor = AgentExecutor(
       agent = agent,
       tools = [google_search, get_weather],
       verbose=True
    )

    # Chat function
    def format_chat_history(history):
        messages = history.messages
        formatted = ""
        for msg in messages:
           role = "User" if msg.type == "human" else "AI"
           formatted += f"{role}: {msg.content}\n"
        return formatted

    def run_chain(question):
       chat_history_text = format_chat_history(st.session_state.chat_history)
       full_input = f"{chat_history_text}User: {question}\nAI:"
    
       response = agent_executor.invoke({"input": full_input})
       ai_reply = response["output"]
    
       st.session_state.chat_history.add_user_message(question)
       st.session_state.chat_history.add_ai_message(ai_reply)
       return ai_reply

    # Streamlit UI
    st.title('🤖 Personal AI Chatbot🤖')
    user_input = st.text_input('Ask Anything 😊 ! ')
    if user_input:
       ai_response = run_chain(user_input)
       st.write('You: ', user_input)
       st.write('AI: ', ai_response)
   
    st.subheader('📜 Chat History')
    for msg in st.session_state.chat_history.messages:
       st.write(f"{msg.type.capitalize()}: {msg.content}")


    
    
    
    
    
    
    
    
    
    
    
    

elif page == "YouTube Summarizer":
    st.title("🎥 YouTube Summarizer")
    st.write("Paste a YouTube link and get the summary...")
    
    
     
    url = st.text_input("enter url")
    
    def get_video_id(url):
      video_id = url.split('/')[3].split('?')[0]
      return str(video_id)
   
    video_id_list = []
   

    if url:
           
         video_id = get_video_id(url)
         video_id_list.append(video_id)
        

         #get transcript
         try:
            transcript = YouTubeTranscriptApi.get_transcripts(video_id_list, languages=['en'])
            transcript = " ".join(chunks['text'] for chunks in transcript[0][video_id])
            
         except Exception as e:
             st.write(e)
             
         

         # split The text 
         splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=150)
        
         chunks = splitter.create_documents([transcript])
         
         #create embeddings
        
         embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

         #create a vector database
         vector_store = FAISS.from_documents(chunks, embeddings)
         

         #define retreiver and model
         retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})
         model= model
         
         
         st.subheader("Data Fethch Successfull")
         summarize = st.button('summarize video')
         st.write("Or ask Query")
         query = st.text_input('input your query')
         
         if summarize:
                st.write("This feature is under maintainance")
         if query:
                prompt = PromptTemplate(
                   template = """ You are an helpful Ai assistant who can read youtube transcripts as context and answer to the query.
                   If the context is insufficient just say, I don't know. 
                   
                   context = {context}\n 
                   query = {query}  """, 
                   
                   input_variables=["context", "query"]
                )
                
                parser = StrOutputParser()
                
                def get_context(retreived_doc):
                       context = " ".join(docs.page_content for docs in retreived_doc)
                       return context
                
                parallel_chain = RunnableParallel({
                   "context": retriever | RunnableLambda(get_context),
                   "query": RunnablePassthrough()
                })
                
                main_chain = parallel_chain | prompt | model | parser
                
                result = main_chain.invoke(query)
                
                st.write(result)
                    

         
        
    


