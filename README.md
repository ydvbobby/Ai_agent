# 🤖 AI Agent – Chat with PDFs, YouTube, and the Web

This is a multi-functional, intelligent AI assistant built with **LangChain**, **Streamlit**, **FAISS**, and **HuggingFace**. It allows you to:

- 💬 **Chat with PDFs** – Ask questions about your uploaded documents.
- 🎥 **Summarize & query YouTube videos** – Just paste a YouTube link.
- 🌐 **Search the web and get live weather** – Using tools integrated via LangChain agents.

> 🧪 Built as part of my AI learning journey to explore agentic workflows, document understanding, RAG pipelines, and tool integration.

---

## 🚀 Live Demo

🔗 [Try it on Streamlit](https://aiagent1950.streamlit.app)  
📂 [Source Code on GitHub](https://github.com/ydvbobby/Ai_agent)

---

## ✨ Features

### 📄 PDF Chatbot
- Upload any PDF and ask natural language questions.
- Uses **HuggingFace embeddings**, **FAISS vector store**, and **LangChain retrievers**.

### 🎥 YouTube Summarizer
- Paste a YouTube link and query its transcript.
- Automatic transcript fetching and chunked analysis.
  
### 💬 Chatbot with Tools
- Conversational LLM agent powered by **Mixtral** or **Gemini**.
- Tools integrated via LangChain:
  - 🔍 Google Search using SerpAPI
  - 🌦️ Weather reports using OpenWeather API

---

## 🧰 Tech Stack

| Technology | Description |
|------------|-------------|
| `LangChain` | Core framework for agents and retrievers |
| `FAISS` | Vector storage for similarity search |
| `HuggingFace` | LLM and embeddings (`all-MiniLM-L6-v2`) |
| `Streamlit` | Web interface for interaction |
| `Google Generative AI` | Gemini 1.5 Flash model |
| `SerpAPI` | Web search integration |
| `OpenWeatherMap API` | Weather tool integration |
| `YouTube Transcript API` | To fetch video transcripts |

---

## 📦 Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/ydvbobby/Ai_agent.git
cd Ai_agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API keys in a .env file
🔑 .env Format

SERPAPI_API_KEY=your_serpapi_key
WEATHER_API_KEY=your_openweather_key
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id



🛠️ Future Improvements
✅ PDF summarization feature (currently under maintenance)
📊 Improve UI layout and error handling
🌐 Add support for more file types (e.g., DOCX, TXT)
🗃️ User chat session persistence
🚀 HuggingFace agent model selector from UI


🙋‍♂️ Author
👤 Bobby Kumar
🎓 AI/ML Enthusiast | LangChain Developer
📬 Connect on LinkedIn
🌐 [Visit my portfolio (optional link)]

⭐️ Show Your Support
If you found this project useful:

🌟 Star this repo

🍴 Fork and try your own version

🧠 Share feedback or issues!