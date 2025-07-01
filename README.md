# GeminiDocChat

## Description
A Streamlit app to chat with your own documents (PDF, DOCX, or web URLs) using Google Gemini AI. Upload files or enter a URL, process them, and ask questions in a conversational interface.

## Features
- Upload PDF or DOCX files, or provide a URL
- Extracts and splits text for efficient retrieval
- Uses Google Gemini for conversational Q&A
- Maintains chat history for context-aware answers

## Installation
```bash
# Clone the repository
git clone https://github.com/Mohammedosama111/GeminiDocChat.git
cd  GeminiDocChat  

# (Optional) Create a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup
Create a `.env` file in the project root with your Google Gemini API key:
```
GOOGLE_API_KEY=your_google_gemini_api_key
```

## Usage
```bash
# Run the Streamlit app
streamlit run app.py
```
- Open the provided local URL in your browser.
- Upload PDF/DOCX files or enter a URL in the sidebar.
- Click "Process" to load your documents.
- Ask questions in the chat input.

## Project Structure
```
 GeminiDocChat/
  app.py            # Main Streamlit app
  requirements.txt  # Python dependencies
  README.md         # Project documentation
```

## Data Flow
1. **User Input**: The user uploads PDF/DOCX files or enters a URL in the sidebar.
2. **Data Loading**: Files are temporarily saved and loaded using PyPDFLoader (for PDFs) or Docx2txtLoader (for DOCX). URLs are loaded using WebBaseLoader. The text is extracted from these sources.
3. **Text Splitting**: The extracted text is split into manageable chunks using RecursiveCharacterTextSplitter.
4. **Vector Store Creation**: Text chunks are embedded using GoogleGenerativeAIEmbeddings and stored in a FAISS vector store for efficient retrieval.
5. **Conversational Chain**: A conversational retrieval chain is created using ChatGoogleGenerativeAI and ConversationalRetrievalChain, with chat history managed by ConversationBufferMemory.
6. **Chat Interaction**: The user asks questions in the chat. The app retrieves relevant information from the vector store and generates answers using the Gemini model, maintaining context with chat history.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
Specify your license here (e.g., MIT, Apache 2.0, etc.)

## Contact
For questions or suggestions, contact [your-email@example.com](mailto:your-email@example.com). 