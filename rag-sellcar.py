import os
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from PyPDF2 import PdfReader
from pythainlp import word_tokenize
import speech_recognition as sr
from gtts import gTTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import base64

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("พูดคำถามของคุณ...")
        audio = r.listen(source)
        st.write("กำลังประมวลผล...")
    try:
        text = r.recognize_google(audio, language='th-TH')
        return text
    except sr.UnknownValueError:
        st.error("ไม่สามารถรับรู้เสียงได้ กรุณาลองใหม่")
    except sr.RequestError:
        st.error("ไม่สามารถเข้าถึงบริการรับรู้เสียงได้")
    return None

def synthesize_speech(text, output_filename):
    tts = gTTS(text=text, lang='th')
    tts.save(output_filename)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                thai_text = word_tokenize(page_text, engine='newmm')
                text += " ".join(thai_text)
            print(f"Processed PDF: {pdf.name}, Text length: {len(text)}")
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    คุณเป็นพนักงานขายรถยนต์ที่มีความสุภาพและกระตือรือร้น คุณมีความรู้เกี่ยวกับรถยนต์อย่างลึกซึ้งและต้องการช่วยลูกค้าให้ได้รถที่เหมาะสมที่สุด 
    ให้ตอบคำถามโดยใช้ข้อมูลจากบริบทที่ให้มา พยายามใช้คำศัพท์ที่น่าสนใจและโน้มน้าวใจเพื่อดึงดูดความสนใจของลูกค้า 
    หากไม่มีข้อมูลในบริบท ให้ตอบอย่างสุภาพว่าคุณไม่มีข้อมูลนั้นและเสนอที่จะหาข้อมูลเพิ่มเติมให้

    บริบท:\n {context}?\n
    คำถาม: \n{question}\n

    คำตอบ:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio id="myAudio" autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                var audio = document.getElementById("myAudio");
                audio.play().catch(function(error) {{
                    console.log("Autoplay prevented. Please interact with the page to enable audio.");
                }});
            </script>
            """
        st.markdown(md, unsafe_allow_html=True)

def main():
    st.set_page_config("เซลล์ขายรถ BYD AI", layout="wide")
    
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 100%;
    }
    .chat-message .message {
      width: 80%;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    with st.sidebar:
        st.title("เมนู:")
        pdf_docs = st.file_uploader("อัพโหลดไฟล์ PDF", accept_multiple_files=True)
        if st.button("ประมวลผล PDF"):
            with st.spinner("กำลังประมวลผล..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("เสร็จสิ้น")

    st.header("เซลล์ขายรถ BYD AI")

    # แสดงประวัติการสนทนา
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file)
        autoplay_audio(st.session_state.audio_file)

    user_question = st.text_input("พิมพ์คำถามของคุณ")
    
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        if st.button("🎤"):
            user_question = transcribe_audio()
            if user_question:
                st.text_input("พิมพ์คำถามของคุณ", value=user_question)

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        response = user_input(user_question)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        audio_file = "response.mp3"
        synthesize_speech(response, audio_file)
        st.session_state.audio_file = audio_file

        # แสดงข้อความและเล่นเสียงล่าสุดทันที
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.audio(audio_file)
        autoplay_audio(audio_file)

        # เคลียร์ค่า input เพื่อพร้อมรับคำถามใหม่
        st.text_input("พิมพ์คำถามของคุณ", value="", key="new_question")

if __name__ == "__main__":
    main()