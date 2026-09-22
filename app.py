import streamlit as st
import chromadb
import ollama

# ===== 1. 頁面初始化 =====
st.set_page_config(
    page_title="金門大學校史智慧導覽助理",
    page_icon="🎓",
    layout="wide"
)

# ===== 2. 快取載入資源（避免重複連接與讀取） =====
@st.cache_resource
def init_resources():
    # 連接 Vector DB
    client = chromadb.PersistentClient(path="./db")
    # 注意：這裡直接讀取，不進行刪除（delete）。資料應由單獨的入庫腳本處理。
    collection = client.get_or_create_collection(name="nqu")
    
    # 讀取系統 Prompt 模板
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        template = "參考資訊：\n{context}\n\n問題：{question}\n請根據參考資訊回答。"
        
    return collection, template

collection, template = init_resources()

# ===== 3. Session 狀態初始化 =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== 4. 介面標題與資訊 =====
st.title("🎓 金門大學校史智慧導覽助理")
st.caption("系統架構：qwen2.5:7b + bge-m3 + ChromaDB (RAG)")

# ===== 5. 顯示對話歷史 =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== 6. 使用者輸入與對話處理 =====
if question := st.chat_input("請問關於金門大學的校史、創校理念或校園景點..."):

    # 紀錄並顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 處理 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("正在翻閱校史文獻..."):
            try:
                # 6-1. 向量化問題 (Embedding)
                query_embedding = ollama.embed(
                    model="bge-m3:latest",
                    input=question
                )["embeddings"][0]

                # 6-2. 檢索知識庫 (Retrieval)
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5
                )

                # 檢查是否有查到資料
                if result and result["documents"] and result["documents"][0]:
                    context = "\n\n".join(result["documents"][0])
                else:
                    context = "查無相關校史資料。"

                # 6-3. 建立 Prompt
                prompt = template.format(context=context, question=question)

                # 6-4. 呼叫大模型並使用串流輸出 (Generation with Streaming)
                response_stream = ollama.chat(
                    model="qwen2.5:7b",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True  # 開啟串流
                )

                # 定義產生器讓 Streamlit 逐字渲染
                def parse_stream(stream):
                    for chunk in stream:
                        yield chunk["message"]["content"]

                # 逐字顯示在介面上並取得完整文字
                answer = st.write_stream(parse_stream(response_stream))

            except Exception as e:
                answer = f"❌ 系統發生錯誤：{str(e)}"
                st.error(answer)

    # 紀錄 AI 回答至歷史紀錄
    st.session_state.messages.append({"role": "assistant", "content": answer})
