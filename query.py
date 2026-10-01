import streamlit as st
import chromadb
import ollama
import os

# ===== 1. 頁面初始化 =====
st.set_page_config(
    page_title="金門大學校史 AI 助手",
    page_icon="🎓",
    layout="wide"
)

# ===== 2. 快取載入資源與外部 Prompt =====
@st.cache_resource
def init_resources():
    # 連接資料庫（只讀取，不刪除！）
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection(name="nqu")
    
    # 自動尋找並讀取與 query.py 同目錄下的 prompt.txt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "prompt.txt")
    
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        # 防呆備用模板
        template = "參考資料：\n{context}\n\n問題：{question}\n請回答。"
        
    return collection, template

collection, template = init_resources()

# ===== 3. 初始化對話紀錄 (Session State) =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== 4. 介面標題 =====
st.title("🎓 金門大學校史 AI 助手")
st.caption("系統架構：Qwen2.5:7b + bge-m3 + ChromaDB (RAG)")

# 側邊欄加上清空按鈕，方便隨時切換情境測試
if st.sidebar.button("🧹 清空對話紀錄"):
    st.session_state.messages = []
    st.rerun()

# ===== 5. 渲染歷史對話紀錄 =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📚 資料來源：{msg['sources']}")

# ===== 6. 使用者輸入處理 =====
question = st.chat_input("請輸入你的問題...")

if question:
    # 顯示並儲存使用者問題
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 處理 AI 回答
    with st.chat_message("assistant"):
        with st.spinner("正在翻閱校史文獻並思考中..."):
            try:
                # 6-1. 向量化問題
                emb_response = ollama.embed(
                    model="bge-m3:latest",
                    input=question
                )
                
                # 自動剝除多層括號殼，避免三維陣列報錯
                raw_emb = emb_response["embeddings"]
                while isinstance(raw_emb, list) and len(raw_emb) > 0 and isinstance(raw_emb, list):
                    raw_emb = raw_emb
                query_embedding = raw_emb

                # 6-2. 從 ChromaDB 搜尋最相關的 2 筆（避免雜訊干擾）
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,
                    include=["documents", "distances", "metadatas"]
                )

                # 6-3. 建立上下文與來源標籤
                if not result or not result["documents"] or not result["documents"]:
                    context = "（目前向量資料庫未檢索到任何相關段落）"
                    source_string = "未知來源"
                else:
                    documents = result["documents"]
                    sources_metadata = result["metadatas"]
                    context = "\n\n".join(documents)
                    
                    # 提取來源檔案名稱
                    source_list = set()
                    for item in sources_metadata:
                        src = item.get("source_file") or item.get("source") or "未知來源"
                        source_list.add(src)
                    source_string = ", ".join(source_list)

                # 🔥【核心優化】：移除了原本寫死的 Python 阻擋機制
                # 這樣所有「無關問題」才能順利進入大模型，讓 prompt.txt 中的拒絕規則生效。

                # 6-4. 帶入外部 Prompt 模板
                prompt = template.format(context=context, question=question)

                # 6-5. 呼叫大模型並串流輸出
                # 顯示資料來源（如果大模型判定無關，這行仍會作為系統紀錄顯示，或可依個人喜好移動位置）
                st.caption(f"📚 系統檢索來源：{source_string}")
                
                placeholder = st.empty()
                full_response = ""

                stream = ollama.chat(
                    model="qwen2.5:7b",  # 完全與你的 chat.py 保持一致
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )

                for chunk in stream:
                    token = chunk["message"]["content"]
                    full_response += token
                    placeholder.markdown(full_response)

                # 儲存助手回答
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": source_string
                })

            except Exception as e:
                error_msg = f"❌ 系統發生錯誤：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
