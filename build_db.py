import os
import chromadb
import ollama
from pypdf import PdfReader

# ===== 1. 初始化資料庫 =====
client = chromadb.PersistentClient(path="./db")

# 每次重建資料庫，清空舊有的 collection
try:
    client.delete_collection("nqu")
    print("已成功刪除舊的 nqu collection")
except Exception:
    pass

collection = client.get_or_create_collection(name="nqu")

# ===== 2. 定義文字讀取與切塊函式 =====
def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # 確保有成功擷取到文字
                text += extracted_text + "\n"
        return text
    except Exception as e:
        print(f"❌ 讀取 PDF 失敗 {path}: {str(e)}")
        return ""

def read_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ 讀取 TXT 失敗 {path}: {str(e)}")
        return ""

def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    return chunks

# ===== 3. 主要處理流程 =====
folders = ["data", "knowledge"]
doc_id = 0

for folder in folders:
    # 檢查資料夾是否存在，不存在就自動建立，避免程式報錯
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 已自動建立空的資料夾: {folder}")
        continue

    print(f"\n📂 開始處理資料夾: {folder}")
    
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        
        # 根據副檔名讀取文字
        if file.endswith(".pdf"):
            print(f"📄 正在讀取 PDF: {file}")
            text = read_pdf(path)
        elif file.endswith(".txt"):
            print(f"📝 正在讀取 TXT: {file}")
            text = read_txt(path)
        else:
            continue  # 忽略其他格式的檔案

        # 如果讀出來是空的（例如掃描版無文字的 PDF），則跳過
        if not text.strip():
            print(f"⚠️ 檔案 {file} 內容為空，跳過。")
            continue

        # 文字切塊
        chunks = chunk_text(text)
        print(f"   ➔ 切割成 {len(chunks)} 個文字區塊，開始匯入向量資料庫...")

        # 將每個區塊轉成向量並存入 ChromaDB
        for idx, chunk in enumerate(chunks):
            try:
                # 呼叫 Ollama 生成向量
                response = ollama.embed(
                    model="bge-m3:latest",
                    input=chunk
                )
                emb = response["embeddings"][0]

                # 存入資料庫
                collection.add(
                    ids=[f"doc_{doc_id}"], # 建議加上前綴，結構更清晰
                    embeddings=[emb],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source_file": file,
                            "folder": folder,
                            "chunk_index": idx
                        }
                    ]
                )
                doc_id += 1
            except Exception as e:
                print(f"❌ 區塊 {idx} 寫入資料庫失敗: {str(e)}")

print(f"\n✅ 向量資料庫建立完成！共匯入 {doc_id} 筆文獻區塊。")
