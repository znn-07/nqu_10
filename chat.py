import sys
import chromadb
import ollama
import os

# ===== 1. 連接資料庫（只讀取，不刪除！） =====
client = chromadb.PersistentClient(path="./db")

try:
    collection = client.get_collection(name="nqu")
    print("✅ 成功連接金門大學校史知識庫。")
except Exception:
    print("❌ 找不到名為 'nqu' 的知識庫，請先執行 build_db.py 匯入資料！")
    sys.exit(1)

# ===== 2. 讀取外部 Prompt 模板 =====
current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(current_dir, "prompt.txt")

try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    print("📝 成功載入自訂 prompt.txt 模板！")
except FileNotFoundError:
    print(f"⚠️ 在路徑 [{prompt_path}] 找不到 prompt.txt，將使用預設 RAG 模板。")
    template = "參考文獻：\n{context}\n\n問題：{question}\n請根據文獻回答："

print("🤖 校史智慧導覽助理已上線（輸入 'exit' 可結束對話）")

# ===== 3. 對話主迴圈 =====
while True:
    try:
        question = input("\n🙋 請輸入問題：").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if not question:
        continue

    if question.lower() == "exit":
        print("👋 謝謝使用，再見！")
        break

    print("🔍 正在檢索校史文獻並思考中...", end="", flush=True)

    try:
        # 3-1. 向量化問題 (Embedding)
        emb_response = ollama.embed(
            model="bge-m3:latest",
            input=question
        )
        
        raw_emb = emb_response["embeddings"]
        while isinstance(raw_emb, list) and len(raw_emb) > 0 and isinstance(raw_emb[0], list):
            raw_emb = raw_emb[0]
        query_embedding = raw_emb

        # 3-2. 檢索知識庫 (Retrieval) - 限制為 2 筆，避免無關雜訊
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        # 組合上下文
        if result and result["documents"]:
            # 自動適應 ChromaDB 的二維或三維回傳格式
            docs_layer = result["documents"][0] if isinstance(result["documents"][0], list) and isinstance(result["documents"][0][0], str) else result["documents"]
            metas_layer = result["metadatas"][0] if isinstance(result["metadatas"][0], list) and isinstance(result["metadatas"][0][0], dict) else result["metadatas"]
            
            context_list = []
            for doc, meta in zip(docs_layer, metas_layer):
                # 取得該區塊真實的檔案名稱
                filename = meta.get("source_file") or meta.get("source") or "未知來源.txt"
                # 強行將檔名標籤與內容黏在一起
                context_list.append(f"【文獻來源檔案：{filename}】\n{doc}")
            
            context = "\n\n".join(context_list)
        else:
            context = "（未檢索到相關文獻）"

        # 3-3. 帶入外部 Prompt 模板（自動取代 template 中的變數）
        prompt = template.format(
            context=context,
            question=question
        )

        print("\n🤖 助理回答：")

        # 3-4. 呼叫大模型並啟用「終端機串流打字效果」
        response_stream = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True
        )

        # 逐字印出
        for chunk in response_stream:
            print(chunk["message"]["content"], end="", flush=True)
        print() # 換行

    except Exception as e:
        print(f"\n❌ 執行過程中發生錯誤：{str(e)}")
