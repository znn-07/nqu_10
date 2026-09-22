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

# ===== 2. 讀取 Prompt 模板 =====
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
        if result and result["documents"] and result["documents"][0]:
            context = "\n\n".join(result["documents"][0])
        else:
            context = "（未檢索到相關文獻）"

        # 3-3. 建立終極提示詞（修正：字串靠左對齊，消除多餘空白）
        prompt = f"""你是一位極度嚴格、絕不說廢話的金門大學校史導覽員。

請你【嚴格且唯一】根據下方的參考文獻內容回答問題。如果答案在文獻中，請直接吐出答案。

【參考文獻內容】
{context}

【使用者提問】
{question}

【回答規則】
1. 請直接回答答案本身，不准做任何自我介紹、不准說歡迎詞（如：你好、很高興為您服務、我是AI助手等廢話一律不准說）。
2. 如果文獻中找不到答案，請直接回答：「資料庫中沒有相關資訊」，絕對不准自己從別的地方湊字來編造。

請直接給出答案："""

        # 修正：改用換行 \n 印出，不再用 \r 覆蓋，避免畫面殘留「思考中...」的髒字
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
