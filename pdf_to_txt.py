import os
import pdfplumber

def convert_with_pdfplumber(pdf_path, output_txt_path):
    print(f"⏳ 開始使用 pdfplumber 轉換大檔案：{pdf_path}")
    
    try:
        # 開啟 PDF
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📄 該 PDF 共有 {total_pages} 頁，正在提取文字...")
            
            with open(output_txt_path, "w", encoding="utf-8") as f:
                for idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        f.write(f"\n--- 第 {idx+1} 頁 ---\n")
                        f.write(text + "\n")
                    
                    if (idx + 1) % 10 == 0 or (idx + 1) == total_pages:
                        print(f" ➔ 已完成 {idx + 1} / {total_pages} 頁...")
                        
        print(f"✅ 轉換成功！請檢查文字檔：{output_txt_path}\n")
        
    except Exception as e:
        print(f"❌ 轉換失敗，錯誤原因：{str(e)}")

target_pdf = "data/上冊四校.pdf"  
output_txt = "knowledge/上冊四校_純文字版.txt"

if os.path.exists(target_pdf):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    convert_with_pdfplumber(target_pdf, output_txt)
else:
    print(f"❌ 找不到檔案 {target_pdf}")

# ===== 設定下冊的檔案路徑 =====
target_pdf = "data/下冊四校.pdf"  # 確保你的 data 資料夾裡有這個檔案
output_txt = "knowledge/下冊四校_純文字版.txt"

if os.path.exists(target_pdf):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    convert_with_pdfplumber(target_pdf, output_txt)
else:
    print(f"❌ 找不到檔案 {target_pdf}")
