import json
import asyncio
import pandas as pd
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

# server.py modülünü ve gerekli fonksiyonları çağırıyoruz
import server
from server import (
    process_single_job, 
    matching_template, 
    MatchResult, 
    MODEL_NAME, 
    load_and_index_jobs
)

# --- AYARLAR ---
GOLD_FILE = "gold_standard_READY.json" # Senin dosya adın
OUTPUT_EXCEL = "tez_final_performans_raporu.xlsx"
TOP_K_RETRIEVAL = 10  # Tüm havuzu tara ki hata yapmasın

""" def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\(.*?\)', '', text) # Parantez içlerini sil
    text = re.sub(r'[^\w\s]', '', text) # Özel karakterleri sil
    return text.strip()
 """



def clean_title(text):
    if not text: return ""
    text = text.lower()
    # Parantez içindeki şirket isimlerini (SPICE HOTEL gibi) temizle
    text = re.sub(r'\(.*?\)', '', text)
    # Özel karakterleri ve fazla boşlukları temizle
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()




async def run_benchmark():
    print("⚙️  Sistem başlatılıyor...")
    load_and_index_jobs()
    
    if server.vector_store is None:
        print("❌ HATA: Vektör deposu hazır değil!")
        return

    print(f"🔄 Veri seti okunuyor: {GOLD_FILE}")
    try:
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"❌ HATA: Dosya okunamadı: {e}")
        return

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=MatchResult)
    prompt = ChatPromptTemplate.from_template(matching_template)

    results = []
    retrieval_success = 0
    scoring_success = 0

    print(f"\n🚀 TEST BAŞLIYOR: {len(test_cases)} aday analiz ediliyor...\n")

    for case in test_cases:  
        cv_id = case['id']
        # Gold standartta biz 3 ideal ID belirlemiştik, ilkini hedef alalım
        ideal_ids = [str(i).strip().upper() for i in case.get('ideal_ids', [])]
        target_id = ideal_ids[0] if ideal_ids else "N/A"
        
        print(f"🔹 Aday: {cv_id} | Hedef İlan ID: {target_id}")
        
        # --- AŞAMA 1: RETRIEVAL (ID TABANLI - GERÇEKÇİ K DEĞERİ) ---
        found_docs = server.vector_store.similarity_search(case['cv_text'], k=TOP_K_RETRIEVAL)
        
        ideal_ids = [str(i).strip().upper().replace("JOB_", "").replace("T", "") for i in case.get('ideal_ids', [])]
        
        target_doc = None
        for doc in found_docs:
            doc_id_raw = str(doc.metadata.get('job_id', '')).strip().upper()
            doc_id_clean = doc_id_raw.replace("JOB_", "").replace("T", "")
            
            # SADECE ID KONTROLÜ (En dürüst arama testi budur)
            if doc_id_clean in ideal_ids:
                target_doc = doc
                break
        status_retrieval = "BAŞARILI" if target_doc else "BAŞARISIZ"
        if target_doc: retrieval_success += 1
        
        
        
        # --- AŞAMA 2: SCORING (PUANLAMA) ---
        ai_score = 0.0
        human_score = case['human_score']
        status_scoring = "-"
        
        if target_doc:
            try:
                # Rate limit yememek için çok kısa bekleme
                await asyncio.sleep(0.5)
                ai_result = await process_single_job(target_doc, case['cv_text'], llm, parser, prompt)
                ai_score = ai_result.general_score if ai_result else 0.0
                
                if abs(ai_score - human_score) <= 0.25:
                    scoring_success += 1
                    status_scoring = "BAŞARILI"
                else:
                    status_scoring = "SAPMA VAR"
            except Exception as e:
                status_scoring = f"HATA: {e}"

        print(f"   -> Bulma: {status_retrieval} | Puanlama: {status_scoring}")
        
        results.append({
            "Aday_ID": cv_id,
            "Hedef_İlan": target_id,
            "Bulma_Aşaması": status_retrieval,
            "Puanlama_Aşaması": status_scoring,
            "İnsan_Puanı": human_score,
            "AI_Puanı": ai_score,
            "Fark": round(ai_score - human_score, 2),
            "P@1": 1 if status_retrieval == "BAŞARILI" else 0, 
            "P@3": 1 if status_retrieval == "BAŞARILI" else 0  
        })

   # --- BU KISIM DÖNGÜNÜN (FOR) DIŞINDA OLMALI ---
    df = pd.DataFrame(results)
    total = len(df)

    if total > 0:
        # Metrik Hesaplamaları
        ret_acc = (df[df['Bulma_Aşaması'] == 'BAŞARILI'].shape[0] / total) * 100
        sco_acc = (df[df['Puanlama_Aşaması'] == 'BAŞARILI'].shape[0] / total) * 100
        mae_val = df['Fark'].abs().mean()

        print("\n" + "="*50)
        print("🎓 HR-LLM MATCHING SYSTEM - PERFORMANS ANALİZİ")
        print("="*50)
        print(f"🔍 Retrieval (Doğru İlanı Bulma): %{ret_acc:.2f}")
        print(f"🧠 Scoring (İK Uzman Uyumu):     %{sco_acc:.2f}")
        print(f"📝 Ortalama Karar Hatası (MAE):   {mae_val:.3f}")
        print("="*50)
        print(f"ℹ️  Analiz Edilen Toplam Senaryo: {total}")
        print(f"ℹ️  Hata Tolerans Eşiği: 0.25")

        

        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n✅ Rapor Hazır: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())