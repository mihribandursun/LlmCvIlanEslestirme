import json
import asyncio
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import server 
from server import process_single_job, matching_template, MatchResult, MODEL_NAME

GOLD_FILE = "gold_standard_READY.json" 
OUTPUT_EXCEL = "gercek_sistem_raporu.xlsx"

async def run_system_test():
    if server.vector_store is None:
        print("🔄 Vektör deposu yükleniyor...")
        server.load_and_index_jobs()

    try:
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"❌ HATA: {GOLD_FILE} okunamadı: {e}")
        return

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=MatchResult)
    prompt = ChatPromptTemplate.from_template(matching_template)

    results = []
    print(f"🚀 KOMPLE SİSTEM TESTİ BAŞLIYOR ({len(test_cases)} aday)...\n")

    for case in test_cases:
        cv_id = case['id']
        current_cv_text = case.get('cv_text', "")
        ideal_ids = [str(x).strip().upper() for x in case.get('ideal_ids', [])]

        print(f"   🔹 Aday: {cv_id} için 52 ilan analiz ediliyor...")

        # Vektör Araması (k=52 yaparak hepsini LLM'e sokuyoruz)
        retrieved_docs = server.vector_store.similarity_search(current_cv_text, k=52)
        
        valid_results = []
        for doc in retrieved_docs:
            res = await process_single_job(doc, current_cv_text, llm, parser, prompt)
            if res:
                valid_results.append(res)
            # RATE LIMIT ENGELLEMEK İÇİN: Her ilandan sonra çok kısa bekle
            time.sleep(0.05) 

        # Puanlara göre sırala
        valid_results.sort(key=lambda x: x.general_score, reverse=True)

        # En iyi sonuçları kontrol et
        top_1_result = valid_results[0] if valid_results else None
        top_3_ids = [str(res.job_id).strip().upper() for res in valid_results[:3]]

        # Metrikler (Garantici Karşılaştırma)
        p1_success = 0
        p3_success = 0
        
        if top_1_result:
            p1_val = str(top_1_result.job_id).strip().upper()
            if p1_val == ideal_ids[0]:
                p1_success = 1
        
        if any(jid in top_3_ids for jid in ideal_ids):
            p3_success = 1

        results.append({
            "ID": cv_id,
            "Beklenen ID": ideal_ids[0] if ideal_ids else "N/A",
            "Sistemin Bulduğu ID": str(top_1_result.job_id) if top_1_result else "BULUNAMADI",
            "Top 3 Liste": ", ".join(top_3_ids),
            "P@1": p1_success,
            "P@3": p3_success,
            "AI Puan": top_1_result.general_score if top_1_result else 0
        })
        
        # Bir aday bittikten sonra biraz daha bekle (OpenAI'ı yormamak için)
        print(f"      ✅ Aday tamamlandı. Bekleniyor...")
        await asyncio.sleep(1)

    # Raporlama
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 GERÇEK SİSTEM PERFORMANSI (END-TO-END)")
    print("="*50)
    print(f"🎯 Precision@1: %{df['P@1'].mean() * 100:.1f}")
    print(f"🎯 Precision@3: %{df['P@3'].mean() * 100:.1f}")
    print("="*50)
    
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✅ Rapor kaydedildi: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    asyncio.run(run_system_test())