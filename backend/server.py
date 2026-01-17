import os
import json
import shutil
import re
import asyncio
from typing import List
from tempfile import NamedTemporaryFile

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# Database bağlantılarını hata almamak için try-except içine alıyoruz
try:
    from database import engine, Base, SessionLocal, get_db
    import models
except ImportError:
    print("⚠️ Veritabanı modülleri bulunamadı, sadece JSON kullanılacak.")

from dotenv import load_dotenv
load_dotenv()

# --- AYARLAR ---
JOB_DATA_FILE = "parsed_jobs_FINAL.json"
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Benchmark ve Uygulama Tutarlılığı İçin Ayarlar
SIMILARITY_THRESHOLD = 0.0 
MIN_GENERAL_SCORE = 0.0    
TOP_K_RESULTS = 10          # Vektör aramasından dönecek aday sayısı
MAX_RETURN_RESULTS = 5      # Kullanıcı arayüzünde gösterilecek max sonuç

# --- Modeller ---
class MatchResult(BaseModel):
    # Benchmark'ın doğru çalışması için bu alanların eksiksiz olması şart
    job_id: str = Field(description="İş ilanının ID'si (T1, B2 vb.)")
    job_title: str
    general_score: float
    skill_match: float
    experience_match: float
    report_summary: str

class CVValidationResult(BaseModel):
    is_cv: bool = Field(description="Is this document a CV/Resume?")
    reason: str = Field(description="Explanation")

app = FastAPI(title="HR TalentScout API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None

# --- Veri Yükleme ---
def load_and_index_jobs():
    global vector_store
    docs = []
    
    # 1. Supabase/DB Denemesi
    try:
        db = SessionLocal()
        db_jobs = db.query(models.JobPosting).all()
        if db_jobs:
            for job in db_jobs:
                # BURAYA DİKKAT: job_id mutlaka eklenmeli
                content = f"Title: {job.title}\nDesc: {job.description}\nQuals: {job.requirements}"
                meta = {
                    "job_id": str(job.id), 
                    "job_title": job.title,
                    "description": job.description,
                    "qualifications_raw": job.requirements,
                    "company": job.company,
                    "location": job.location
                }
                docs.append(Document(page_content=content, metadata=meta))
            print(f"✅ Supabase'den {len(docs)} ilan yüklendi.")

    except Exception as e:
        print(f"⚠️ DB Yükleme Hatası (JSON'a geçiliyor): {e}")
    finally:
        db.close() 

    # 2. JSON Fallback
    if not docs and os.path.exists(JOB_DATA_FILE):
        try:
            with open(JOB_DATA_FILE, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            for job in jobs_data:
                content = f"Title: {job.get('job_title', '')}\nDesc: {job.get('description', '')}\nQuals: {job.get('qualifications_raw', '')}"
                docs.append(Document(page_content=content, metadata=job))
            print(f"✅ JSON'dan {len(docs)} ilan yüklendi.")
        except Exception as e:
            print(f"❌ JSON Hata: {e}")

    if docs:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(docs, embeddings)
        print("✅ Vektör Deposu Hazır.")
    else:
        print("⚠️ Hiç ilan bulunamadı.")

"""# --- Veri Yükleme  ---
def load_and_index_jobs():
    global vector_store
    docs = []
    
    # 1. JSON Veri Yükleme (Öncelikli ve Güvenli)
    if os.path.exists(JOB_DATA_FILE):
        try:
            with open(JOB_DATA_FILE, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
            for job in jobs_data:
                # KRİTİK: ID'yi temizleyerek alıyoruz
                jid = str(job.get('job_id') or job.get('id') or 'UNK').strip().upper()
                content = f"Title: {job.get('job_title', '')}\nDesc: {job.get('description', '')}"
                
                meta = {
                    "job_id": jid,
                    "job_title": job.get('job_title', ''),
                    "description": job.get('description', ''),
                    "company": job.get('company', 'Bilinmiyor')
                }
                docs.append(Document(page_content=content, metadata=meta))
            print(f"✅ JSON'dan {len(docs)} ilan yüklendi.")
        except Exception as e:
            print(f"❌ JSON Hata: {e}")

    if docs:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(docs, embeddings)
        print("✅ Vektör Deposu Hazır.")
    else:
        print("⚠️ Hiç ilan bulunamadı. Lütfen JSON dosyasını kontrol edin.")"""


@app.on_event("startup")
async def startup_event():
    load_and_index_jobs()

# --- PROMPTLAR  ---
validation_template = """
You are a Document Classifier. Determine if the text below is a **Personal CV / Resume** or something else.

TEXT CONTENT:
---
{text_sample}
---

RULES:
1. A CV **MUST** describe a PERSON (Education, Experience, Skills, Contact Info).
2. It is **NOT A CV** if it is:
   - A list of criteria or questions (e.g., "Kriter No", "Alt Soru").
   - A project plan, an article, a form, or a book excerpt.
   - A job advertisement itself.
   
OUTPUT JSON:
{{
    "is_cv": boolean,
    "reason": "Short explanation in Turkish"
}}
"""
matching_template = """
You are an expert Senior Technical Recruiter and HR AI.
Your task is to evaluate the relevance of a candidate's CV for a specific Job Posting.

JOB POSTING:
Title: {job_title}
Description: {description}

CANDIDATE CV:
{cv_content}

### EVALUATION GUIDELINES (BE OPTIMISTIC & CONSTRUCTIVE):

1. **PRIORITIZE SKILLS & POTENTIAL:**
   - Look for *transferable skills*. If a Philosophy grad knows Python, that's a WIN.
   - **Internships, Bootcamps, and Personal Projects are REAL experience.** Treat them with respect.
   - If the candidate lacks exact experience but shows strong learning ability, give them a chance (higher score).

2. **SCORING RUBRIC (0.0 to 1.0):**
   - **0.85 - 1.00:** Excellent match.
   - **0.65 - 0.84:** Good match. Has core skills, maybe lacks years.
   - **0.45 - 0.64:** Potential match. Junior or career switcher. Worth considering.
   - **0.00 - 0.44:** Mismatch.

3. **ANALYSIS RULES:**
   - **Career Switchers:** Be generous. If they have the skills, ignore the unrelated degree.
   - **Optimism:** Focus on what the candidate *CAN* do, not just what they can't.

4. **REPORT SUMMARY (CRITICAL):**
   - Write in **TURKISH**, 2-3 sentences.
   - **Tone:** Encouraging, professional, and insightful.
   - **Structure:**
     1. Start with their STRENGTHS (e.g., "Adayın X ve Y konusundaki projeleri etkileyici...").
     2. Mention gaps gently if necessary.
     3. **THE TWIST:** If the score is below 0.85, **END the summary** by suggesting 1 or 2 better-fitting roles based on their skills (e.g., "...mevcut pozisyon yerine 'Junior Developer' veya 'QA Tester' rolleri için daha güçlü bir aday olabilir.").

### OUTPUT FORMAT:
Return a valid JSON object:
- **job_title**: The title from the job posting.
- **general_score**: Float 0.0-1.0.
- **skill_match**: Float 0.0-1.0.
- **experience_match**: Float 0.0-1.0.
- **report_summary**: Turkish summary containing the analysis AND alternative role suggestions if needed.

JSON OUTPUT:
"""

# --- Yardımcı Fonksiyonlar ---
def extract_text_from_upload(file: UploadFile) -> str:
    ext = file.filename.split('.')[-1].lower()
    with NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    text = ""
    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            for page in loader.load(): text += page.page_content + "\n"
        elif ext in ["docx", "doc"]:
            loader = Docx2txtLoader(tmp_path)
            text = "\n".join([d.page_content for d in loader.load()])
        elif ext == "txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
            text = loader.load()[0].page_content
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
    return re.sub(r'\s+', ' ', text).strip()

async def process_single_job(doc, cv_text, llm, parser, prompt_template):
    meta = doc.metadata
    jid = str(meta.get('job_id', 'UNK')).strip().upper()
    try:
        chain = prompt_template | llm | parser
        res = await chain.ainvoke({
            "job_id": jid,
            "job_title": meta.get('job_title', ''),
            "description": meta.get('description', ''),
            "cv_content": cv_text[:4000]
        })
        # Gelen JSON'da job_id eksikse metadata'dan tamamla
        if 'job_id' not in res or not res['job_id']:
            res['job_id'] = jid
        return MatchResult(**res)
    except Exception as e:
        print(f"⚠️ Hata ({jid}): {e}")
        return None

@app.post("/api/match_cv", response_model=List[MatchResult])
async def match_cv(file: UploadFile = File(...)):
    cv_text = extract_text_from_upload(file)
    if len(cv_text) < 20:
        raise HTTPException(status_code=400, detail="CV okunamaz durumda.")

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    
    # 1. Aşama: CV mi?
    v_parser = JsonOutputParser(pydantic_object=CVValidationResult)
    v_prompt = ChatPromptTemplate.from_template(validation_template)
    v_res = await (v_prompt | llm | v_parser).ainvoke({"text_sample": cv_text[:1000]})
    
    if not v_res.get('is_cv'):
        return [MatchResult(job_id="0", job_title="Geçersiz", general_score=0, skill_match=0, experience_match=0, report_summary=v_res.get('reason'))]

    # 2. Aşama: Vektör Arama + LLM Puanlama
    if not vector_store: load_and_index_jobs()
    
    docs = vector_store.similarity_search(cv_text, k=TOP_K_RESULTS)
    m_parser = JsonOutputParser(pydantic_object=MatchResult)
    m_prompt = ChatPromptTemplate.from_template(matching_template)
    
    tasks = [process_single_job(d, cv_text, llm, m_parser, m_prompt) for d in docs]
    results = await asyncio.gather(*tasks)
    
    final = [r for r in results if r and r.general_score >= MIN_GENERAL_SCORE]
    final.sort(key=lambda x: x.general_score, reverse=True)
    
    return final[:MAX_RETURN_RESULTS]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)