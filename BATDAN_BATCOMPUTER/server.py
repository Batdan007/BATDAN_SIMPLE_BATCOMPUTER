import base64
import io
import json
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .simple_batcomputer import SimpleBatComputer


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_PATH = DATA_DIR / "prompts.json"


def load_prompts() -> List[dict]:
    if PROMPTS_PATH.exists():
        try:
            return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_prompts(prompts: List[dict]):
    PROMPTS_PATH.write_text(json.dumps(prompts, indent=2), encoding="utf-8")


def db_path(name: str) -> Path:
    safe = "".join([c for c in name if c.isalnum() or c in ("_", "-")])
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid database name")
    return DATA_DIR / f"{safe}.db"


app = FastAPI(title="BATCOMPUTER Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bat = SimpleBatComputer()
bat.setup_dolphin_mistral()


class ChatRequest(BaseModel):
    message: str
    include_vision: Optional[bool] = False
    use_reasoning: Optional[bool] = True
    show_reasoning: Optional[bool] = True
    reasoning_preset: Optional[str] = "default"
    image_base64: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    image_bytes = None
    if req.image_base64:
        try:
            image_bytes = base64.b64decode(req.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image_base64")

    text = await bat.chat(
        req.message,
        include_vision=req.include_vision,
        include_voice=False,
        use_reasoning=req.use_reasoning,
        show_reasoning=req.show_reasoning,
        reasoning_preset=req.reasoning_preset,
        image_bytes=image_bytes,
    )
    return {"response": text}


# Prompt templates CRUD
@app.get("/prompts")
def list_prompts():
    return load_prompts()


class PromptTemplate(BaseModel):
    id: Optional[str] = None
    name: str
    content: str


@app.post("/prompts")
def create_prompt(tpl: PromptTemplate):
    prompts = load_prompts()
    tpl_dict = tpl.dict()
    if not tpl_dict.get("id"):
        tpl_dict["id"] = f"tpl_{len(prompts)+1}"
    prompts.append(tpl_dict)
    save_prompts(prompts)
    return tpl_dict


@app.put("/prompts/{tpl_id}")
def update_prompt(tpl_id: str, tpl: PromptTemplate):
    prompts = load_prompts()
    for i, p in enumerate(prompts):
        if p.get("id") == tpl_id:
            prompts[i] = {"id": tpl_id, "name": tpl.name, "content": tpl.content}
            save_prompts(prompts)
            return prompts[i]
    raise HTTPException(status_code=404, detail="Template not found")


@app.delete("/prompts/{tpl_id}")
def delete_prompt(tpl_id: str):
    prompts = load_prompts()
    new_prompts = [p for p in prompts if p.get("id") != tpl_id]
    save_prompts(new_prompts)
    return {"deleted": tpl_id}


# Databases
@app.get("/dbs")
def list_dbs():
    return [p.stem for p in DATA_DIR.glob("*.db")]


class CreateDbRequest(BaseModel):
    name: str


@app.post("/dbs")
def create_db(body: CreateDbRequest):
    path = db_path(body.name)
    if path.exists():
        raise HTTPException(status_code=400, detail="Database already exists")
    conn = sqlite3.connect(path)
    conn.close()
    return {"created": body.name}


@app.delete("/dbs/{name}")
def delete_db(name: str):
    path = db_path(name)
    if path.exists():
        path.unlink()
        return {"deleted": name}
    raise HTTPException(status_code=404, detail="Database not found")


@app.get("/dbs/{name}/tables")
def list_tables(name: str):
    path = db_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        return {"tables": tables}
    finally:
        conn.close()


class ExecuteSQLRequest(BaseModel):
    sql: str
    params: Optional[List] = None


@app.post("/dbs/{name}/execute")
def execute_sql(name: str, body: ExecuteSQLRequest):
    path = db_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(body.sql, body.params or [])
        rows = cur.fetchall()
        if rows and isinstance(rows[0], sqlite3.Row):
            cols = rows[0].keys()
            data = [dict(r) for r in rows]
            return {"columns": list(cols), "rows": data[:500]}
        else:
            conn.commit()
            return {"columns": [], "rows": [], "status": "ok"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@app.post("/dbs/{name}/ingest/csv")
async def ingest_csv(name: str, file: UploadFile = File(...), table: Optional[str] = None):
    import csv
    path = db_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="Empty CSV")
    headers = rows[0]
    table_name = table or (file.filename.split(".")[0] if file.filename else "ingest")
    conn = sqlite3.connect(path)
    try:
        cols_sql = ", ".join([f'"{h}" TEXT' for h in headers])
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({cols_sql})')
        placeholders = ", ".join(["?"] * len(headers))
        conn.executemany(
            f'INSERT INTO "{table_name}" ({", ".join([f"\"{h}\"" for h in headers])}) VALUES ({placeholders})',
            rows[1:]
        )
        conn.commit()
        return {"table": table_name, "inserted": max(0, len(rows) - 1)}
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

