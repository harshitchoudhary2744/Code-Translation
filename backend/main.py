import time, threading
from pathlib import Path

import pandas as pd
import torch
import tree_sitter_c_sharp as tscs
from tree_sitter import Language, Parser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (RobertaTokenizer, T5ForConditionalGeneration,
                          PLBartTokenizer, PLBartForConditionalGeneration)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MAX_LEN = 256

MODELS = {
    "codet5-small": {"folder": "codet5-small-java-cs", "kind": "codet5"},
    "codet5-base":  {"folder": "codet5-base-java-cs",  "kind": "codet5"},
    "plbart-base":  {"folder": "plbart-base-java-cs",  "kind": "plbart"},
}

# ---------- syntax checker (same idea as Step 12) ----------
parser = Parser(Language(tscs.language()))
def is_valid_cs(code: str) -> bool:
    tree = parser.parse(bytes("class A { " + code + " }", "utf8"))
    return not tree.root_node.has_error

# ---------- model loading (lazy, cached) ----------
_loaded, _lock = {}, threading.Lock()

def available():
    return [n for n, m in MODELS.items() if (MODEL_DIR / m["folder"] / "config.json").exists()]

def load(name):
    with _lock:
        if name in _loaded:
            return _loaded[name]
        meta = MODELS[name]
        path = str(MODEL_DIR / meta["folder"])
        if meta["kind"] == "codet5":
            tok = RobertaTokenizer.from_pretrained(path)
            model = T5ForConditionalGeneration.from_pretrained(path)
            start_id = None
        else:
            tok = PLBartTokenizer.from_pretrained(path, src_lang="java", tgt_lang="java")
            model = PLBartForConditionalGeneration.from_pretrained(path)
            start_id = tok.lang_code_to_id["__java__"]
        model.eval()
        _loaded[name] = (tok, model, start_id)
        return _loaded[name]

def generate(name, code, beams, n_return, max_len):
    tok, model, start_id = load(name)
    enc = tok(code, return_tensors="pt", max_length=MAX_LEN, truncation=True)
    enc.pop("token_type_ids", None)
    kwargs = dict(max_length=max_len, num_beams=beams, num_return_sequences=n_return)
    if start_id is not None:
        kwargs["decoder_start_token_id"] = start_id
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    return tok.batch_decode(out, skip_special_tokens=True)

def translate_one(name, code, retry):
    t0 = time.time()
    first = generate(name, code, 5, 1, MAX_LEN)[0]
    first_valid = is_valid_cs(first)
    final, retried, fixed = first, False, False
    if retry and not first_valid:
        retried = True
        for cand in generate(name, code, 10, 10, 512):
            if is_valid_cs(cand):
                final, fixed = cand, True
                break
    return {
        "model": name,
        "translation": final,
        "first_attempt": first,
        "first_attempt_valid": first_valid,
        "retried": retried,
        "fixed_by_retry": fixed,
        "final_valid": is_valid_cs(final),
        "seconds": round(time.time() - t0, 2),
    }

# ---------- API ----------
app = FastAPI(title="Java to C# Code Translator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    code: str
    model: str = "codet5-base"
    retry: bool = True

def check_code(code):
    if not code.strip():
        raise HTTPException(400, "Please enter some Java code.")
    if len(code) > 5000:
        raise HTTPException(400, "Code too long (max 5000 characters). Try one method at a time.")

@app.get("/models")
def list_models():
    return {"available": available()}

@app.post("/translate")
def translate(req: TranslateRequest):
    check_code(req.code)
    if req.model not in available():
        raise HTTPException(404, f"Model '{req.model}' not found. Available: {available()}")
    return translate_one(req.model, req.code, req.retry)

@app.post("/compare")
def compare(req: TranslateRequest):
    check_code(req.code)
    return {"results": [translate_one(n, req.code, req.retry) for n in available()]}

def read_csv(fname):
    p = ROOT / fname
    return pd.read_csv(p) if p.exists() else None

@app.get("/results")
def results():
    base = [d for d in (read_csv("results.csv"), read_csv("results_plbart.csv")) if d is not None]
    if not base:
        return []
    df = pd.concat(base).drop_duplicates("model", keep="last")
    cb = [d for d in (read_csv("results_codebleu.csv"), read_csv("results_codebleu_plbart.csv")) if d is not None]
    if cb:
        df = df.merge(pd.concat(cb).drop_duplicates("model", keep="last"), on="model", how="left")
    val = read_csv("validity.csv")
    if val is not None:
        df = df.merge(val, on="model", how="left")
    return df.astype(object).where(df.notna(), None).to_dict("records")