import soundfile as sf

from voxcpm.core import VoxCPM
import os
import shutil
import re  # <-- thêm
import random
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
import os
from pathlib import Path
from phonemizer import phonemize
from phonemizer.separator import Separator

import os
from pathlib import Path

import os
from pathlib import Path

def _setup_espeak_ng_for_phonemizer():
    here = Path(__file__).resolve().parent  # ...\deepfake\tts

    # folder đúng theo bạn mô tả
    base = here / "third_party" / "espeak-ng" / "extract" / "eSpeak NG"

    dll = base / "libespeak-ng.dll"
    if not dll.exists():
        raise RuntimeError(f"Không thấy DLL: {dll}")

    # tìm thư mục data
    data_dir = base / "espeak-ng-data"
    if not data_dir.exists():
        # fallback: dò toàn bộ tree
        found = [p for p in (here / "third_party" / "espeak-ng").rglob("espeak-ng-data") if p.is_dir()]
        if not found:
            raise RuntimeError(f"Không tìm thấy espeak-ng-data dưới: {here/'third_party'/'espeak-ng'}")
        data_dir = found[0]

    # ESPEAK_DATA_PATH nên trỏ tới thư mục CHA của espeak-ng-data (hoặc chính nó)
    os.environ["ESPEAK_DATA_PATH"] = str(data_dir.parent)

    # đảm bảo loader tìm được các DLL phụ (nếu có) + chính libespeak-ng.dll
    os.environ["PATH"] = str(dll.parent) + os.pathsep + os.environ.get("PATH", "")

    # cho phonemizer tìm DLL
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(dll)

    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(str(dll))  # theo hướng dẫn chính thức của phonemizer :contentReference[oaicite:1]{index=1}

try:
    _setup_espeak_ng_for_phonemizer()
except Exception as ex:
    print("Lỗi khởi tạo espeak-ng cho phonemizer trên window:", ex)



model = VoxCPM.from_pretrained(
    hf_model_id="kjanh/ViVoxCPM-1.5",
    load_denoiser=False,
    optimize=False,
)

from typing import List, Dict, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments
)
LABEL_LIST = ["O", "B-EN", "I-EN"]
LABEL2ID = {l:i for i,l in enumerate(LABEL_LIST)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

model_name = "meandyou200175/detect_english"
model_detect = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(LABEL_LIST),
    id2label=ID2LABEL, label2id=LABEL2ID
)
tokenizer_detect = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokens_to_pred_spans(offsets: List[Tuple[int,int]], pred_ids: List[int]) -> List[Tuple[int,int]]:
    spans=[]; cur=None
    for (start,end), lid in zip(offsets, pred_ids):
        if start==end: continue
        lab = ID2LABEL.get(lid,"O")
        if lab=="B-EN":
            if cur: spans.append(cur)
            cur=[start,end]
        elif lab=="I-EN":
            if cur: cur[1]=end
            else: cur=[start,end]
        else:
            if cur: spans.append(cur); cur=None
    if cur: spans.append(cur)
    return [tuple(x) for x in spans]
    
def merge_close_spans(spans: List[Dict], max_gap: int = 2) -> List[Dict]:
    if not spans:
        return []
    merged = [spans[0]]
    for cur in spans[1:]:
        prev = merged[-1]
        if cur["start"] - prev["end"] <= max_gap:
            # gộp lại
            prev["end"] = cur["end"]
        else:
            merged.append(cur)
    return merged


def infer_spans(text: str, tokenizer, model, max_length: int = 256) -> List[Dict]:
    text = text.lower()
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items() if k != "offset_mapping"})
        pred_ids = out.logits.argmax(-1)[0].tolist()
    spans = tokens_to_pred_spans(offsets, pred_ids)
    spans = [{"start": s, "end": e} for (s, e) in spans]
    spans = merge_close_spans(spans, max_gap=2)
    # print(spans)
    return spans
import unicodedata

def is_letter(ch: str) -> bool:
    if not ch:
        return False
    # Nếu người dùng lỡ truyền vào tổ hợp có dấu (e + ◌́), chuẩn hoá về NFC:
    ch = unicodedata.normalize("NFC", ch)
    # Chỉ chấp nhận đúng 1 ký tự sau chuẩn hoá
    if len(ch) != 1:
        return False
    # Nhóm 'L*' của Unicode: Lu, Ll, Lt, Lm, Lo
    return unicodedata.category(ch).startswith('L')

# Ví dụ:
tests = [","]
# print({t: is_letter(t) for t in tests})

sep = Separator(phone="", word=" ", syllable="")  # tùy bạn: cách tách phone/word
def to_ipa_espeak(text: str, lang: str) -> str:
    return phonemize(
        text,
        language=lang,          # "en-us", "en-gb", "vi", ...
        backend="espeak", #festival, segments, espeak-mbrola, espeak
        strip=True,
        preserve_punctuation=True,
        with_stress=True,       # EN thường hữu ích
        separator=sep,
        njobs=1,
    ).replace("(en)","").replace("(vi)","")

print(to_ipa_espeak("xin", "en-us"))
print(to_ipa_espeak("xin chào. trào school ", "vi"))

import re
from itertools import chain
from typing import List, Dict, Optional
import logging
from functools import reduce


def flatten(phs):
    """Phẳng hóa list-of-lists (hoặc trả lại list nếu đã phẳng)."""
    if not phs:
        return []
    if isinstance(phs[0], (list, tuple)):
        return list(chain.from_iterable(phs))
    return list(phs)

def g2p_chunk(text: str, lang: str):
    tokens = []
    start = 0
    for t in text:
        if is_letter(t):
            break
        start = start + 1
        
    # Giữ lại: khoảng trắng (\s+), từ (\w+), ký tự khác [^\w\s]
    if start > 0 :
        tokens.extend(flatten(text[0:start]))
    phs = to_ipa_espeak(text[start:], lang)   # có thể trả về list-of-lists
    tokens.extend(flatten(phs))
    return tokens

import re
import logging
from typing import List, Optional, Tuple

TAG_RE = re.compile(r"\[(vi|en(?:-[a-z]{2})?)\]", re.IGNORECASE)

def _norm_lang(tag: str) -> str:
    t = tag.lower()
    if t == "vi":
        return "vi"
    if t == "en":
        return "en-us"          # default cho [en]
    return t                    # en-us / en-gb ...

def _maybe_add_space(tokens_all: List[str], next_chunk: str):
    """Chỉ chèn 1 space khi cần (tránh dính phoneme nếu g2p_chunk strip whitespace)."""
    if not tokens_all:
        return
    last = tokens_all[-1]
    if not last:
        return
    if last[-1].isspace():
        return
    if next_chunk and next_chunk[0].isspace():
        return
    tokens_all.append(" ")

def parse_tagged_segments(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Trả về list (lang, chunk) theo tag.
    - lang=None: đoạn không được gắn tag -> cần auto (infer_spans)
    - lang='vi'/'en-us'...: đoạn đã xác định
    """
    segs: List[Tuple[Optional[str], str]] = []

    cur_lang: Optional[str] = None
    last = 0
    for m in TAG_RE.finditer(text):
        s, e = m.span()
        # text trước tag
        if s > last:
            segs.append((cur_lang, text[last:s]))
        cur_lang = _norm_lang(m.group(1))
        last = e

    # phần còn lại sau tag cuối
    if last < len(text):
        segs.append((cur_lang, text[last:]))

    return segs

def g2p_auto(text: str) -> str:
    """
    Logic cũ: infer_spans để tách các đoạn EN, còn lại coi là VI.
    """
    spans = infer_spans(text, tokenizer_detect, model_detect)
    spans = sorted(spans, key=lambda x: x["start"])

    tokens_all: List[str] = []
    last = 0
    for sp in spans:
        s, e = sp["start"], sp["end"]

        # trước EN -> VI
        if s > last:
            vi_chunk = text[last:s]
            if vi_chunk:
                tokens_all.extend(g2p_chunk(vi_chunk, "vi"))

        # EN
        en_chunk = text[s:e]
        if en_chunk:
            _maybe_add_space(tokens_all, en_chunk)
            tokens_all.extend(g2p_chunk(en_chunk, "en-us"))

        last = e

    # đuôi -> VI
    if last < len(text):
        vi_chunk = text[last:]
        if vi_chunk:
            tokens_all.extend(g2p_chunk(vi_chunk, "vi"))

    return "".join(tokens_all)

def g2p(text: str) -> str:
    """
    - Nếu có tag [vi]/[en]...: dùng tag để chia.
      + đoạn nào lang=None (chưa tag) => g2p_auto trên đoạn đó
    - Nếu không có tag: g2p_auto toàn chuỗi (như cũ)
    """
    try:
        segs = parse_tagged_segments(text)
        has_any_tag = any(lang is not None for lang, _ in segs)

        if not has_any_tag:
            return g2p_auto(text)

        tokens_all: List[str] = []
        for lang, chunk in segs:
            if not chunk:
                continue

            if lang is None:
                # phần chưa xác định -> auto detect EN/VI
                tokens_all.append(g2p_auto(chunk))
            else:
                # phần đã xác định -> phonemize thẳng
                if lang.startswith("en"):
                    _maybe_add_space(tokens_all, chunk)
                tokens_all.extend(g2p_chunk(chunk, lang))

        return "".join(tokens_all)

    except Exception as ex:
        logging.warning(f"Tokenization of mixed texts failed: {ex}")
        return ""
    
def trim_leading_silence_torch(
    wav: torch.Tensor,
    sample_rate: int,
    silence_thresh: float = 0.086,
    chunk_ms: int = 10,
    extend_ms: int = 20,
    ratio: float = 0.95,  # % sample phải dưới ngưỡng để coi là im lặng
):
    wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
    norm_wav = wav_np / (np.max(np.abs(wav_np)) + 1e-8)

    chunk_size = int(sample_rate * chunk_ms / 1000)
    total_chunks = int(len(norm_wav) / chunk_size)

    start_idx = 0
    for i in range(total_chunks):
        chunk = norm_wav[i * chunk_size : (i + 1) * chunk_size]
        # Tính tỷ lệ sample dưới ngưỡng
        silent_ratio = np.mean(np.abs(chunk) < silence_thresh)
        if silent_ratio < ratio:  # nếu ít hơn 95% sample im lặng → coi là có tiếng
            start_idx = max(0, i * chunk_size - int(sample_rate * extend_ms / 1000))
            break

    return wav[:, start_idx:]


import re
import re
from typing import List

def split_text_into_chunks(
    s: str,
    min_chars: int = 100,
    max_chars: int = 500,
    force_period: bool = False,
) -> List[str]:
    """
    Ưu tiên tách theo '.', chỉ khi 1 câu > max_chars mới tách tiếp theo ','.
    Nếu vẫn > max_chars thì xẻ theo từ (không cắt giữa từ).

    force_period=False:
        - các chunk do tách trong cùng 1 câu sẽ kết thúc bằng ',' (chunk cuối của câu kết thúc '.')
    force_period=True:
        - mọi chunk đều kết thúc bằng '.'
    """
    s = re.sub(r"\s+", " ", (s or "").strip())
    if not s:
        return []

    # bỏ dấu '.' cuối để tránh sinh câu rỗng khi split
    if s.endswith("."):
        s = s[:-1].rstrip()

    sentences = [seg.strip() for seg in re.split(r"\s*\.\s*", s) if seg.strip()]

    def split_by_words(text: str) -> List[str]:
        words = text.split()
        out, buf = [], []
        cur_len = 0
        for w in words:
            add_len = len(w) if cur_len == 0 else len(w) + 1
            if cur_len + add_len <= max_chars:
                buf.append(w)
                cur_len += add_len
            else:
                if buf:
                    out.append(" ".join(buf))
                buf = [w]
                cur_len = len(w)
        if buf:
            out.append(" ".join(buf))
        return out

    def pack_units(units: List[str], joiner: str = ", ") -> List[str]:
        chunks = []
        cur = ""
        for u in units:
            cand = u if not cur else f"{cur}{joiner}{u}"
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    chunks.append(cur)
                    cur = u
                else:
                    chunks.append(u)
                    cur = ""
        if cur:
            chunks.append(cur)
        return chunks

    def enforce_min(chunks: List[str], joiner: str = ", ") -> List[str]:
        i = 0
        while i < len(chunks) and len(chunks) > 1:
            if len(chunks[i]) >= min_chars:
                i += 1
                continue

            # thử gộp với chunk sau
            if i + 1 < len(chunks):
                cand = chunks[i] + joiner + chunks[i + 1]
                if len(cand) <= max_chars:
                    chunks[i] = cand
                    del chunks[i + 1]
                    continue

            # thử gộp với chunk trước
            if i - 1 >= 0:
                cand = chunks[i - 1] + joiner + chunks[i]
                if len(cand) <= max_chars:
                    chunks[i - 1] = cand
                    del chunks[i]
                    i -= 1
                    continue

            i += 1
        return chunks

    out: List[str] = []

    for sent in sentences:
        # 1) Ưu tiên: nếu câu không quá dài thì giữ nguyên
        if len(sent) <= max_chars:
            sent_chunks = [sent]
        else:
            # 2) Nếu câu quá dài: split theo dấu ','
            clauses = [c.strip() for c in re.split(r"\s*,\s*", sent) if c.strip()]

            # 2.1) mệnh đề nào vẫn quá dài thì xẻ theo từ
            units: List[str] = []
            for c in clauses:
                units.extend(split_by_words(c) if len(c) > max_chars else [c])

            # 2.2) đóng gói lại để mỗi chunk <= max_chars (ưu tiên biên ',')
            sent_chunks = pack_units(units, joiner=", ")
            sent_chunks = enforce_min(sent_chunks, joiner=", ")

        # gắn dấu câu cho từng chunk của câu này
        for j, ch in enumerate(sent_chunks):
            if force_period:
                out.append(ch.rstrip(",.") + ".")
            else:
                out.append(ch.rstrip(",.") + ("," if j < len(sent_chunks) - 1 else "."))

    # nếu vẫn có chunk quá ngắn, gộp với hàng xóm (giữ dấu câu sẵn có)
    i = 0
    while i < len(out) and len(out) > 1:
        if len(out[i]) >= min_chars:
            i += 1
            continue

        if i + 1 < len(out):
            cand = (out[i].rstrip() + " " + out[i + 1].lstrip()).strip()
            if len(cand) <= max_chars:
                out[i] = cand
                del out[i + 1]
                continue

        if i - 1 >= 0:
            cand = (out[i - 1].rstrip() + " " + out[i].lstrip()).strip()
            if len(cand) <= max_chars:
                out[i - 1] = cand
                del out[i]
                i -= 1
                continue

        i += 1

    return out
def trim_internal_silence_segment_torch(
    wav: torch.Tensor,
    sample_rate: int,
    silence_thresh: float = 0.05,
    max_silence_ms: int = 350,
):
    """
    Cắt các đoạn silence liên tục ở GIỮA audio sao cho
    mỗi đoạn silence không dài quá max_silence_ms.
    """
    if wav.dim() == 2:
        wav = wav.squeeze(0)

    wav_np = wav.cpu().numpy().astype(np.float32)

    # Normalize để detect silence ổn định
    max_amp = np.max(np.abs(wav_np)) + 1e-8
    norm = wav_np / max_amp

    max_silence = int(sample_rate * max_silence_ms / 1000)

    segments = []
    i = 0
    T = len(norm)

    while i < T:
        if abs(norm[i]) < silence_thresh:
            # bắt đầu silence segment
            start = i
            while i < T and abs(norm[i]) < silence_thresh:
                i += 1
            end = i
            length = end - start

            if length <= max_silence:
                # giữ nguyên
                segments.append(wav_np[start:end])
            else:
                # giữ phần ĐẦU và CUỐI silence, cắt giữa
                keep = max_silence // 2
                head = wav_np[start : start + keep]
                tail = wav_np[end - (max_silence - keep) : end]
                segments.append(np.concatenate([head, tail]))
        else:
            # voice segment
            start = i
            while i < T and abs(norm[i]) >= silence_thresh:
                i += 1
            segments.append(wav_np[start:i])

    out = np.concatenate(segments) if segments else np.zeros(0, np.float32)
    return torch.from_numpy(out).unsqueeze(0)

import re
import unicodedata

import re
import unicodedata
def number_to_vietnamese(n: int) -> str:
    units = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    tens = ["", "mười", "hai mươi", "ba mươi", "bốn mươi",
            "năm mươi", "sáu mươi", "bảy mươi", "tám mươi", "chín mươi"]

    if n < 10:
        return units[n]

    if n < 20:
        if n == 10:
            return "mười"
        return "mười " + units[n % 10]

    if n < 100:
        t = n // 10
        u = n % 10
        if u == 0:
            return tens[t]
        if u == 1:
            return tens[t] + " mốt"
        if u == 5:
            return tens[t] + " lăm"
        return tens[t] + " " + units[u]

    # fallback: đọc từng chữ số
    return " ".join(units[int(d)] for d in str(n))

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Chuẩn hóa unicode (rất quan trọng cho tiếng Việt)
    text = unicodedata.normalize("NFC", text)
    def replace_number(match):
        num = int(match.group())
        return number_to_vietnamese(num)

    text = re.sub(r"\d+", replace_number, text)
    # Giữ lại chữ cái, số và khoảng trắng
    text = text.replace("AI", "ây ai")
    text = text.replace("\n\n", ".")
    text = text.replace("\n", ".")
    text = text.replace(".",". ")
    text = text.replace(",",", ").replace("/",", ").replace("\\",", ")
    text = text.replace("\n"," ").replace("  "," ").replace('“',"").replace('”',"").replace('"','').replace("-",", ").replace("!",".").replace("?",".").replace(":",",").replace("–",",").replace(". ,", ". ").replace(". .",". ").replace(", ,",", ").replace("  "," ")

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("..",".")

    return text


def text_to_speech(
    texts,
    prompt_wav_path=None,
    prompt_text=None,
    cfg_value=2.0,
    inference_timesteps=10,
    out_path="col.wav",
    silence_ms=170,
):
    texts = normalize_text(texts)
    sr = int(model.tts_model.sample_rate)
    prompt_text = g2p(prompt_text)
    prompt_text = prompt_text

    silence = np.zeros(int(sr * silence_ms / 1000.0), dtype=np.float32)
    wavs = []

    for i, chunk in enumerate(split_text_into_chunks(texts)):
        text = g2p(chunk)
        text = text.replace(".",",")
        text = text[:-1] + ' "."'
        
        # print(text)
        with torch.inference_mode():
            if prompt_wav_path is not None:
                audio = model.generate(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=float(cfg_value),
                    inference_timesteps=int(inference_timesteps),
                    max_len=600,
                    denoise=False,
                )
            else:
                audio = model.generate(
                    text=text,
                    cfg_value=float(cfg_value),
                    inference_timesteps=int(inference_timesteps),
                    max_len=600,
                    denoise=False,
                )

        # audio: (T,) hoặc (1, T)
        audio_t = torch.as_tensor(audio).float()
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)

        # ⚠️ Trim silence đầu CHỈ cho các chunk sau
        if i > 0:
            audio_t = trim_leading_silence_torch(
                audio_t,
                sample_rate=sr,
                silence_thresh=0.086,
                chunk_ms=10,
                extend_ms=20,
                ratio=0.95,
            )

        audio_np = audio_t.squeeze(0).cpu().numpy().astype(np.float32)

        if i > 0:
            wavs.append(silence)   # khoảng lặng giữa các đoạn

        wavs.append(audio_np)

    final = np.concatenate(wavs) if wavs else np.zeros(0, np.float32)
    
    final_t = torch.from_numpy(final).unsqueeze(0)
    
    final_t = trim_internal_silence_segment_torch(
        final_t,
        sample_rate=sr,
        silence_thresh=0.05,
        max_silence_ms=350,
    )
    
    final = final_t.squeeze(0).cpu().numpy()
    
    sf.write(out_path, final, sr)
    return out_path
