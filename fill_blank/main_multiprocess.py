#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_datasets.py  (modified)
遍历多数据集并调用大模型生成 QA 输出，强制模型以 JSON-only 结构返回（并具备修复重试机制）。
保持 OpenAI 调用方式不变（client.chat.completions.create）。

附加功能：可开启/关闭 "跳过已存在样本但计入目标数量" 的功能。
配置项：SKIP_EXISTING_SAMPLES = True / False
当开启时，运行时会在每个 dataset/job 的输出文件中检测已存在的成功生成样本（可解析为 JSON 且不包含 "error" 字段），
这些样本将被视为已计入 MAX_SAMPLES_PER_DATASET。已存在的样本会被跳过不重复生成，输出文件会以追加模式打开以保留已有内容。

注意：主体逻辑和大部分函数未被结构性改变，仅在各 processing loop 中加入了跳过已有样本的检查与计数逻辑。
"""

import base64
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
import os
import concurrent.futures
import threading
import json

# ====== 如果你使用官方 openai 包 ======
from openai import OpenAI

from bi_dataset_prompt import *
# ====== 你自己的模块引用（按需修改/保留） ======
from common_part import get_mask_palette_and_prior
from dsm_dataset_prompt import *
from one_dataset_prompt import *
from tri_dataset_prompt import *

# =======================================================================
# === Configuration (请根据需要修改) ====================================
# =======================================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_API_BASE", "")
)

MODEL = "gpt-5"
TEMPERATURE = 0.0
MAX_TOKENS = 16384
MAX_RETRIES = 2  # 当返回不符合 schema 时的修复重试次数

# 数据集根路径列表（按需调整）
DATASET_ROOTS = [
    # "../b2m_dataset/FAST",
    # "../b2m_dataset/SIOR",
    # "../b2m_dataset/SOTA",
    # "../sem_dataset/GID",
    # "../sem_dataset/loveDA",
    #
    ###################  images  masks  dsms  masks_txt
    # "../sem_dataset/Potsdam",
    # "../sem_dataset/Vaihingen",

    ##################  im1 im2 label1 label2  label1_txt  label2_txt
    # "../bi-change/SECOND",
    # "../bi-change/xView2",

    ##################  im1 im2 im3 label1 label2 label3
    # "../multi-change/miniucd",
    "../multi-change/spaceNet7"
]

JOBS = [ "state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]
# "st_reason", "st_counterfactual", "st_evolution", "st_consistency", "reason", "counterfactual", "influ", "sem", "plan", "estimate",

MAX_SAMPLES_PER_DATASET = 800

RESULT_OUTDIR = Path("./gpt_results1")
RESULT_OUTDIR.mkdir(parents=True, exist_ok=True)

COMMON_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

MAX_WORKERS = int(os.getenv("RUN_ALL_MAX_WORKERS", "8"))

# ----------------- 新增配置：是否启用跳过已存在样本（但计入目标数量） -----------------
# True: 如果在输出文件中检测到已存在的成功样本（JSON 且无 error 字段），则跳过重复生成并把它们算在 MAX_SAMPLES_PER_DATASET 中。
# False: 行为与原先一致（每次 job 会重新以写模式覆盖输出文件）。
SKIP_EXISTING_SAMPLES = True

# 全局临时 holder：在每个 job 开始前会被设置为该 job/数据集的已存在样本 id 集合（sample_name 列表）
EXISTING_SAMPLE_IDS = set()
# =======================================================================
# === JSON generation enforcement (SYSTEM / USER templates) ==============
# =======================================================================
SYSTEM_MSG = (
    "You are a strict JSON-only question generator specialized in image-grounded remote-sensing QA.\n"
    "You MUST RETURN EXACTLY ONE JSON OBJECT and NOTHING ELSE (no explanations, no backticks).\n"
    "If you cannot comply, return {\"error\":\"<short reason>\"}.\n"
    "DO NOT include the tokens 'mask', 'DSM', 'im1', 'im2', 'im3', or 'segmentation' in the 'question' field.\n"
    "Output must be valid JSON parsable by standard JSON parsers.\n"
)

USER_JSON_TEMPLATE = """
You are given (generation-only):
- dataset: {dataset}
- visibility_stage: {visibility_stage}    # e.g., "human_only" or "with_hidden_info"
- mask_palette (authoritative): {mask_palette}
- prior: {prior}
- visible_images: {visible_images}   # list of filenames humans will see
- hidden_images: {hidden_images}     # list of filenames provided to you only

Also you are provided with a task instruction (task_prompt) that defines the reasoning goal:
{task_prompt}

Task:
Generate exactly one JSON object matching the schema below. The JSON will be consumed automatically; DO NOT output ANY extra fields or free text outside the object.

Schema (fields required):
{{
  "id": "<string - sample id or filename>",
  "design_ideas": "<string - detailed step-by-step design rationale including: 1) image analysis from coarse to fine scale, 2) key observable features identification, 3) reasoning chain construction, 4) blank positioning strategy, 5) answer determination logic>",
  "question": "<string - fill-in-the-blank question with [BLANK1], [BLANK2], [BLANK3] markers using probabilistic language; MUST NOT contain forbidden tokens>",
  "qa_type": "fill_blank",
  "blanks": {{"Blank1":"<exact answer for BLANK1>","Blank2":"<exact answer for BLANK2 or empty>","Blank3":"<exact answer for BLANK3 or empty>"}},
  "num_blanks": <integer 1-3>,
  "declared_answer_text": "<combined answer format: BLANK1: ans1; BLANK2: ans2; etc.>",
  "thinking": "<string - detailed structured solving rationale for each blank: BLANK1: visual evidence → logical reasoning → probable conclusion, structured as summary -> breakdown -> summary>",
  "meta": { "dataset":"{dataset}", "visibility_stage":"{visibility_stage}",
             "images_used": {visible_images_path}, "masks_used": {hidden_images_path} }
}}

Hard constraints:
- qa_type is fixed as 'fill_blank'
- question must contain 1-3 blanks marked as [BLANK1], [BLANK2], [BLANK3] (consecutive numbering starting at 1)
- num_blanks must be an integer between 1 and 3, matching the actual number of blanks in the question
- blanks object must have keys 'Blank1', 'Blank2', 'Blank3' - fill used blanks with exact answers, leave unused blanks empty ("")
- each blank answer must be ≤ 6 words, canonical and unambiguous
- question must NOT contain forbidden tokens (mask/DSM/im1/im2/im3/segmentation)
- do NOT include extra top-level keys outside schema (if you must indicate error, return: {{"error":"<reason>"}})
- The thinking processes in the "thinking" and "design_ideas" fields, as well as the thinking and logic behind the question design process, must be prohibited from appearing in the "question" fields. 

Detailed requirements for design_ideas field:
You must provide a comprehensive step-by-step design rationale covering:
1. **Scene Analysis**: Describe the overall scene composition, spatial patterns, and key observable characteristics at multiple scales
2. **Feature Identification**: Identify specific visual features, structures, patterns, or spatial relationships that are relevant and observable in RGB
3. **Reasoning Chain**: Explain the logical reasoning path from visual observation to the information needed for each blank
4. **Blank Positioning Strategy**: Describe why specific information gaps were chosen as blanks and how they test visual reasoning skills
5. **Answer Determination Logic**: Explain the evidence-based reasoning for determining the most appropriate answer for each blank

Question formulation guidelines:
- Use probabilistic and qualified language in context: "most likely", "primarily", "typically", "generally", "predominantly", "appears to", "suggests"
- Frame the question context with uncertainty acknowledgment while keeping blanks answerable
- Example phrasings:
  * "The predominant land cover type in this area appears to be [BLANK1], which typically indicates [BLANK2]"
  * "Based on the spatial pattern, this development most likely represents [BLANK1] with approximately [BLANK2] density"
  * "The observed features generally suggest [BLANK1] activity, primarily concentrated in the [BLANK2] portion of the image"

Blank design requirements:
- Each blank should test different aspects of visual reasoning (e.g., identification, quantification, spatial relationship)
- Blanks must be answerable from visible RGB image(s) alone without hidden information  
- Answers should be specific enough to be verifiable but acknowledge the probabilistic nature of remote sensing interpretation
- Prefer blanks that require multi-step reasoning over simple pixel identification
- Each blank answer must be the most probable/likely response based on observable evidence

Answer quality standards:
- Each answer must be ≤ 6 words, using standard terminology
- Answers should be canonical (e.g., "residential buildings" not "houses", "agricultural land" not "farms")
- When quantification is required, use ranges or qualified terms (e.g., "high density", "approximately 50%", "moderate coverage")
- Avoid absolute statements in answers when uncertain (e.g., "likely urban" instead of "urban")

Thinking section requirements:
- Structure as: summary (overall assessment) -> breakdown (detailed evidence analysis for each blank) -> summary (conclusion synthesis)
- For each blank, provide: "BLANK[X]: visual evidence → logical reasoning → probable conclusion"
- Reference specific visual regions, patterns, textures, or spatial relationships
- Explain how the evidence supports the chosen answer as the most likely option
- Acknowledge any ambiguities or alternative interpretations where relevant

declared_answer_text format:
- Follow exact format: "BLANK1: <ans1>; BLANK2: <ans2>; BLANK3: <ans3>"
- Omit unused blanks from the format
- Use the exact same wording as in the blanks object

If your first output is not valid JSON or fails constraints, you will be asked (programmatically) to return the corrected JSON object only (no commentary)."""

# =======================================================================
# === Utilities / helpers ================================================
# =======================================================================
def file_to_data_url(path: Path):
    return base64.b64encode(path.read_bytes()).decode()

def extract_text_from_response(resp) -> str:
    """
    Robust extraction of textual content from various client response shapes.
    """
    try:
        if resp is None:
            return ""
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if not choices:
                return ""
            choice = choices[0]
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict):
                    return msg.get("content", "") or ""
                if isinstance(msg, str):
                    return msg
                if "text" in choice:
                    return choice.get("text", "") or ""
                return str(choice)
            else:
                return str(choice)
        else:
            # object-like (new client)
            if hasattr(resp, "choices"):
                chs = getattr(resp, "choices")
                if chs:
                    choice = chs[0]
                    # try nested message.content
                    if hasattr(choice, "message"):
                        msg = getattr(choice, "message")
                        if isinstance(msg, dict):
                            return msg.get("content", "") or ""
                        if hasattr(msg, "content"):
                            return getattr(msg, "content") or ""
                    if hasattr(choice, "text"):
                        return getattr(choice, "text") or ""
                    if hasattr(choice, "content"):
                        return getattr(choice, "content") or ""
                    return str(choice)
            return str(resp)
    except Exception as e:
        print(f"[WARN] extract_text_from_response failed: {e}")
        return str(resp)

def extract_json_text(raw_text: str):
    """
    Try to find the first JSON object in raw_text:
    - strip possible ``` fencing
    - locate first '{' and matching final '}' (greedy last '}')
    Return the substring or None.
    """
    if not raw_text:
        return None
    txt = raw_text.strip()
    # remove triple backticks fences and leading language hints
    txt = re.sub(r"^```(?:json|\w+)?\s*", "", txt, flags=re.I)
    txt = re.sub(r"\s*```$", "", txt, flags=re.I)
    # find first { and last }
    first = txt.find("{")
    last = txt.rfind("}")
    if first == -1 or last == -1 or last < first:
        return None
    candidate = txt[first:last+1]
    return candidate


def validate_schema(obj: dict):
    """
    Validate the generated JSON object. Return (ok:bool, issues:list).
    """
    if not isinstance(obj, dict):
        return False, ["not_object"]
    if "error" in obj:
        # allow error object
        return True, []
    issues = []
    required = ["id", "design_ideas", "question", "qa_type", "blanks", "num_blanks", "thinking", "meta",
                "declared_answer_text"]
    for k in required:
        if k not in obj:
            issues.append(f"missing_{k}")

    # qa_type must be fill_blank
    if obj.get("qa_type") != "fill_blank":
        issues.append("qa_type_must_be_fill_blank")

    # blanks must be mapping with keys Blank1, Blank2, Blank3
    if "blanks" in obj:
        blanks = obj["blanks"]
        if not isinstance(blanks, dict):
            issues.append("blanks_not_object")
        else:
            expected = ["Blank1", "Blank2", "Blank3"]
            keys = list(blanks.keys())
            if set(keys) != set(expected):
                issues.append("blanks_must_include_Blank1_Blank2_Blank3")

    # num_blanks validation
    if "num_blanks" in obj:
        num_blanks = obj["num_blanks"]
        if not isinstance(num_blanks, int) or num_blanks < 1 or num_blanks > 3:
            issues.append("num_blanks_must_be_integer_1_to_3")

        # Check consistency between num_blanks and actual blanks
        if "blanks" in obj and isinstance(obj["blanks"], dict):
            blanks = obj["blanks"]
            non_empty_blanks = sum(1 for v in blanks.values() if v and v.strip())
            if non_empty_blanks != num_blanks:
                issues.append("num_blanks_inconsistent_with_actual_blank_answers")

    # question should contain [BLANK1], [BLANK2], [BLANK3] patterns
    if "question" in obj:
        question = obj["question"]
        if isinstance(question, str):
            blank_count = sum(1 for i in range(1, 4) if f"[BLANK{i}]" in question)
            if "num_blanks" in obj and blank_count != obj.get("num_blanks", 0):
                issues.append("question_blank_markers_inconsistent_with_num_blanks")

    # question forbidden tokens
    q = obj.get("question", "")
    if isinstance(q, str) and re.search(r"\b(mask|DSM|im1|im2|im3|segmentation)\b", q, flags=re.I):
        issues.append("question_contains_forbidden_terms")

    # meta.images_used present
    meta = obj.get("meta", {})
    if not isinstance(meta, dict):
        issues.append("meta_not_object")
    else:
        if "images_used" not in meta:
            issues.append("meta_must_include_images_used")

    # all good?
    return (len(issues) == 0), issues


# ----------------- 新增 helper: 读取已存在输出文件中的已成功样本 id -----------------
def read_existing_output_ids(path: Path):
    ids = set()
    if not path.exists():
        return ids
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                sid, rest = line.split("\t", 1)
                try:
                    obj = json.loads(rest)
                    # 仅把可解析为 JSON 且不包含 error 的条目计为成功样本
                    if isinstance(obj, dict) and "error" not in obj:
                        ids.add(sid)
                except Exception:
                    # 如果第二列不是 JSON，则忽略（可能是 warn 行）
                    continue
    except Exception as e:
        print(f"[WARN] failed to read existing output file {path}: {e}")
    return ids


def build_content_items_for_generation(task_prompt, mask_palette, prior, visible_images_filenames,
                                       hidden_images_filenames, visible_images_path, hidden_images_path,
                                       visible_images_data, hidden_images_data, dataset, visibility_stage):
    """
    Build the content items list for message payload: first item is text prompt,
    then image_url items for visible_images and optional hidden_images.

    Args:
        task_prompt: str
        mask_palette: str
        prior: str
        visible_images_filenames: list of filenames for visible images (for prompt description)
        hidden_images_filenames: list of filenames for hidden images (for prompt description)
        visible_images_path: list of full paths for visible images (for JSON record)
        hidden_images_path: list of full paths for hidden images (for JSON record)
        visible_images_data: list of tuples (filename, b64) for actual image data
        hidden_images_data: list of tuples (filename, b64) for actual image data
        dataset: str
        visibility_stage: str
    """
    # Shorten mask_palette to avoid overly long system prompt; it's already authoritative.
    mp_short = mask_palette if mask_palette and len(mask_palette) < 4000 else (mask_palette[:4000] + "...")

    # NOTE: use literal replacements (str.replace) instead of str.format to avoid
    # interpreting other JSON braces in USER_JSON_TEMPLATE.
    user_text = USER_JSON_TEMPLATE
    user_text = user_text.replace("{dataset}", str(dataset))
    user_text = user_text.replace("{visibility_stage}", str(visibility_stage))
    user_text = user_text.replace("{mask_palette}", mp_short.replace("\n", "\\n"))
    user_text = user_text.replace("{prior}", prior.replace("\n", " "))
    # 在prompt中使用文件名（简洁）
    user_text = user_text.replace("{visible_images}", json.dumps(visible_images_filenames, ensure_ascii=False))
    user_text = user_text.replace("{hidden_images}", json.dumps(hidden_images_filenames, ensure_ascii=False))
    # 在JSON schema中使用完整路径（用于记录）
    user_text = user_text.replace("{visible_images_path}", json.dumps(visible_images_path, ensure_ascii=False))
    user_text = user_text.replace("{hidden_images_path}", json.dumps(hidden_images_path, ensure_ascii=False))
    user_text = user_text.replace("{task_prompt}", task_prompt.replace("\n", "\\n"))

    content_items = [
        {"type": "text", "text": user_text}
    ]
    # attach visible images first
    for fname, b64 in visible_images_data:
        content_items.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "alt": fname}
        })
    # attach hidden images (masks/dsm) as additional images (for generation only)
    for fname, b64 in hidden_images_data:
        if b64:
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "alt": fname}
            })
    return content_items


# central method that calls the API, tries to parse JSON and repair if needed, then writes result to fout
def call_and_repair_and_write_json(sample_name: str, content_items, dataset_name: str, job: str, fout):
    """
    content_items: list for the user message content (first is text prompt, then image_url items)
    Returns True if wrote a successful JSON; always writes a line to fout: sample_name \t json_string
    """
    # build messages: system + user (list content)
    messages = [
        {"role":"system", "content": SYSTEM_MSG},
        {"role":"user", "content": content_items}
    ]

    # initial call
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        raw_text = extract_text_from_response(resp)
    except Exception as e:
        tb = traceback.format_exc()
        err_obj = {"error":"api_call_failed", "exception": str(e), "trace": tb}
        with write_lock:
            fout.write(f"{sample_name}\t{json.dumps(err_obj, ensure_ascii=False)}\n")
            fout.flush()
        print(f"[ERR] API call failed for {sample_name}: {e}")
        return False

    # extract JSON
    json_text = extract_json_text(raw_text)
    parsed = None
    issues = []
    if json_text:
        try:
            parsed = json.loads(json_text)
        except Exception as e:
            issues = [f"json_parse_error: {e}"]
            parsed = None
    else:
        issues = ["no_json_found"]

    ok = False
    if parsed is not None:
        ok, issues = validate_schema(parsed)

    attempt = 0
    last_raw = raw_text
    # repair loop if not ok
    while (not ok) and attempt < MAX_RETRIES:
        attempt += 1
        repair_msg = {
            "role":"user",
            "content": [
                {"type":"text", "text": (
                    "Previous response failed schema validation. Return ONLY the corrected JSON object (no commentary).\n"
                    f"Validation issues: {issues}\n"
                    "Previous raw response:\n" + last_raw + "\n\n"
                    "Return a single JSON object strictly matching the schema in the task prompt."
                )}
            ]
        }
        try:
            resp2 = client.chat.completions.create(model=MODEL, messages=messages + [repair_msg], temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            raw2 = extract_text_from_response(resp2)
        except Exception as e:
            tb = traceback.format_exc()
            err_obj = {"error":"api_call_failed_during_repair", "exception": str(e), "trace": tb}
            with write_lock:
                fout.write(f"{sample_name}\t{json.dumps(err_obj, ensure_ascii=False)}\n")
                fout.flush()
            print(f"[ERR] repair API call failed for {sample_name}: {e}")
            return False

        last_raw = raw2
        json_text2 = extract_json_text(raw2)
        if not json_text2:
            issues = ["no_json_found_on_repair"]
            parsed = None
            continue
        try:
            parsed = json.loads(json_text2)
        except Exception as e:
            issues = [f"json_parse_error_on_repair: {e}"]
            parsed = None
            continue
        ok, issues = validate_schema(parsed)
        if ok:
            break

    # final write (either parsed valid JSON or error object)
    if ok and parsed is not None:
        with write_lock:
            fout.write(f"{sample_name}\t{json.dumps(parsed, ensure_ascii=False)}\n")
            fout.flush()
        print(f"[OK] {dataset_name}/{job} wrote {sample_name}")
        return True
    else:
        # write error record containing last_raw and issues
        err_obj = {"error":"validation_failed", "issues": issues, "last_raw": last_raw}
        with write_lock:
            fout.write(f"{sample_name}\t{json.dumps(err_obj, ensure_ascii=False)}\n")
            fout.flush()
        print(f"[ERR] validation failed for {sample_name}: {issues}")
        return False

# =======================================================================
# === Existing helpers (unchanged) ======================================
# =======================================================================
def find_by_stem(directory: Path, stem: str):
    if not directory.exists():
        return None
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.stem == stem:
            return p
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.stem.lower() == stem.lower():
            return p
    for ext in COMMON_EXTS:
        cand = directory / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def months_to_text(months):
    years = months // 12
    remaining_months = months % 12
    parts = []
    if years > 0:
        if years == 1:
            parts.append("one year")
        elif years == 2:
            parts.append("two years")
        else:
            parts.append(f"{years} years")
    if remaining_months > 0:
        month_text = {
            1: "one month", 2: "two months", 3: "three months",
            4: "four months", 5: "five months", 6: "six months",
            7: "seven months", 8: "eight months", 9: "nine months",
            10: "ten months", 11: "eleven months"
        }
        parts.append(month_text.get(remaining_months, f"{remaining_months} months"))
    if not parts:
        return "zero months"
    return " and ".join(parts)

def calculate_month_diff(date1, date2):
    return abs((date2.year - date1.year) * 12 + (date2.month - date1.month))

# =======================================================================
# === Process functions (modified to call JSON-enforcing worker) =========
# =======================================================================
write_lock = threading.Lock()


def _do_json_generation_and_write(sample_name: str, task_prompt: str, visible_image_paths, hidden_image_paths,
                                  dataset_name: str, job: str, fout, visibility_stage="generation"):
    """
    visible_image_paths: list of Path (files humans will see) -> we convert to tuples (filename, b64)
    hidden_image_paths: list of Path (masks/dsm) -> tuples (filename, b64)
    """
    try:
        # 准备文件名列表（用于prompt中的描述）
        visible_images_filenames = [p.name for p in visible_image_paths]
        hidden_images_filenames = [p.name for p in hidden_image_paths if p is not None]

        # 准备完整路径列表（用于JSON输出中的记录）
        visible_images_path = [str(p) for p in visible_image_paths]
        hidden_images_path = [str(p) for p in hidden_image_paths if p is not None]

        # 准备图像数据（用于实际发送给API）
        visible_images_data = [(p.name, file_to_data_url(p)) for p in visible_image_paths]
        hidden_images_data = [(p.name, file_to_data_url(p)) for p in hidden_image_paths if p is not None]

    except Exception as e:
        err = {"error": "file_read_error", "exception": str(e)}
        with write_lock:
            fout.write(f"{sample_name}\t{json.dumps(err, ensure_ascii=False)}\n")
            fout.flush()
        print(f"[ERR] reading image files for {sample_name}: {e}")
        return

    # get mask_palette & prior for generation use (best-effort)
    try:
        ref_image = visible_image_paths[0].name if visible_image_paths else sample_name
        mask_palette, prior = get_mask_palette_and_prior(dataset_name,
                                                         Path(os.path.join("..", dataset_name)) if False else Path("."),
                                                         ref_image)
    except Exception:
        # fallback: empty palette
        mask_palette, prior = "", ""

    # Build content items for generation (user content)
    content_items = build_content_items_for_generation(
        task_prompt, mask_palette, prior,
        visible_images_filenames, hidden_images_filenames,  # 用于prompt描述
        visible_images_path, hidden_images_path,  # 用于JSON记录
        visible_images_data, hidden_images_data,  # 用于API发送
        dataset_name, visibility_stage
    )
    # call and repair then write
    call_and_repair_and_write_json(sample_name, content_items, dataset_name, job, fout)


def process_single_phase(dataset_path: Path, dataset_name: str, job: str, fout):
    image_dir = dataset_path / "images"
    mask_dir = dataset_path / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        fout.write(f"[WARN] expected images/ and masks/ under {dataset_path}\n")
        return
    image_paths = [p for p in sorted(image_dir.iterdir()) if p.is_file()]
    if not image_paths:
        return
    # 改为遍历全部文件，使用计数器控制是否达到上限（以便跳过已存在样本后还能继续寻找新的样本）
    count = len(EXISTING_SAMPLE_IDS) if SKIP_EXISTING_SAMPLES else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for image_path in image_paths:
            if count >= MAX_SAMPLES_PER_DATASET:
                break
            sample_id = image_path.name  # 这里 sample_name 在其他地方也以 filename (含扩展名) 作为 key
            if SKIP_EXISTING_SAMPLES and sample_id in EXISTING_SAMPLE_IDS:
                print(f"[SKIP] {dataset_name}/{job} {sample_id} already exists -> counted toward limit")
                continue

            stem = image_path.stem
            mask_path = find_by_stem(mask_dir, stem)
            if mask_path is None:
                with write_lock:
                    fout.write(f"{image_path.name}\t[WARN: missing mask for stem {stem}]\n")
                    fout.flush()
                continue

            def worker(img_p=image_path, m_p=mask_path, ds_name=dataset_name, ds_path=dataset_path, job_local=job, fout_local=fout):
                try:
                    task_prompt = get_prompt_for_one(job_local, *get_mask_palette_and_prior(ds_name, ds_path, img_p.name), datasets=ds_name)
                    # visibility rules: first-stage (human_only) -> only visible image provided, but for generation we still supply hidden images to model per your prior design.
                    # Here we use visibility_stage="human_only" during generation call to indicate that solver will later see only visible image (the generator must adhere to that).
                    _do_json_generation_and_write(img_p.name, task_prompt, [img_p], [m_p], ds_name, job_local, fout_local, visibility_stage="human_only")
                except Exception as e:
                    ans = {"error": str(e)}
                    tb = traceback.format_exc()
                    print(tb)
                    with write_lock:
                        fout_local.write(f"{img_p.name}\t{json.dumps(ans, ensure_ascii=False)}\n")
                        fout_local.flush()

            futures.append(executor.submit(worker))
            count += 1

        for f in concurrent.futures.as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"[WARN] worker raised: {e}")


def process_bi_phase(dataset_path: Path, dataset_name: str, job: str, fout):
    im1_dir = dataset_path / "im1"
    im2_dir = dataset_path / "im2"
    label1_dir = dataset_path / "label1"
    label2_dir = dataset_path / "label2"
    missing_any = False
    for d in (im1_dir, im2_dir, label1_dir, label2_dir):
        if not d.exists():
            fout.write(f"[WARN] missing expected dir: {d}\n")
            missing_any = True
    if missing_any:
        return
    im1_paths = [p for p in sorted(im1_dir.iterdir()) if p.is_file()]
    # 不提前截断，允许跳过已存在样本后继续寻找
    count = len(EXISTING_SAMPLE_IDS) if SKIP_EXISTING_SAMPLES else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for im1_path in im1_paths:
            if count >= MAX_SAMPLES_PER_DATASET:
                break
            sample_id = im1_path.name
            if SKIP_EXISTING_SAMPLES and sample_id in EXISTING_SAMPLE_IDS:
                print(f"[SKIP] {dataset_name}/{job} {sample_id} already exists -> counted toward limit")
                continue

            stem = im1_path.stem
            im2_path = find_by_stem(im2_dir, stem)
            label1_path = find_by_stem(label1_dir, stem)
            label2_path = find_by_stem(label2_dir, stem)
            if im2_path is None or label1_path is None or label2_path is None:
                with write_lock:
                    fout.write(f"{im1_path.name}\t[WARN: missing counterpart (im2/label1/label2) for stem {stem}]\n")
                    fout.flush()
                continue

            def worker(im1_p=im1_path, im2_p=im2_path, l1_p=label1_path, l2_p=label2_path, ds_name=dataset_name, ds_path=dataset_path, job_local=job, fout_local=fout):
                try:
                    task_prompt = get_prompt_for_bi(job_local, *get_mask_palette_and_prior(ds_name, ds_path, im1_p.name), datasets=ds_name)
                    # For bi-phase, follow your visibility rules (generation stage needs to know which images humans will see).
                    # Here we set visibility_stage="human_only" (you can change per-job policy)
                    visible = [im2_p] if job_local in ["reason","counterfactual"] else [im1_p, im2_p]
                    hidden = [l1_p, l2_p]
                    _do_json_generation_and_write(im1_p.name, task_prompt, visible, hidden, ds_name, job_local, fout_local, visibility_stage="human_only")
                except Exception as e:
                    ans = {"error": str(e)}
                    tb = traceback.format_exc()
                    print(tb)
                    with write_lock:
                        fout_local.write(f"{im1_p.name}\t{json.dumps(ans, ensure_ascii=False)}\n")
                        fout_local.flush()

            futures.append(executor.submit(worker))
            count += 1

        for f in concurrent.futures.as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"[WARN] worker raised: {e}")


def process_dsm_phase(dataset_path: Path, dataset_name: str, job: str, fout):
    im1_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    dsm_dir = dataset_path / "dsm"
    missing_any = False
    for d in (im1_dir, masks_dir, dsm_dir):
        if not d.exists():
            fout.write(f"[WARN] missing expected dir: {d}\n")
            missing_any = True
    if missing_any:
        return
    im1_paths = [p for p in sorted(im1_dir.iterdir()) if p.is_file()]
    # 不提前截断，允许跳过已存在样本后继续寻找
    count = len(EXISTING_SAMPLE_IDS) if SKIP_EXISTING_SAMPLES else 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for im1_path in im1_paths:
            if count >= MAX_SAMPLES_PER_DATASET:
                break
            sample_id = im1_path.name
            if SKIP_EXISTING_SAMPLES and sample_id in EXISTING_SAMPLE_IDS:
                print(f"[SKIP] {dataset_name}/{job} {sample_id} already exists -> counted toward limit")
                continue

            stem = im1_path.stem
            masks_path = find_by_stem(masks_dir, stem)
            dsm_path = find_by_stem(dsm_dir, stem)
            if masks_path is None or dsm_path is None:
                with write_lock:
                    fout.write(f"{im1_path.name}\t[WARN] missing counterpart (masks/dsm/) for stem {stem}\n")
                    fout.flush()
                continue

            def worker(im1_p=im1_path, masks_p=masks_path, dsm_p=dsm_path, ds_name=dataset_name, ds_path=dataset_path, job_local=job, fout_local=fout):
                try:
                    task_prompt = get_prompt_for_dsm(job_local, *get_mask_palette_and_prior(ds_name, ds_path, im1_p.name), datasets=ds_name)
                    # First-stage visibility: only images (no masks/dsm)
                    visible = [im1_p]
                    hidden = [masks_p, dsm_p]
                    _do_json_generation_and_write(im1_p.name, task_prompt, visible, hidden, ds_name, job_local, fout_local, visibility_stage="human_only")
                except Exception as e:
                    ans = {"error": str(e)}
                    tb = traceback.format_exc()
                    print(tb)
                    with write_lock:
                        fout_local.write(f"{im1_p.name}\t{json.dumps(ans, ensure_ascii=False)}\n")
                        fout_local.flush()

            futures.append(executor.submit(worker))
            count += 1

        for f in concurrent.futures.as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"[WARN] worker raised: {e}")


def process_miniucd(dataset_path: Path, dataset_name: str, job: str, fout):
    image_base = dataset_path / "image"
    label_base = dataset_path / "label"
    years = ["2017", "2018", "2019"]
    count = len(EXISTING_SAMPLE_IDS) if SKIP_EXISTING_SAMPLES else 0
    year_images = {}
    for year in years:
        image_dir = image_base / year
        if not image_dir.exists():
            fout.write(f"[WARN] Missing image directory for year {year}\n")
            return
        year_images[year] = list(image_dir.glob("*.tif")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        if not year_images[year]:
            fout.write(f"[WARN] No images found for year {year}\n")
            return
    stems_2017 = set(p.stem for p in year_images["2017"])
    stems_2018 = set(p.stem for p in year_images["2018"])
    stems_2019 = set(p.stem for p in year_images["2019"])
    common_stems = list(stems_2017 & stems_2018 & stems_2019)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for stem in common_stems:
            if count >= MAX_SAMPLES_PER_DATASET:
                break
            if SKIP_EXISTING_SAMPLES and stem in EXISTING_SAMPLE_IDS:
                print(f"[SKIP] {dataset_name}/{job} {stem} already exists -> counted toward limit")
                continue
            selected_images = []
            selected_masks = []
            for year in years:
                img_path = find_by_stem(image_base / year, stem)
                mask_path = find_by_stem(label_base / year, stem)
                if mask_path is None:
                    mask_dir = label_base / "masks_txt"
                    if mask_dir.exists():
                        mask_path = find_by_stem(mask_dir, stem)
                if img_path and mask_path:
                    selected_images.append(img_path)
                    selected_masks.append(mask_path)
                else:
                    break
            if len(selected_images) == 3:
                interval_1to2 = "one year"
                interval_2to3 = "one year"
                interval_total = "two years"

                def worker(sel_imgs=selected_images, sel_masks=selected_masks, ds_name=dataset_name, ds_path=dataset_path, job_local=job, fout_local=fout, stem_local=stem):
                    try:
                        task_prompt = get_prompt_for_three(job_local, *get_mask_palette_and_prior(ds_name, ds_path, sel_imgs[0].name),
                                                           datasets=ds_name, interval_1to2=interval_1to2, interval_2to3=interval_2to3, interval_total=interval_total)
                        # visibility rules per job (修正：直接使用 Path 对象)
                        if job_local in ["st_reason", "st_counterfactual"]:
                            visible = [sel_imgs[0], sel_imgs[2]]  # first & third
                        elif job_local in ["st_evolution", "st_consistency"]:
                            visible = [sel_imgs[0], sel_imgs[2]]  # first & third
                        elif job_local in ["state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]:
                            visible = [sel_imgs[0], sel_imgs[1]]  # first & second
                        else:
                            visible = [sel_imgs[0], sel_imgs[2]]  # default
                        hidden = sel_masks
                        _do_json_generation_and_write(stem_local, task_prompt, visible, hidden, ds_name, job_local, fout_local, visibility_stage="human_only")
                    except Exception as e:
                        ans = {"error": str(e)}
                        tb = traceback.format_exc()
                        print(tb)
                        with write_lock:
                            fout_local.write(f"{stem_local}\t{json.dumps(ans, ensure_ascii=False)}\n")
                            fout_local.flush()

                futures.append(executor.submit(worker))
                count += 1

        for f in concurrent.futures.as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"[WARN] worker raised: {e}")


def process_spacenet7(dataset_path: Path, dataset_name: str, job: str, fout):
    subfolders = [d for d in dataset_path.iterdir() if d.is_dir() and (d / "images").exists()]
    if not subfolders:
        fout.write("[ERROR] No valid subfolders found in spaceNet7\n")
        return
    count = len(EXISTING_SAMPLE_IDS) if SKIP_EXISTING_SAMPLES else 0
    # 根据 job 选择倍数
    high_jobs = {"state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"}
    mid_jobs = {"st_evolution", "st_consistency", "st_reason", "st_counterfactual"}
    if job in high_jobs:
        multiplier = 3
    elif job in mid_jobs:
        multiplier = 1
    else:
        multiplier = 30
    space7_Sample = MAX_SAMPLES_PER_DATASET * multiplier

    processed_folders = random.sample(subfolders, min(len(subfolders), space7_Sample))[:space7_Sample]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for selected_folder in processed_folders:
            if count >= space7_Sample:
                break
            images_dir = selected_folder / "images"
            labels_dir = selected_folder / "out_label"
            image_files = list(images_dir.glob("*.png"))
            # parse dates
            images_with_dates = []
            date_pattern = r"global_monthly_(\d{4})_(\d{2})_mosaic"
            for img_path in image_files:
                match = re.search(date_pattern, img_path.name)
                if match:
                    year = int(match.group(1)); month = int(match.group(2))
                    images_with_dates.append((img_path, datetime(year, month, 1)))
            if len(images_with_dates) < 3:
                with write_lock:
                    fout.write(f"{selected_folder.name}\t[WARN] Not enough valid images in {selected_folder.name}\n")
                    fout.flush()
                continue
            images_with_dates.sort(key=lambda x: x[1])
            valid_triplets = []
            n = len(images_with_dates)
            for i in range(n):
                for j in range(i+1, n):
                    # require i->j gap > 4 months
                    if calculate_month_diff(images_with_dates[i][1], images_with_dates[j][1]) <= 4:
                        continue
                    for k in range(j+1, n):
                        # require j->k gap > 4 months to be considered (符合原逻辑)
                        if calculate_month_diff(images_with_dates[j][1], images_with_dates[k][1]) > 4:
                            valid_triplets.append([images_with_dates[i], images_with_dates[j], images_with_dates[k]])
            # 如果没有符合条件的 triplets，保持原先回退行为（只取一组默认三张）
            if not valid_triplets:
                valid_triplets = [[images_with_dates[0], images_with_dates[n//2], images_with_dates[-1]]]

            # 逐个提交该 folder 中的 triplet，直到达到全局上限 MAX_SAMPLES_PER_DATASET
            submitted_this_folder = 0
            for selected_images in valid_triplets:
                if count >= space7_Sample:
                    break

                date1, date2, date3 = [img[1] for img in selected_images]
                months_1to2 = calculate_month_diff(date1, date2)
                months_2to3 = calculate_month_diff(date2, date3)
                months_total = calculate_month_diff(date1, date3)
                interval_1to2 = months_to_text(months_1to2)
                interval_2to3 = months_to_text(months_2to3)
                interval_total = months_to_text(months_total)

                selected_masks = []
                for img_path, _ in selected_images:
                    stem = img_path.stem
                    mask_path = find_by_stem(labels_dir, stem)
                    selected_masks.append(mask_path)
                if any(m is None for m in selected_masks):
                    with write_lock:
                        fout.write(f"{selected_folder.name}\t[WARN] Missing masks for selected images in {selected_folder.name}\n")
                        fout.flush()
                    continue

                # compute the sample id (和写入时的 sample_name 保持一致)
                resp_output_name = f"{selected_folder.name}_{selected_images[0][1].strftime('%Y%m')}_{selected_images[1][1].strftime('%Y%m')}_{selected_images[2][1].strftime('%Y%m')}"

                # 在提交前在全局集合中“预占”该 id，防止同一次运行内重复提交
                if SKIP_EXISTING_SAMPLES:
                    with write_lock:
                        already = resp_output_name in EXISTING_SAMPLE_IDS
                        if not already:
                            EXISTING_SAMPLE_IDS.add(resp_output_name)
                    if already:
                        print(f"[SKIP] {dataset_name}/{job} {resp_output_name} already exists -> counted toward limit")
                        continue

                def worker(sel_imgs=selected_images, sel_masks=selected_masks, ds_name=dataset_name, ds_path=dataset_path, job_local=job, fout_local=fout, folder_name=selected_folder.name, resp_name=resp_output_name):
                    try:
                        task_prompt = get_prompt_for_three(job_local, *get_mask_palette_and_prior(ds_name, ds_path, sel_imgs[0][0].name),
                                                         datasets=ds_name, interval_1to2=interval_1to2, interval_2to3=interval_2to3, interval_total=interval_total)
                        # visibility rules per job:
                        if job_local in ["st_reason", "st_counterfactual"]:
                            visible_paths = [sel_imgs[0][0], sel_imgs[2][0]]  # second & third
                        elif job_local in ["st_evolution", "st_consistency"]:
                            visible_paths = [sel_imgs[0][0], sel_imgs[2][0]]  # first & third
                        elif job_local in ["state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]:
                            visible_paths = [sel_imgs[0][0], sel_imgs[1][0]]
                        else:
                            visible_paths = [sel_imgs[0][0], sel_imgs[2][0]]  # default
                        hidden_paths = [m for m in sel_masks if m is not None]
                        _do_json_generation_and_write(resp_name, task_prompt, visible_paths, hidden_paths, ds_name, job_local, fout_local, visibility_stage="human_only")
                    except Exception as e:
                        ans = {"error": str(e)}
                        tb = traceback.format_exc()
                        print(tb)
                        with write_lock:
                            fout_local.write(f"{folder_name}\t{json.dumps(ans, ensure_ascii=False)}\n")
                            fout_local.flush()

                futures.append(executor.submit(worker))
                count += 1
                submitted_this_folder += 1

            # 可选：记录每个 folder 实际提交数量（不改变主体逻辑）
            with write_lock:
                fout.write(f"{selected_folder.name}\tsubmitted_triplets={submitted_this_folder}\n")
                fout.flush()

        for f in concurrent.futures.as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"[WARN] worker raised: {e}")


from multi_dataset_prompt import get_prompt_for_multi


def process_multi_spacenet7(dataset_path: Path, dataset_name: str, job: str, fout, n_phases: int = 3):
    if n_phases < 3:
        n_phases = 3  # 最小3时相
    subfolders = [d for d in dataset_path.iterdir() if d.is_dir() and (d / "images").exists()]
    if not subfolders:
        fout.write("[ERROR] No valid subfolders found in spaceNet7\n")
        return
    # 根据 job 选择倍数
    # high_jobs = {"state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"}
    # mid_jobs = {"st_evolution", "st_consistency", "st_reason", "st_counterfactual"}
    # if job in high_jobs:
    #     multiplier = 1
    # elif job in mid_jobs:
    #     return
    # else:
    #     multiplier = 30
  # n_phases 对应倍率：4~9 时相
    phase_multiplier = {
    4: 15,
    5: 13,
    6: 11,
    7: 9,
    8: 7,
    9: 5,
    }
    phase_mult = phase_multiplier.get(n_phases, 1)

    space7_Sample = MAX_SAMPLES_PER_DATASET * phase_mult
    count = len(EXISTING_SAMPLE_IDS) if SKIP_EXISTING_SAMPLES else 0

    processed_folders = random.sample(subfolders, min(len(subfolders), space7_Sample))[:space7_Sample]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for selected_folder in processed_folders:
            if count >= space7_Sample:
                break
            images_dir = selected_folder / "images"
            labels_dir = selected_folder / "out_label"
            image_files = list(images_dir.glob("*.png"))
            # parse dates
            images_with_dates = []
            date_pattern = r"global_monthly_(\d{4})_(\d{2})_mosaic"
            for img_path in image_files:
                match = re.search(date_pattern, img_path.name)
                if match:
                    year = int(match.group(1));
                    month = int(match.group(2))
                    images_with_dates.append((img_path, datetime(year, month, 1)))
            images_with_dates.sort(key=lambda x: x[1])
            n = len(images_with_dates)
            if n < n_phases:
                with write_lock:
                    fout.write(
                        f"{selected_folder.name}\t[WARN] Not enough valid images ({n} < {n_phases}) in {selected_folder.name}\n")
                    fout.flush()
                continue

            # 生成valid_sequences：长度==n_phases，每个相邻gap >4 months
            valid_sequences = []
            for start in range(n - n_phases + 1):
                seq = images_with_dates[start:start + n_phases]
                valid = True
                for idx in range(len(seq) - 1):
                    if calculate_month_diff(seq[idx][1], seq[idx + 1][1]) <= 4:
                        valid = False
                        break
                if valid:
                    valid_sequences.append(seq)

            # 如果没有，回退默认均匀选择
            if not valid_sequences:
                step = (n - 1) / (n_phases - 1)
                indices = [int(round(i * step)) for i in range(n_phases)]
                indices = sorted(set(indices))  # 去重，确保唯一
                if len(indices) == n_phases:
                    valid_sequences = [[images_with_dates[idx] for idx in indices]]

            # 如果还是没有，跳过
            if not valid_sequences:
                with write_lock:
                    fout.write(
                        f"{selected_folder.name}\t[WARN] No valid sequences of length {n_phases} in {selected_folder.name}\n")
                    fout.flush()
                continue

            # 随机选择序列提交，直到上限
            random.shuffle(valid_sequences)
            submitted_this_folder = 0
            for selected_images in valid_sequences:
                if count >= space7_Sample:
                    break
                dates = [img[1] for img in selected_images]
                intervals_list = []
                for i in range(len(dates) - 1):
                    months = calculate_month_diff(dates[i], dates[i + 1])
                    intervals_list.append(months_to_text(months))
                months_total = calculate_month_diff(dates[0], dates[-2])
                interval_total = months_to_text(months_total)

                selected_masks = []
                for img_path, _ in selected_images:
                    stem = img_path.stem
                    mask_path = find_by_stem(labels_dir, stem)
                    selected_masks.append(mask_path)
                if any(m is None for m in selected_masks):
                    with write_lock:
                        fout.write(
                            f"{selected_folder.name}\t[WARN] Missing masks for selected images in {selected_folder.name}\n")
                        fout.flush()
                    continue

                # resp_name with all dates
                date_str = "_".join([d.strftime('%Y%m') for d in dates])
                resp_output_name = f"{selected_folder.name}_{date_str}"

                if SKIP_EXISTING_SAMPLES:
                    with write_lock:
                        already = resp_output_name in EXISTING_SAMPLE_IDS
                        if not already:
                            EXISTING_SAMPLE_IDS.add(resp_output_name)
                    if already:
                        print(f"[SKIP] {dataset_name}/{job} {resp_output_name} already exists -> counted toward limit")
                        continue

                def worker(sel_imgs=selected_images, sel_masks=selected_masks, ds_name=dataset_name,
                           ds_path=dataset_path, job_local=job, fout_local=fout, folder_name=selected_folder.name,
                           resp_name=resp_output_name, intervals_local=intervals_list, total_local=interval_total):
                    try:
                        task_prompt = get_prompt_for_multi(job_local, *get_mask_palette_and_prior(ds_name, ds_path,
                                                                                                  sel_imgs[0][0].name),
                                                           datasets=ds_name, intervals=intervals_local,
                                                           interval_total=total_local)
                        # visibility rules per job (universal for any n_phases)
                        n_local = len(sel_imgs)
                        n_local = len(sel_imgs)
                        mid_idx = n_local // 2  # 奇偶统一：仅隐藏这一张中间图if job_local in ["st_reason", "st_counterfactual"]:

                        if job_local in ["st_reason", "st_counterfactual"]:
                            hidden_idx = 1
                            visible_paths = [img[0] for idx, img in enumerate(sel_imgs) if idx not in hidden_idx]
                        elif job in ["st_evolution", "st_consistency"]:
                            hidden_idx = mid_idx if n_local > 0 else 0
                            visible_paths = [img[0] for idx, img in enumerate(sel_imgs) if idx not in hidden_idx]

                        elif job_local in ["state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]:
                            visible_paths = [img[0] for img in sel_imgs[:-1]]  # all except last
                            hidden_idx = n_local - 1 if n_local > 0 else 0
                        else:
                            visible_paths = [sel_imgs[0][0], sel_imgs[-1][0]]  # default
                            hidden_idx = n_local - 1 if n_local > 0 else 0

                        hidden_paths = [img[0] for idx, img in enumerate(sel_imgs) if idx == hidden_idx]
                        _do_json_generation_and_write(resp_name, task_prompt, visible_paths, hidden_paths, ds_name,
                                                      job_local, fout_local, visibility_stage="human_only")
                    except Exception as e:
                        ans = {"error": str(e)}
                        tb = traceback.format_exc()
                        print(tb)
                        with write_lock:
                            fout_local.write(f"{folder_name}\t{json.dumps(ans, ensure_ascii=False)}\n")
                            fout_local.flush()

                futures.append(executor.submit(worker))
                count += 1
                submitted_this_folder += 1

            with write_lock:
                fout.write(f"{selected_folder.name}\tsubmitted_sequences={submitted_this_folder}\n")
                fout.flush()

        for f in concurrent.futures.as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                print(f"[WARN] worker raised: {e}")


def process_tri_phase(dataset_path: Path, dataset_name: str, job: str, fout, n_phases: int):
    """
    处理三时相数据集（miniucd 和 spaceNet7）
    保持原有逻辑，仅把实际发送请求与写入并发化
    """
    n_phases = n_phases

    if dataset_name.lower() == "miniucd":
        process_miniucd(dataset_path, dataset_name, job, fout)
    elif dataset_name.lower() == "spacenet7":
        if n_phases == 3:
            process_spacenet7(dataset_path, dataset_name, job, fout)
        else:
            process_multi_spacenet7(dataset_path, dataset_name, job, fout, n_phases=n_phases)
    else:
        fout.write(f"[WARN] Unknown tri-phase dataset: {dataset_name}\n")


# =======================================================================
# === Runner =============================================================
# =======================================================================
def run_all_datasets():
    for dataset_path_str in DATASET_ROOTS:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            print(f"[WARN] 数据集路径不存在: {dataset_path}, 跳过")
            continue
        dataset_name = dataset_path.name
        print(f"\n=== Processing dataset: {dataset_name} (path: {dataset_path}) ===")
        base_outdir = RESULT_OUTDIR  # 不要在循环里改全局 RESULT_OUTDIR for dataset_path_str in DATASET_ROOTS:
        dataset_path = Path(dataset_path_str)
        if not dataset_path.exists():
            print(f"[WARN] 数据集路径不存在: {dataset_path}, 跳过")
            continue
        dataset_name = dataset_path.name
        print(f"\n=== Processing dataset: {dataset_name} (path: {dataset_path}) ===")

        # 仅 spaceNet7 一次性跑4~9 时相；其他数据集保持3 时相
        if dataset_name.lower() == "spacenet7":
            phase_list = [4, 5, 6, 7, 8, 9]
        else:
            phase_list = [3]

        for n_phases in phase_list:
            # 输出目录：3时相保持原目录；4~9时相写入子目录，避免互相覆盖
            if n_phases ==3:
                outdir = base_outdir / dataset_name
            else:
                outdir = base_outdir / str(n_phases) / dataset_name
            outdir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] n_phases={n_phases}, outdir={outdir}")

            for job in JOBS:
                output_path = outdir / f"QASet_{job}_{dataset_name}_{MODEL}.txt"
                print(f"-> job={job}, n_phases={n_phases}, writing to {output_path}")

                global EXISTING_SAMPLE_IDS
                EXISTING_SAMPLE_IDS = set()
                open_mode = "w"
                if SKIP_EXISTING_SAMPLES and output_path.exists():
                    EXISTING_SAMPLE_IDS = read_existing_output_ids(output_path)
                open_mode = "a"
                print(
                    f"[INFO] Found {len(EXISTING_SAMPLE_IDS)} existing successful samples in {output_path}, "
                    f"they will be counted toward {MAX_SAMPLES_PER_DATASET}."
                )

                with open(output_path, open_mode, encoding="utf-8") as fout:
                    if dataset_name.lower() in ["xview2", "second"]:
                        if job in ["reason", "counterfactual", "st_reason", "st_counterfactual", "st_evolution",
                                   "st_consistency"]:
                            process_bi_phase(dataset_path, dataset_name, job, fout)
                    elif dataset_name.lower() in ["potsdam", "vaihingen"]:
                        if job in ["reason", "counterfactual", "influ", "sem", "plan", "estimate"]:
                            process_dsm_phase(dataset_path, dataset_name, job, fout)
                    elif dataset_name.lower() in ["miniucd", "spacenet7"]:
                        if job in ["st_reason", "st_counterfactual", "st_evolution", "st_consistency",
                                   "state_predict", "shape_predict", "scene_uncertainty", "sequence_predict"]:
                            process_tri_phase(dataset_path, dataset_name, job, fout, n_phases)
                    else:
                        if job in ["reason", "counterfactual", "influ", "sem", "plan", "estimate"]:
                            process_single_phase(dataset_path, dataset_name, job, fout)

                print(f"[DONE] {dataset_name}-{job}-n{n_phases}, results saved to {output_path}")


if __name__ == "__main__":
    # allow overriding behavior via environment variables before running
    # global SKIP_EXISTING_SAMPLES, MAX_SAMPLES_PER_DATASET, MAX_WORKERS
    #
    # env_skip = os.getenv("SKIP_EXISTING_SAMPLES")
    # if env_skip is not None:
    #     SKIP_EXISTING_SAMPLES = env_skip.strip() not in ("0", "false", "False")
    #
    # env_max = os.getenv("MAX_SAMPLES_PER_DATASET")
    # if env_max:
    #     try:
    #         MAX_SAMPLES_PER_DATASET = int(env_max)
    #     except Exception:
    #         print(f"[WARN] invalid MAX_SAMPLES_PER_DATASET env value: {env_max}")

    # env_workers = os.getenv("RUN_ALL_MAX_WORKERS")
    # if env_workers:
    #     try:
    #         MAX_WORKERS = int(env_workers)
    #     except Exception:
    #         print(f"[WARN] invalid RUN_ALL_MAX_WORKERS env value: {env_workers}")
    #
    # print(f"[CONFIG] SKIP_EXISTING_SAMPLES={SKIP_EXISTING_SAMPLES}, MAX_SAMPLES_PER_DATASET={MAX_SAMPLES_PER_DATASET}, MAX_WORKERS={MAX_WORKERS}")

    try:
        run_all_datasets()
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user (KeyboardInterrupt)")
    except Exception:
        print("[ERROR] unexpected exception during run:")
        traceback.print_exc()
    else:
        print("[INFO] run_all_datasets completed normally")
