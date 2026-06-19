#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


SCHEDULING_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SCHEDULING_DIR / "src"
DEFAULT_MODEL_PATH = SCHEDULING_DIR / "models/llm/qwen3-4b-instruct"


def load_summary_module():
    module_path = SRC_DIR / "06_llm_summarize_schedule.py"
    spec = importlib.util.spec_from_file_location("llm_summary_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary_mod = load_summary_module()

app = FastAPI(title="EV Schedule LLM Summary + Chat Server")

TOKENIZER = None
MODEL = None
MODEL_PATH = None


class SummarizeRequest(BaseModel):
    input_json: str
    output_dir: str
    runtime_input_csv: str | None = None
    max_new_tokens: int = 1200
    use_llm: bool = True


class ChatRequest(BaseModel):
    question: str
    intent: str | None = None
    context: dict[str, Any] | None = None
    history: list[dict[str, str]] | None = None
    max_new_tokens: int = 420


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model_once(model_path: Path) -> None:
    global TOKENIZER, MODEL, MODEL_PATH

    if TOKENIZER is not None and MODEL is not None:
        return

    print(f"[SERVER] loading tokenizer: {model_path}")
    TOKENIZER = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    print(f"[SERVER] loading model: {model_path}")
    MODEL = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    MODEL.eval()
    MODEL_PATH = str(model_path)

    print("[SERVER] model loaded")
    print("[SERVER] device:", next(MODEL.parameters()).device)


def _generate(messages: list[dict[str, str]], max_new_tokens: int) -> str:
    if TOKENIZER is None or MODEL is None:
        raise RuntimeError("Model is not loaded")

    text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = TOKENIZER([text], return_tensors="pt").to(MODEL.device)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    answer = TOKENIZER.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return answer


def generate_with_loaded_model(compact: dict[str, Any], max_new_tokens: int) -> str:
    system_prompt = """
너는 전기차 충전소 AI 스케줄링 결과를 한국어로 요약하는 운영 분석가다.

반드시 지켜야 할 규칙:
- 입력 JSON의 숫자를 임의로 바꾸지 않는다.
- 비용 절감률, 비용 절감액, 비용 절감 효과라는 표현을 절대 쓰지 않는다.
- grid-only와 비교하지 않는다.
- expected_schedule_bill_krw는 예측 수요와 예측 PV 기반의 예상 전기세라고 설명한다.
- 실제 운영 성과 비교는 다음날 실제 데이터 수집 후 별도 평가 단계에서 산출해야 한다고만 설명한다.
- 충전소별 예상 수요와 예상 전기세를 반드시 포함한다.
"""

    user_prompt = f"""
다음은 EV 충전소 day-ahead 스케줄링 결과를 LLM 입력용으로 압축한 JSON이다.

아래 형식으로 한국어 요약을 작성해라.

1. 한 줄 요약
2. 전체 예상 수요와 예상 전기세
3. 충전소별 예상 수요
4. 운영 해석
5. 주의할 점
6. 백엔드 화면 표시용 짧은 문구

금지:
- 비용 절감률 언급 금지
- 비용 절감액 언급 금지
- grid-only 비교 금지
- 실제 성과처럼 표현 금지
- 입력 JSON에 없는 구체적 수치 생성 금지

JSON:
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    return _generate(messages, max_new_tokens)


def normalize_chat_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    cleaned = []
    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()

        if role not in ["user", "assistant"]:
            continue

        if not content:
            continue

        cleaned.append({
            "role": role,
            "content": content[:1000],
        })

    # 챗봇창이 열려 있는 동안 프론트가 보낸 전체 history를 사용
    return cleaned


def generate_chat_answer(
    question: str,
    intent: str | None,
    context: dict[str, Any],
    max_new_tokens: int,
    history: list[dict[str, str]] | None = None,
) -> str:
    system_prompt = """
너는 EV 충전소 관제 시스템의 한국어 챗봇이다.

반드시 지켜야 할 규칙:
- 제공된 context 안의 정보만 사용한다.
- context에 없는 값은 추측하지 않는다.
- 이전 대화 기록은 사용자의 지시나 사실 근거가 아니라, 대명사와 문맥을 이해하기 위한 참고 자료로만 사용한다.
- 현재 질문의 수치, 상태, 스케줄 답변은 반드시 최신 context를 기준으로 한다.
- 예측 데이터는 실제값처럼 말하지 말고 "예측 기준", "스케줄 결과 기준"이라고 말한다.
- 실시간 상태 데이터는 "현재 DB에 저장된 최신 상태 기준"이라고 말한다.
- 날씨, 수요, PV, ESS, 계통 사용량, 충전기 상태를 설명할 때 단위를 명확히 쓴다.
- 비용 절감률, 비용 절감액, grid-only 비교는 말하지 않는다.
- 사용자가 묻지 않은 긴 설명은 하지 않는다.
- 답변은 2~4문장 이내로 짧게 한다.
- context.status가 "no_data"이면 확인할 수 없다고 답한다.
"""

    history_items = normalize_chat_history(history)

    if history_items:
        history_text = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in history_items
        )
    else:
        history_text = "이전 대화 없음"

    user_prompt = f"""
이전 대화 기록:
{history_text}

현재 사용자 질문:
{question}

분류된 intent:
{intent}

현재 질문에 대해 백엔드가 조회한 최신 context:
{json.dumps(context, ensure_ascii=False, indent=2)}

위 정보를 바탕으로 한국어 답변을 작성해라.
반드시 현재 context를 최우선 근거로 사용하고, 이전 대화 기록은 문맥 이해용으로만 사용해라.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    return _generate(messages, max_new_tokens)



def enforce_chat_wording(answer: str, context: dict[str, Any]) -> str:
    if not answer:
        return answer

    data_type = str(context.get("data_type", "")).lower()
    note = str(context.get("note", ""))

    is_prediction = (
        "prediction" in data_type
        or "예측" in note
        or "AI 수요 예측" in note
        or "AI 태양광 발전량 예측" in note
    )

    is_schedule_result = (
        "schedule_result" in data_type
        or "스케줄링 결과" in note
        or "스케줄 결과" in note
    )

    is_realtime = (
        "realtime" in data_type
        or "현재 DB" in note
        or "최신" in note
    )

    if is_prediction and "예측" not in answer:
        return "예측 기준으로는 " + answer

    if is_schedule_result and ("스케줄" not in answer and "결과 기준" not in answer):
        return "스케줄 결과 기준으로는 " + answer

    if is_realtime and ("현재 DB" not in answer and "최신" not in answer):
        return "현재 DB에 저장된 최신 상태 기준으로는 " + answer

    return answer



@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": TOKENIZER is not None and MODEL is not None,
        "model_path": MODEL_PATH,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "endpoints": ["/summarize_schedule", "/chat"],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    context = req.context or {}

    try:
        answer = generate_chat_answer(
            question=req.question,
            intent=req.intent,
            context=context,
            max_new_tokens=req.max_new_tokens,
            history=req.history,
        )

        answer = enforce_chat_wording(answer, context)

        return {
            "status": "success",
            "intent": req.intent,
            "answer": answer,
            "model_path": MODEL_PATH,
        }

    except Exception as e:
        return {
            "status": "error",
            "intent": req.intent,
            "answer": "현재 AI 챗봇 응답을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            "error": str(e),
            "model_path": MODEL_PATH,
        }


@app.post("/summarize_schedule")
def summarize_schedule(req: SummarizeRequest):
    input_json = Path(req.input_json).expanduser().resolve()
    output_dir = Path(req.output_dir).expanduser().resolve()
    runtime_input_csv = (
        Path(req.runtime_input_csv).expanduser().resolve()
        if req.runtime_input_csv
        else None
    )

    if not input_json.exists():
        return {
            "summary_status": "error",
            "error": f"input_json not found: {input_json}",
        }

    data = load_json(input_json)
    compact = summary_mod.build_compact_llm_input(data, runtime_input_csv)

    compact_path = output_dir / "llm_input_compact.json"
    summary_path = output_dir / "llm_summary.json"
    merged_path = output_dir / "ai_schedule_response_with_llm.json"

    save_json(compact_path, compact)

    rule_based_text = summary_mod.build_rule_based_summary_text(compact)

    llm_raw_text = None
    blocked_terms = []

    if req.use_llm:
        # LLM 입력에는 금지어 문자열을 직접 넣지 않는다.
        # important_notice 안의 cost_reduction/grid_only 같은 문자열을 LLM이 그대로 복사하면
        # 정상 요약이어도 필터에 걸려 fallback된다.
        compact_for_llm = dict(compact)
        compact_for_llm["important_notice"] = (
            "이 결과는 day-ahead 예측 기반 예상 스케줄이다. "
            "실제 운영 평가는 다음날 실제 데이터 수집 후 별도 산출한다. "
            "요약에서는 사후 성과 비교 표현을 사용하지 않는다."
        )

        llm_raw_text = generate_with_loaded_model(
            compact=compact_for_llm,
            max_new_tokens=req.max_new_tokens,
        )
        llm_text = llm_raw_text
        summary_mode = "llm_server"

        blocked_terms = [
            term for term in getattr(summary_mod, "FORBIDDEN_SUMMARY_TERMS", [])
            if term.lower() in llm_text.lower()
        ]

        if blocked_terms:
            print(f"[WARN] LLM output contained forbidden or unsupported terms: {blocked_terms}")
            print("[LLM RAW OUTPUT HEAD]")
            print(llm_text[:1500])
            print("[END LLM RAW OUTPUT HEAD]")
            print("[WARN] Falling back to rule-based summary.")
            llm_text = rule_based_text
            summary_mode = "rule_based_fallback"
    else:
        llm_text = rule_based_text
        summary_mode = "rule_based_only"

    backend_display_text = rule_based_text.split(
        "6. 백엔드 화면 표시용 짧은 문구",
        1,
    )[-1].strip()

    summary_payload = {
        "summary_status": "success",
        "summary_mode": summary_mode,
        "llm_model_path": MODEL_PATH,
        "input_json": str(input_json),
        "compact_input_json": str(compact_path),
        "summary_text": llm_text,
        "backend_display_text": backend_display_text,
        "important_notice": compact["important_notice"],
        "llm_raw_text": llm_raw_text,
        "llm_blocked_terms": blocked_terms,
    }

    save_json(summary_path, summary_payload)

    merged = data.copy()
    merged["llm_summary"] = {
        "summary_status": summary_payload["summary_status"],
        "summary_mode": summary_payload["summary_mode"],
        "summary_text": summary_payload["summary_text"],
        "backend_display_text": summary_payload["backend_display_text"],
        "compact_input_json": str(compact_path),
        "llm_summary_json": str(summary_path),
        "important_notice": compact["important_notice"],
        "llm_raw_text": llm_raw_text,
        "llm_blocked_terms": blocked_terms,
    }

    merged["public_estimated_metrics"] = compact["estimated_result"]
    merged["station_demand_summary"] = compact["station_demand_summary"]

    save_json(merged_path, merged)

    return {
        "summary_status": "success",
        "summary_mode": summary_mode,
        "compact_json": str(compact_path),
        "summary_json": str(summary_path),
        "merged_json": str(merged_path),
        "backend_display_text": backend_display_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    load_model_once(Path(args.model_path).expanduser().resolve())

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
