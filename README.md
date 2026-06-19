# EV AI Scheduling Inference Package

전기차 충전소 AI 스케줄링 추론 패키지입니다.

## 포함 항목

- 충전 수요 예측 추론 코드
- PV 발전량 예측 추론 코드
- SAC/RL 스케줄링 추론 코드
- 충전소 간 transfer 후처리 코드
- LLM 스케줄 요약 서버 코드
- 챗봇 /chat 엔드포인트 코드
- 학습 완료된 수요/PV/RL 모델

## 제외 항목

- Qwen LLM 모델 원본
- 학습 코드
- 원본 학습 데이터
- output/log/cache/venv
- API key, token, SSH key

## 실행 방법

스케줄링만 실행:

python3 src/05_run_scheduling_pipeline.py --input-json runtime_requests/sample_backend_request.json

LLM 서버 실행:

python3 src/llm_summary_server.py --host 0.0.0.0 --port 18080 --model-path models/llm/qwen3-4b-instruct

LLM 요약 포함 전체 파이프라인:

python3 src/05_run_scheduling_pipeline_v2_llm.py --input-json runtime_requests/sample_backend_request.json

## LLM 모델

Qwen 모델 원본은 저장소에 포함하지 않았습니다.
LLM 요약/챗봇 기능을 사용하려면 models/llm/qwen3-4b-instruct/ 경로에 모델을 별도로 배치해야 합니다.
