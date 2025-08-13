# DNN Channel Estimation Training - 프로젝트 문서

> 5G/6G 통신을 위한 딥러닝 기반 DMRS 채널 추정 시스템

## 📚 문서 구조

### 핵심 문서
- **[CLAUDE.md](./CLAUDE.md)** - 프로젝트 전체 기술 분석 및 메모리
- **[engine_v4_development.md](./engine_v4_development.md)** - v4 베이스 모델 개발 가이드  
- **[code_architecture.md](./code_architecture.md)** - 시스템 아키텍처 상세 가이드

## 🚀 빠른 시작 가이드

### 환경 설정 (Vast AI)
```bash
# 자동 설치 스크립트
curl -sSL https://raw.githubusercontent.com/joowonoil/channel-estimation-training/main/setup_vast_ai.sh | bash

# 또는 수동 설치
git clone https://github.com/joowonoil/channel-estimation-training.git
cd channel-estimation-training
```

### 모델 훈련
```bash
# v4 베이스 모델 훈련
python engine_v4.py

# LoRA 전이학습 (InF 환경)
python Transfer_v4_InF.py

# LoRA 전이학습 (RMa 환경)  
python Transfer_v4_RMa.py

# 성능 테스트
python simple_model_test.py
```

## 🏗️ 프로젝트 구조

```
DNN_channel_estimation_training/
├── 🎯 실행 파일
│   ├── engine_v4.py           # v4 베이스 모델 훈련
│   ├── Transfer_v4_*.py       # LoRA 전이학습
│   └── simple_model_test.py   # 성능 검증
├── 🧠 model/                  # DNN 모델 아키텍처
│   ├── estimator_v4.py        # LoRA 호환 채널 추정기
│   └── transformer_v4.py      # 분리된 projection Transformer
├── ⚙️ config/                 # 설정 파일
│   └── config_transfer_v4_*.yaml
├── 📊 dataset/                # 채널 데이터 (Git LFS)
└── 💾 saved_model/            # 훈련된 모델
```

## 🔬 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **프레임워크** | PyTorch 2.4.1, CUDA 12.1 |
| **모델** | Transformer + LoRA |
| **최적화** | TensorRT, ONNX |
| **실험 관리** | Weights & Biases |
| **배포** | Docker, Vast AI |

## 🎯 핵심 기능

### 1. LoRA 전이학습
- **Low-Rank Adaptation**을 통한 효율적 파라미터 적응
- 베이스 모델 대비 1% 미만의 파라미터로 높은 성능 달성
- InF (Indoor Factory), RMa (Rural Macro) 환경 특화

### 2. v4 아키텍처
- 분리된 projection layer로 LoRA 타겟 모듈 명확화
- 완벽한 가중치 호환성 보장
- 설정 기반 유연한 시스템

### 3. 채널 추정 성능
- DMRS 기반 5G/6G 채널 추정
- 복소수 채널 응답 정확도 향상
- 실시간 추론 가능 (TensorRT 최적화)

## 📊 데이터셋

### 지원 채널 타입
- **InF**: Indoor Factory (Los/NLos) - 50,000 샘플
- **RMa**: Rural Macro (Los/NLos) - 50,000 샘플
- **InH**: Indoor Hotspot (Los/NLos)
- **UMa/UMi**: Urban Macro/Micro

### 데이터 형식
- PDP (Power Delay Profile): `.mat` 파일
- 샘플 데이터: `.npy`, `.npz` 파일
- 모델 체크포인트: `.pt` 파일

## ⚙️ 설정 관리

### config.yaml 구조
```yaml
dataset:
  channel_type: ["InF_Los", "InF_Nlos"]
  batch_size: 32

training:
  lr: 0.00001
  optimizer: Adam
  num_iter: 200000
  
ch_estimation:
  peft:  # LoRA 설정
    r: 8
    lora_alpha: 8
    target_modules: ["mha_q_proj", "mha_k_proj", "mha_v_proj"]
```

## 🔄 개발 워크플로우

### 1. 베이스 모델 훈련
```bash
# config.yaml 수정 후
python engine_v4.py
# → saved_model/Large_estimator_v4_base.pt 생성
```

### 2. LoRA 전이학습
```bash
# config_transfer_v4_InF.yaml 수정 후
python Transfer_v4_InF.py
# → saved_model/Large_estimator_v4_to_InF_*.pt 생성
```

### 3. TensorRT 최적화
```bash
python tensorrt_conversion_v4.py
# → *.engine 파일 생성
```

## 🐳 Docker 환경

```bash
# 사전 준비된 환경 사용
docker pull joowonoil/channel-estimation-env:latest
docker run --gpus all -it joowonoil/channel-estimation-env:latest
```

포함 내용:
- PyTorch 2.4.1 + CUDA 12.1
- transformers, peft (LoRA)
- TensorRT, ONNX
- 모든 필수 의존성

## 📈 성능 메트릭

- **NMSE**: Normalized Mean Square Error
- **채널 추정 정확도**: 95%+ (InF 환경)
- **추론 속도**: 10ms 이하 (TensorRT)
- **메모리 효율**: LoRA로 90% 감소

## 🛠️ 트러블슈팅

### CUDA 메모리 부족
```yaml
# config에서 batch_size 줄이기
batch_size: 16  # 32 → 16
```

### 가중치 로드 실패
```python
# v4 베이스 모델 경로 확인
pretrained_model_name: 'Large_estimator_v4_base'
```

### LoRA 파라미터 확인
```python
python check_lora_params.py
```

## 📝 기여 가이드

1. 코드 수정 시 관련 문서 업데이트
2. 새로운 기능은 config 파일에 설정 추가
3. 모델 변경 시 호환성 테스트 필수

## 🔮 향후 계획

- [ ] 멀티 GPU 분산 훈련 지원
- [ ] 동적 LoRA 랭크 조정
- [ ] 더 많은 채널 환경 지원
- [ ] 실시간 적응형 채널 추정

## 📄 라이선스

MIT License

## 🙋‍♂️ 지원

문제 발생 시 [GitHub Issues](https://github.com/joowonoil/channel-estimation-training/issues)에 등록

---

*최종 업데이트: 2025-01-13*