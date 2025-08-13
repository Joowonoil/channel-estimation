# DNN 채널 추정 프로젝트 분석 - Claude 메모리

## 🎯 프로젝트 개요
- **목적**: 5G/6G 통신을 위한 DNN 기반 DMRS 채널 추정 시스템
- **핵심 기술**: LoRA(Low-Rank Adaptation) 전이학습을 통한 효율적 모델 적응
- **특화 환경**: InF(Indoor Factory) 채널 환경

## 🏗️ 핵심 아키텍처

### 모델 구조
- **기반 모델**: Transformer 기반 채널 추정기
- **어텐션**: Self-attention + Cross-attention
- **정규화**: Pre-LayerNorm 방식
- **입력 형태**: (batch, 14_symbols, 3072_subcarriers, 2_channels)
- **출력**: 추정 채널 응답 + 보상된 수신 신호

### LoRA 설정
```yaml
peft:
  peft_type: LORA
  r: 8                    # LoRA rank
  lora_alpha: 8          # scaling factor
  target_modules: ["mha_q_proj", "mha_k_proj", "mha_v_proj", "out_proj", "ffnn_linear1", "ffnn_linear2"]
  lora_dropout: 0.1
```

## 📊 데이터 구조

### 채널 타입
- **InF_Los_50000**: Indoor Factory Line-of-Sight (50,000 샘플)
- **InF_Nlos_50000**: Indoor Factory Non-Line-of-Sight (50,000 샘플)
- **RMa_Los**: Rural Macro (전이학습 소스)

### 참조 신호 설정
- **DMRS**: [0, 3072, 6] - 첫 번째 심볼의 6간격 서브캐리어
- **FFT 크기**: 4096
- **유효 서브캐리어**: 3072 (가드 제외)
- **심볼 수**: 14

## 🔧 주요 파일 구조

### 핵심 실행 파일
- **Transfer_v4.py**: LoRA 전이학습 메인 엔진
- **model/estimator_v4.py**: LoRA 적용 가능한 채널 추정 모델
- **model/transformer_v4.py**: 분리된 projection layer Transformer

### 설정 파일
- **config/config_transfer_v4.yaml**: v4 모델 전용 설정
- **config/config_transfer_v4_InF.yaml**: InF 환경 특화 설정
- **config/config_transfer_v4_RMa.yaml**: RMa 환경 특화 설정

### 데이터 관련
- **dataset.py**: 채널 데이터셋 로더 (PDP 기반)
- **dataset/PDP_processed/**: 전처리된 채널 PDP 데이터 (.mat)
- **sample_data_*/**: 다양한 거리별 샘플 데이터 (.npy, .npz)

## 🚀 훈련 설정

### 최적화 파라미터
```yaml
training:
  lr: 0.00001                    # 낮은 학습률 (전이학습)
  num_iter: 200000              # 최대 이터레이션
  batch_size: 32
  use_scheduler: true           # Cosine Annealing
  max_norm: 1.0                # 그래디언트 클리핑
```

### 모델 로드 방식
- **pretrained**: 사전 훈련된 `Large_estimator_PreLN.pt` 로드
- **finetune**: 기존 체크포인트에서 계속 학습

## 📈 성능 평가
- **손실 함수**: NMSE (Normalized Mean Square Error)
- **평가 지표**: 채널 NMSE, 검증 NMSE
- **로깅**: Weights & Biases 연동
- **Early Stopping**: 선택적 사용 가능

## 🔄 전이학습 전략

### 단계별 접근
1. **기본 모델**: RMa 환경에서 사전 훈련
2. **LoRA 적용**: 핵심 attention/FFN 레이어에 적용
3. **InF 전이**: InF 데이터로 LoRA 파라미터만 학습
4. **모델 병합**: 최종 배포용 통합 모델 생성

### 저장 방식
- **중간 저장**: LoRA 가중치만 별도 저장
- **최종 저장**: 병합된 모델을 engine.py 호환 형태로 저장

## 🐳 배포 환경

### Docker 설정
```bash
# 사전 구성된 환경
docker pull joowonoil/channel-estimation-env:latest
```

### 핵심 라이브러리
- PyTorch 2.4.1 + CUDA 12.1
- transformers, peft (LoRA)
- tensorrt, onnx (최적화)
- wandb (실험 관리)

## ⚡ 추론 최적화
- **TensorRT 변환**: tensorrt_conversion_v4.py
- **ONNX 지원**: 크로스 플랫폼 배포
- **엔진 파일**: saved_model/에서 .engine 파일 관리

## 🎛️ 실행 명령어

### 기본 훈련
```bash
python Transfer_v4.py
```

### TensorRT 변환
```bash
python tensorrt_conversion_v4.py
```

### 추론 테스트
```bash
# DNN_channel_estimation_inference/ 디렉토리에서
python compare.py
```

## 📝 주의사항

### 모델 저장 관련
- LoRA 모델 저장 후 requires_grad 상태 변화 주의
- engine.py 호환성을 위한 state_dict 변환 필요
- Early stopping 시 병합된 모델 자동 로드

### 데이터 형태
- 복소수 → 실수부/허수부 분리 처리
- DMRS 인덱싱 방식 일관성 유지
- 채널 정규화/역정규화 순서 주의

## 🔍 디버깅 포인트
- 학습 가능한 파라미터 수 확인 (LoRA 적용 후)
- 그래디언트 흐름 이상 감지
- 채널 추정 정확도 시각화 (wandb 플롯)
- 메모리 사용량 모니터링

이 프로젝트는 **실용적인 5G 채널 추정**을 위한 **LoRA 전이학습 시스템**으로, 효율적인 파라미터 적응과 높은 성능을 동시에 달성하는 잘 설계된 아키텍처입니다.