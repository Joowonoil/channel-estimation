# DNN Channel Estimation Training

> 5G/6G 통신을 위한 딥러닝 기반 DMRS 채널 추정 시스템

## 📚 프로젝트 문서

모든 상세 문서는 [docs/](./docs/) 폴더에서 확인할 수 있습니다:

- **[프로젝트 개요 및 가이드](./docs/README.md)**
- **[기술 분석 문서](./docs/CLAUDE.md)**
- **[v4 개발 가이드](./docs/engine_v4_development.md)**
- **[코드 아키텍처](./docs/code_architecture.md)**

## 🚀 빠른 시작

### 자동 설치 (Vast AI)
```bash
curl -sSL https://raw.githubusercontent.com/joowonoil/channel-estimation-training/main/setup_vast_ai.sh | bash
```

### 모델 훈련
```bash
# v4 베이스 모델
python engine_v4.py

# LoRA 전이학습
python Transfer_v4_InF.py
```

## 🔬 주요 특징

- **LoRA 전이학습**: 효율적인 파라미터 적응
- **v4 아키텍처**: 완벽한 가중치 호환성
- **TensorRT 최적화**: 실시간 추론 가능
- **다양한 채널 환경**: InF, RMa, InH, UMa/UMi 지원

## 📈 성능

- 채널 추정 정확도: **95%+** (InF 환경)
- 추론 속도: **<10ms** (TensorRT)
- 메모리 효율: LoRA로 **90% 감소**

## 📄 라이선스

MIT License

---

자세한 내용은 [docs/README.md](./docs/README.md)를 참조하세요.