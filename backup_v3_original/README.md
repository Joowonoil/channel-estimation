# Backup v3 Original - Complete Original Implementation

이 폴더는 순수 베이스 모델 워크플로우 개발 전의 완전한 원본 v3 구현을 백업한 것입니다.

## 백업된 파일들

### Python Scripts
- `engine_v3.py` - v3 베이스 모델 훈련 엔진 (Adapter 포함)
- `Transfer_v3_InF.py` - InF 환경 Adapter 전이학습
- `Transfer_v3_RMa.py` - RMa 환경 Adapter 전이학습
- `extract_pure_transformer_v3.py` - 오염된 베이스 모델에서 순수 Transformer 추출
- `validate_adapter_training.py` - Adapter 학습 검증 스크립트
- `v3_adapter_comparison.py` - v3 모델들 성능 비교

### Model Files
- `model/estimator_v3.py` - v3 Estimator 클래스 (원본)
- `model/transformer_v3.py` - v3 Transformer 클래스 (원본)
- `model/adapter.py` - Adapter 모듈

### Config Files
- `config/config_v3.yaml` - v3 베이스 모델 훈련 설정
- `config/config_transfer_v3_InF.yaml` - InF Adapter 전이학습 설정  
- `config/config_transfer_v3_RMa.yaml` - RMa Adapter 전이학습 설정

### Saved Models
- `saved_model/Large_estimator_v3_base_*.pt` - 원본 v3 베이스 모델들
- `saved_model/Large_estimator_v3_to_*_adapter.pt` - 원본 Adapter 전이학습 모델들
- `saved_model/Large_estimator_v3_base_pure_transformer.pt` - 추출된 순수 Transformer

### Result Images
- `adapter_v3_comparison.png` - v3 Adapter 비교 결과
- `v3_adapter_comparison.png` - v3 모델 성능 비교 차트
- `v3_adapter_improvement.png` - v3 Adapter 개선도 차트

## 원본 구현의 문제점

1. **오염된 베이스 모델**: 베이스 모델 훈련 시 Adapter가 포함되어 저장됨
2. **추출 과정 필요**: 순수 Transformer를 얻기 위해 별도 추출 과정 필요
3. **파라미터 제약**: bottleneck_dim 변경 시 베이스 모델 재학습 필요
4. **비교 부정확성**: 성능 비교 시 베이스 모델에도 랜덤 Adapter가 포함됨

## 개선된 구현

메인 디렉토리의 새로운 구현에서는:
- Adapter 비활성화 옵션 추가
- 순수 베이스 모델 직접 학습
- 깔끔한 워크플로우 구현

## 사용법 (백업 코드)

```bash
# 베이스 모델 훈련 (Adapter 포함)
python engine_v3.py

# 순수 Transformer 추출
python extract_pure_transformer_v3.py

# 전이학습
python Transfer_v3_InF.py
python Transfer_v3_RMa.py

# 성능 비교
python v3_adapter_comparison.py

# 검증
python validate_adapter_training.py
```