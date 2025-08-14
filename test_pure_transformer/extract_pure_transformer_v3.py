"""
v3 베이스 모델에서 Adapter 파라미터 제거하여 순수 Transformer 추출

베이스 모델에 실수로 Adapter가 포함되어 학습된 경우,
순수 Transformer 파라미터만 추출하여 새로운 베이스 모델 생성
"""

import torch
from pathlib import Path

def extract_pure_transformer():
    """베이스 모델에서 Adapter 파라미터를 제거하여 순수 Transformer 추출"""
    
    base_model_path = Path("saved_model/Large_estimator_v3_base_final_iter_10000.pt")
    output_path = Path("saved_model/Large_estimator_v3_base_pure_transformer_test.pt")
    
    print("=" * 60)
    print("v3 Pure Transformer Extraction")
    print("=" * 60)
    print(f"Input: {base_model_path}")
    print(f"Output: {output_path}")
    
    # 베이스 모델 로드
    try:
        model_obj = torch.load(base_model_path, map_location='cpu')
        print(f"[OK] Loaded base model from {base_model_path}")
        
        # 모델 객체에서 state_dict 추출
        if hasattr(model_obj, 'state_dict'):
            model_state_dict = model_obj.state_dict()
            print("[INFO] Extracted state_dict from model object")
        else:
            # 이미 state_dict인 경우
            model_state_dict = model_obj
            print("[INFO] Loaded state_dict directly")
            
    except Exception as e:
        print(f"[ERROR] Failed to load base model: {e}")
        return
    
    # 원본 파라미터 수 계산
    original_params = sum(param.numel() for param in model_state_dict.values())
    print(f"Original model parameters: {original_params:,}")
    
    # Adapter 관련 키 찾기
    adapter_keys = []
    transformer_keys = []
    
    for key in model_state_dict.keys():
        if 'adapter' in key:
            adapter_keys.append(key)
        else:
            transformer_keys.append(key)
    
    print(f"\nFound {len(adapter_keys)} adapter parameters:")
    for key in adapter_keys:
        print(f"  - {key}: {model_state_dict[key].shape}")
    
    # Adapter 파라미터 제거
    pure_transformer_dict = {}
    adapter_param_count = 0
    
    for key in transformer_keys:
        pure_transformer_dict[key] = model_state_dict[key]
    
    for key in adapter_keys:
        adapter_param_count += model_state_dict[key].numel()
    
    # 순수 Transformer 파라미터 수 계산
    pure_params = sum(param.numel() for param in pure_transformer_dict.values())
    
    print(f"\nParameter extraction summary:")
    print(f"  Original parameters: {original_params:,}")
    print(f"  Adapter parameters: {adapter_param_count:,}")
    print(f"  Pure Transformer parameters: {pure_params:,}")
    print(f"  Removed: {original_params - pure_params:,} parameters")
    print(f"  Removal ratio: {(adapter_param_count/original_params)*100:.1f}%")
    
    # 순수 Transformer 모델 저장
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pure_transformer_dict, output_path)
        print(f"\n[OK] Pure Transformer saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save pure transformer: {e}")
        return
    
    # 검증: 저장된 모델 다시 로드해서 확인
    try:
        verification_dict = torch.load(output_path, map_location='cpu')
        verification_params = sum(param.numel() for param in verification_dict.values())
        
        if verification_params == pure_params:
            print(f"[OK] Verification passed: {verification_params:,} parameters")
        else:
            print(f"[ERROR] Verification failed: {verification_params:,} != {pure_params:,}")
            
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
    
    print("\n" + "=" * 60)
    print("Pure Transformer extraction completed!")
    print("=" * 60)

if __name__ == "__main__":
    extract_pure_transformer()