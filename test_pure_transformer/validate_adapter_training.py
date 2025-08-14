"""
Adapter 학습 검증 스크립트

실제로 Adapter가 제대로 학습되었는지, 그리고 추론 시 제대로 작동하는지 검증
"""

import torch
import numpy as np
from pathlib import Path

def validate_adapter_parameters():
    """저장된 Adapter 모델의 파라미터를 분석"""
    
    print("=" * 60)
    print("Adapter Parameter Validation")
    print("=" * 60)
    
    saved_model_dir = Path("saved_model")
    
    # 1. InF Adapter 모델 분석
    inf_adapter_path = saved_model_dir / 'Large_estimator_v3_to_InF_adapter.pt'
    if inf_adapter_path.exists():
        print("\n1. InF Adapter Analysis:")
        adapter_state_dict = torch.load(inf_adapter_path, map_location='cpu')
        
        adapter_params = {}
        for key, param in adapter_state_dict.items():
            if 'adapter' in key:
                adapter_params[key] = param
        
        print(f"Found {len(adapter_params)} adapter parameters")
        
        # Adapter 파라미터 통계
        for key, param in list(adapter_params.items())[:4]:  # 첫 4개만 출력
            mean_val = param.mean().item()
            std_val = param.std().item()
            max_val = param.max().item()
            min_val = param.min().item()
            
            print(f"  {key}:")
            print(f"    Shape: {param.shape}")
            print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
            
            # 파라미터가 0에 가까운지 확인
            zero_ratio = (torch.abs(param) < 1e-6).float().mean().item()
            print(f"    Near-zero ratio: {zero_ratio:.2%}")
    
    # 2. RMa Adapter 모델 분석
    rma_adapter_path = saved_model_dir / 'Large_estimator_v3_to_RMa_adapter.pt'
    if rma_adapter_path.exists():
        print("\n2. RMa Adapter Analysis:")
        adapter_state_dict = torch.load(rma_adapter_path, map_location='cpu')
        
        adapter_params = {}
        for key, param in adapter_state_dict.items():
            if 'adapter' in key:
                adapter_params[key] = param
        
        print(f"Found {len(adapter_params)} adapter parameters")
        
        # 첫 번째 Adapter 레이어만 분석
        first_adapter_key = 'ch_tf._layers.0.adapter1.fc1.weight'
        if first_adapter_key in adapter_params:
            param = adapter_params[first_adapter_key]
            mean_val = param.mean().item()
            std_val = param.std().item()
            
            print(f"  {first_adapter_key}:")
            print(f"    Shape: {param.shape}")
            print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            
            zero_ratio = (torch.abs(param) < 1e-6).float().mean().item()
            print(f"    Near-zero ratio: {zero_ratio:.2%}")
    
    # 3. 베이스 모델과 비교
    print("\n3. Base Model Comparison:")
    base_path = saved_model_dir / 'Large_estimator_v3_base_final_final.pt'
    if base_path.exists():
        base_model = torch.load(base_path, map_location='cpu')
        base_state_dict = base_model.state_dict()
        
        # 같은 Transformer 레이어 파라미터 비교
        base_key = 'ch_tf._layers.0.multi_head_attention.query_projection.weight'
        if base_key in base_state_dict:
            base_param = base_state_dict[base_key]
            print(f"  {base_key}:")
            print(f"    Shape: {base_param.shape}")
            print(f"    Mean: {base_param.mean().item():.6f}")
            print(f"    Std: {base_param.std().item():.6f}")

def test_adapter_forward_pass():
    """Adapter가 포함된 모델과 없는 모델의 forward pass 비교"""
    
    print("\n" + "=" * 60)
    print("Adapter Forward Pass Test")
    print("=" * 60)
    
    try:
        from model.estimator_v3 import Estimator_v3
        from Transfer_v3_InF import TransferLearningEngine
        
        # 1. 순수 베이스 모델 (Adapter 없음)
        print("\n1. Loading Pure Base Model...")
        base_model = Estimator_v3('config_transfer_v3_InF.yaml')
        pure_state_dict = torch.load('saved_model/Large_estimator_v3_base_pure_transformer.pt', map_location='cpu')
        base_model.load_state_dict(pure_state_dict, strict=False)
        base_model.eval()
        
        # 2. InF Adapter 모델
        print("2. Loading InF Adapter Model...")
        transfer_engine = TransferLearningEngine('config_transfer_v3_InF.yaml')
        transfer_engine.load_model()
        adapter_state_dict = torch.load('saved_model/Large_estimator_v3_to_InF_adapter.pt', map_location='cpu')
        transfer_engine._estimator.load_state_dict(adapter_state_dict)
        transfer_engine._estimator.eval()
        
        # 3. 동일한 입력으로 테스트
        print("3. Forward Pass Comparison...")
        
        # 랜덤 입력 생성 (실제 데이터 형태)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3072, 2)  # [B, L, C]
        
        with torch.no_grad():
            # 베이스 모델 출력
            base_output, _ = base_model(input_tensor)
            
            # Adapter 모델 출력
            adapter_output, _ = transfer_engine._estimator(input_tensor)
        
        # 출력 차이 분석
        output_diff = torch.abs(base_output - adapter_output)
        mean_diff = output_diff.mean().item()
        max_diff = output_diff.max().item()
        
        print(f"  Output shape: {base_output.shape}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print(f"  Max absolute difference: {max_diff:.6f}")
        
        if mean_diff < 1e-6:
            print("  ⚠️  WARNING: Outputs are nearly identical!")
            print("     This suggests Adapter may not be working properly.")
        else:
            print("  ✅ Outputs differ significantly - Adapter is working!")
            
        # 4. Adapter 활성화 확인
        print("\n4. Adapter Activation Check...")
        
        # 첫 번째 레이어의 Adapter 출력 확인
        adapter_model = transfer_engine._estimator
        if hasattr(adapter_model.ch_tf._layers[0], 'adapter1'):
            adapter1 = adapter_model.ch_tf._layers[0].adapter1
            
            # Adapter에 입력 전달 (간단한 테스트)
            test_input = torch.randn(1, 128)  # d_model 크기
            with torch.no_grad():
                adapter_output = adapter1(test_input)
            
            adapter_activation = torch.abs(adapter_output).mean().item()
            print(f"  Adapter1 activation magnitude: {adapter_activation:.6f}")
            
            if adapter_activation < 1e-6:
                print("  ⚠️  WARNING: Adapter activation is near zero!")
            else:
                print("  ✅ Adapter is actively contributing!")
        
    except Exception as e:
        print(f"❌ Error during forward pass test: {e}")

def main():
    """메인 검증 함수"""
    validate_adapter_parameters()
    test_adapter_forward_pass()
    
    print("\n" + "=" * 60)
    print("Validation Summary:")
    print("- Check if adapter parameters are non-zero")
    print("- Compare forward pass outputs")
    print("- Verify adapter activation levels")
    print("=" * 60)

if __name__ == "__main__":
    main()