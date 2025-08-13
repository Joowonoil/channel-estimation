import torch
from model.estimator_v4 import Estimator_v4
from peft import LoraConfig, get_peft_model
import yaml
from pathlib import Path

def check_lora_parameters():
    print("="*60)
    print("LoRA Parameter Analysis for V4 Model")
    print("="*60)
    
    # 설정 파일 로드
    conf_file = 'config_transfer_v4_RMa.yaml'
    conf_path = Path(__file__).parent / 'config' / conf_file
    
    with open(conf_path, encoding='utf-8') as f:
        conf = yaml.safe_load(f)
    
    print(f"Config file: {conf_file}")
    print(f"LoRA settings:")
    peft_config = conf['ch_estimation']['peft']
    print(f"  - r (rank): {peft_config['r']}")
    print(f"  - lora_alpha: {peft_config['lora_alpha']}")
    print(f"  - target_modules: {peft_config['target_modules']}")
    print(f"  - lora_dropout: {peft_config['lora_dropout']}")
    print()
    
    # 베이스 모델 로드
    print("Loading base Estimator_v4 model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    estimator = Estimator_v4(conf_file).to(device)
    
    # 베이스 모델의 파라미터 수 확인
    total_params = sum(p.numel() for p in estimator.parameters())
    trainable_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    
    print(f"Base model total parameters: {total_params:,}")
    print(f"Base model trainable parameters: {trainable_params:,}")
    print()
    
    # LoRA 적용
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=peft_config['r'],
        lora_alpha=peft_config['lora_alpha'],
        target_modules=peft_config['target_modules'],
        lora_dropout=peft_config['lora_dropout'],
        bias="none"
    )
    
    peft_model = get_peft_model(estimator, lora_config)
    
    # LoRA 적용 후 파라미터 수 확인
    peft_model.print_trainable_parameters()
    
    total_params_peft = sum(p.numel() for p in peft_model.parameters())
    trainable_params_peft = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    print(f"\nAfter LoRA:")
    print(f"Total parameters: {total_params_peft:,}")
    print(f"Trainable parameters: {trainable_params_peft:,}")
    print(f"Trainable ratio: {trainable_params_peft/total_params_peft*100:.2f}%")
    
    # LoRA 파라미터 상세 분석
    print("\n" + "="*60)
    print("Detailed LoRA Parameters Analysis")
    print("="*60)
    
    lora_params = 0
    for name, param in peft_model.named_parameters():
        if param.requires_grad and 'lora' in name.lower():
            print(f"{name}: {param.numel():,} parameters")
            lora_params += param.numel()
    
    print(f"\nTotal LoRA parameters: {lora_params:,}")
    
    # 이론적 LoRA 파라미터 수 계산
    print("\n" + "="*60)
    print("Theoretical LoRA Parameter Calculation")
    print("="*60)
    
    # Transformer 설정
    d_model = conf['ch_estimation']['transformer']['d_model']  # 128
    num_layers = conf['ch_estimation']['transformer']['num_layers']  # 4
    dim_feedforward = conf['ch_estimation']['transformer']['dim_feedforward']  # 1024
    r = peft_config['r']  # 8
    
    print(f"Model dimensions:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_layers: {num_layers}")
    print(f"  - dim_feedforward: {dim_feedforward}")
    print(f"  - LoRA rank (r): {r}")
    print()
    
    # 각 레이어별 LoRA 파라미터 계산
    # MHA: q_proj, k_proj, v_proj, out_proj
    # FFN: ffnn_linear1, ffnn_linear2
    
    mha_params_per_layer = 4 * (d_model * r + r * d_model)  # 4개 projection * (down + up)
    ffn_params_per_layer = (d_model * r + r * dim_feedforward) + (dim_feedforward * r + r * d_model)
    
    print(f"Per layer LoRA parameters:")
    print(f"  - MHA (4 projections): {mha_params_per_layer:,}")
    print(f"  - FFN (2 linear layers): {ffn_params_per_layer:,}")
    print(f"  - Total per layer: {mha_params_per_layer + ffn_params_per_layer:,}")
    print()
    
    total_theoretical = num_layers * (mha_params_per_layer + ffn_params_per_layer)
    print(f"Total theoretical LoRA parameters: {total_theoretical:,}")
    print(f"Actual vs Theoretical: {lora_params:,} vs {total_theoretical:,}")
    
    # LoRA 효율성 분석
    print("\n" + "="*60)
    print("LoRA Efficiency Analysis")
    print("="*60)
    
    reduction_ratio = trainable_params_peft / total_params
    print(f"Parameter reduction: {(1-reduction_ratio)*100:.1f}%")
    print(f"Memory efficiency: {reduction_ratio:.4f}x smaller")
    
    if trainable_params_peft > 50000:
        print("\n⚠️  WARNING: LoRA trainable parameters seem quite large!")
        print("   Consider reducing the rank (r) or number of target modules")
        print("   Typical LoRA should have <1% trainable parameters")
    else:
        print("\n✅ LoRA configuration looks reasonable")
    
    return {
        'total_params': total_params_peft,
        'trainable_params': trainable_params_peft,
        'lora_params': lora_params,
        'reduction_ratio': reduction_ratio
    }

if __name__ == "__main__":
    results = check_lora_parameters()