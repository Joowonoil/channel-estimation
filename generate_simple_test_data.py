import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from dataset import get_dataset_and_dataloader

def generate_simple_test_data():
    """간단한 테스트 데이터 생성 (TensorRT, 제어신호 등 제거)"""
    
    # InF 테스트 데이터 설정
    inf_params = {
        'channel_type': ["InF_Los_10000"],  # Los만 사용
        'batch_size': 100,
        'noise_spectral_density': -174.0,
        'subcarrier_spacing': 120.0,
        'transmit_power': 30.0,
        'distance_range': [50.0, 50.0],  # 50m 고정
        'carrier_freq': 28.0,
        'mod_order': 64,
        'ref_conf_dict': {'dmrs': [0, 3072, 6]},
        'fft_size': 4096,
        'num_guard_subcarriers': 1024,
        'num_symbol': 14,
        'cp_length': 590,
        'max_random_tap_delay_cp_proportion': 0.2,
        'rnd_seed': 0,
        'num_workers': 0,
        'is_phase_noise': False,
        'is_channel': True,
        'is_noise': True
    }
    
    # RMa 테스트 데이터 설정
    rma_params = inf_params.copy()
    rma_params['channel_type'] = ["RMa_Los_10000"]  # Los만 사용
    rma_params['distance_range'] = [300.0, 300.0]  # 300m 고정
    
    # 데이터 생성 및 저장
    for dataset_name, params in [('InF_50m', inf_params), ('RMa_300m', rma_params)]:
        print(f"Generating {dataset_name} test data...")
        
        # 데이터셋 및 데이터로더 생성
        dataset, dataloader = get_dataset_and_dataloader(params)
        
        # 한 배치만 가져오기
        for batch_idx, (rx_signal, ch_true, *_) in enumerate(dataloader):
            if batch_idx == 0:  # 첫 번째 배치만
                # NumPy 배열로 변환
                rx_input = rx_signal.numpy()  # (batch, 14, 3072, 2)
                ch_true_np = ch_true.numpy()  # (batch, 3072) - 복소수
                
                # 저장 경로 설정
                test_data_dir = Path(__file__).parent / 'simple_test_data'
                test_data_dir.mkdir(exist_ok=True)
                
                # 파일 저장
                np.save(test_data_dir / f'{dataset_name}_input.npy', rx_input)
                np.save(test_data_dir / f'{dataset_name}_true.npy', ch_true_np)
                
                print(f"  Saved: {rx_input.shape} input samples")
                print(f"  Saved: {ch_true_np.shape} ground truth samples")
                break
    
    print("Test data generation complete!")

if __name__ == "__main__":
    generate_simple_test_data()