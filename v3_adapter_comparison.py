"""
v3 Adapter Performance Comparison Tool

이 코드는 v3 Adapter 모델들의 성능을 비교 분석하는 도구입니다.

주요 기능:
1. Base v3 모델과 Adapter 전이학습 모델들의 성능 비교
2. InF/RMa 환경별 Adapter 전이학습 효과 비교
3. 파라미터 효율성 분석 (Adapter 파라미터 vs 전체 파라미터)
4. v3 아키텍처의 Adapter 방식 전이학습 효과 시각화

테스트 대상 모델:
- Base_v3: v3 베이스 모델 (engine_v3.py로 학습)
- InF_Adapter_v3: Adapter로 InF 전이학습된 모델
- RMa_Adapter_v3: Adapter로 RMa 전이학습된 모델

출력:
- v3_adapter_comparison.png: 모든 v3 모델 성능 비교 차트
- v3_adapter_efficiency.png: Adapter 파라미터 효율성 분석 차트
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator_v3 import Estimator_v3

class V3AdapterTester:
    def __init__(self, device='cuda:0'):
        self.device = device
        
    def load_models(self):
        """학습된 v3 모델들 로드"""
        models = {}
        saved_model_dir = Path(__file__).parent / 'saved_model'
        
        # Base v3 모델 로드 (engine_v3.py로 학습)
        base_path = saved_model_dir / 'Large_estimator_v3_base_final.pt'
        if base_path.exists():
            try:
                base_model = torch.load(base_path, map_location=self.device)
                base_model.eval()
                models['Base_v3'] = base_model
                print("[OK] Loaded Base_v3 (engine_v3 trained)")
            except Exception as e:
                print(f"[ERROR] Failed to load Base_v3: {e}")
        else:
            print("[WARNING] Large_estimator_v3_base_final.pt not found")
            print("[INFO] Run engine_v3.py first to train the base model")
            
        # InF Adapter 모델 로드
        inf_adapter_path = saved_model_dir / 'Large_estimator_v3_to_InF_adapter.pt'
        if inf_adapter_path.exists():
            try:
                # Adapter 모델은 state_dict로 저장되므로, 모델 구조를 만들고 로드
                from Transfer_v3_InF import TransferLearningEngine
                transfer_engine = TransferLearningEngine('config_transfer_v3_InF.yaml')
                transfer_engine.load_model()
                
                # 학습된 Adapter 가중치 로드
                adapter_state_dict = torch.load(inf_adapter_path, map_location=self.device)
                transfer_engine._estimator.load_state_dict(adapter_state_dict)
                transfer_engine._estimator.eval()
                
                models['InF_Adapter_v3'] = transfer_engine._estimator
                print("[OK] Loaded InF_Adapter_v3 (Transfer_v3_InF trained)")
                
                # Adapter 파라미터 수 확인
                adapter_params = sum(p.numel() for n, p in transfer_engine._estimator.named_parameters() 
                                   if p.requires_grad and 'adapter' in n)
                total_params = sum(p.numel() for p in transfer_engine._estimator.parameters())
                
                print(f"[INFO] InF Adapter parameters: {adapter_params:,}")
                print(f"[INFO] Total parameters: {total_params:,}")
                print(f"[INFO] Adapter ratio: {adapter_params/total_params:.1%}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load InF_Adapter_v3: {e}")
        else:
            print("[WARNING] Large_estimator_v3_to_InF_adapter.pt not found")
            print("[INFO] Run Transfer_v3_InF.py first to train the InF adapter model")
            
        # RMa Adapter 모델 로드
        rma_adapter_path = saved_model_dir / 'Large_estimator_v3_to_RMa_adapter.pt'
        if rma_adapter_path.exists():
            try:
                # Adapter 모델은 state_dict로 저장되므로, 모델 구조를 만들고 로드
                from Transfer_v3_RMa import TransferLearningEngine
                transfer_engine = TransferLearningEngine('config_transfer_v3_RMa.yaml')
                transfer_engine.load_model()
                
                # 학습된 Adapter 가중치 로드
                adapter_state_dict = torch.load(rma_adapter_path, map_location=self.device)
                transfer_engine._estimator.load_state_dict(adapter_state_dict)
                transfer_engine._estimator.eval()
                
                models['RMa_Adapter_v3'] = transfer_engine._estimator
                print("[OK] Loaded RMa_Adapter_v3 (Transfer_v3_RMa trained)")
                
                # Adapter 파라미터 수 확인
                adapter_params = sum(p.numel() for n, p in transfer_engine._estimator.named_parameters() 
                                   if p.requires_grad and 'adapter' in n)
                total_params = sum(p.numel() for p in transfer_engine._estimator.parameters())
                
                print(f"[INFO] RMa Adapter parameters: {adapter_params:,}")
                print(f"[INFO] Total parameters: {total_params:,}")
                print(f"[INFO] Adapter ratio: {adapter_params/total_params:.1%}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load RMa_Adapter_v3: {e}")
        else:
            print("[WARNING] Large_estimator_v3_to_RMa_adapter.pt not found")
            print("[INFO] Run Transfer_v3_RMa.py first to train the RMa adapter model")
            
        return models
    
    def load_test_data(self):
        """실제 데이터셋 생성하여 테스트 데이터 생성"""
        from dataset import get_dataset_and_dataloader
        datasets = {}
        
        # InF 테스트 데이터 생성
        inf_params = {
            'channel_type': ["InF_Los", "InF_Nlos"],
            'batch_size': 8,
            'noise_spectral_density': -174.0,
            'subcarrier_spacing': 120.0,
            'transmit_power': 30.0,
            'distance_range': [40.0, 60.0],  # InF 거리 범위
            'carrier_freq': 28.0,
            'mod_order': 64,
            'ref_conf_dict': {'dmrs': (0, 3072, 6)},
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
        
        try:
            inf_dataset, inf_dataloader = get_dataset_and_dataloader(params=inf_params)
            # 첫 번째 배치만 사용
            for data in inf_dataloader:
                rx_signal = data['ref_comp_rx_signal']
                true_channel = data['ch_freq']
                
                # 복소수를 실수/허수 분리
                rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
                true_channel = np.stack((np.real(true_channel), np.imag(true_channel)), axis=-1)
                
                datasets['InF_50m'] = {
                    'input': torch.tensor(rx_signal, dtype=torch.float32).to(self.device),
                    'true': torch.tensor(true_channel, dtype=torch.float32).to(self.device)
                }
                print(f"[OK] Generated test dataset: InF_50m")
                break
        except Exception as e:
            print(f"[ERROR] Failed to generate InF test data: {e}")
            
        # RMa 테스트 데이터 생성  
        rma_params = {
            'channel_type': ["RMa_Los"],
            'batch_size': 8,
            'noise_spectral_density': -174.0,
            'subcarrier_spacing': 120.0,
            'transmit_power': 30.0,
            'distance_range': [300.0, 500.0],  # RMa 거리 범위
            'carrier_freq': 28.0,
            'mod_order': 64,
            'ref_conf_dict': {'dmrs': (0, 3072, 6)},
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
        
        try:
            rma_dataset, rma_dataloader = get_dataset_and_dataloader(params=rma_params)
            # 첫 번째 배치만 사용
            for data in rma_dataloader:
                rx_signal = data['ref_comp_rx_signal']
                true_channel = data['ch_freq']
                
                # 복소수를 실수/허수 분리
                rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
                true_channel = np.stack((np.real(true_channel), np.imag(true_channel)), axis=-1)
                
                datasets['RMa_300m'] = {
                    'input': torch.tensor(rx_signal, dtype=torch.float32).to(self.device),
                    'true': torch.tensor(true_channel, dtype=torch.float32).to(self.device)
                }
                print(f"[OK] Generated test dataset: RMa_300m")
                break
        except Exception as e:
            print(f"[ERROR] Failed to generate RMa test data: {e}")
                
        return datasets
    
    def evaluate_model(self, model, test_data):
        """모델 성능 평가 (NMSE 계산)"""
        with torch.no_grad():
            input_signal = test_data['input']
            true_channel = test_data['true']
            
            # 모델 추론
            estimated_channel, _ = model(input_signal)
            
            # NMSE 계산
            mse = torch.mean(torch.square(true_channel - estimated_channel))
            var = torch.mean(torch.square(true_channel))
            nmse = mse / var
            nmse_db = 10 * torch.log10(nmse)
            
            return nmse_db.item()
    
    def run_comparison(self):
        """v3 모델들 성능 비교 실행"""
        print("=" * 60)
        print("v3 Adapter Performance Comparison Started")
        print("=" * 60)
        
        # 모델 로드
        print("\nLoading v3 models...")
        models = self.load_models()
        
        if not models:
            print("[ERROR] No models loaded. Please train models first.")
            return
        
        # 테스트 데이터 로드
        print("\nLoading test datasets...")
        datasets = self.load_test_data()
        
        if not datasets:
            print("[ERROR] No test data found. Please prepare test data.")
            return
        
        # 성능 평가
        print("\nEvaluating model performance...")
        results = {}
        
        for model_name, model in models.items():
            results[model_name] = {}
            print(f"\nTesting {model_name}:")
            
            for dataset_name, test_data in datasets.items():
                try:
                    nmse_db = self.evaluate_model(model, test_data)
                    results[model_name][dataset_name] = nmse_db
                    print(f"  {dataset_name}: {nmse_db:.2f} dB")
                except Exception as e:
                    print(f"  {dataset_name}: ERROR - {e}")
                    results[model_name][dataset_name] = float('inf')
        
        # 결과 시각화
        print("\nGenerating comparison charts...")
        self.plot_results(results)
        
        # 요약 출력
        print("\n" + "=" * 60)
        print("📋 v3 Adapter Performance Summary")
        print("=" * 60)
        for model_name, model_results in results.items():
            print(f"\n{model_name}:")
            for dataset_name, nmse_db in model_results.items():
                if nmse_db != float('inf'):
                    print(f"  {dataset_name}: {nmse_db:.2f} dB")
                else:
                    print(f"  {dataset_name}: ERROR")
        
        print("\nv3 Adapter comparison completed!")
        return results
    
    def plot_results(self, results):
        """결과 시각화"""
        # 1. 성능 비교 차트
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = list(results.keys())
        dataset_names = ['InF_50m', 'RMa_300m']
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # InF 성능
        inf_scores = [results[model].get('InF_50m', float('inf')) for model in model_names]
        inf_scores = [score if score != float('inf') else 0 for score in inf_scores]
        
        # RMa 성능
        rma_scores = [results[model].get('RMa_300m', float('inf')) for model in model_names]
        rma_scores = [score if score != float('inf') else 0 for score in rma_scores]
        
        # 막대 그래프
        ax1.bar(x - width/2, inf_scores, width, label='InF Environment', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, rma_scores, width, label='RMa Environment', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('NMSE (dB)')
        ax1.set_title('v3 Adapter Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 개선도 비교 (베이스 모델 대비)
        if 'Base_v3' in results:
            base_inf = results['Base_v3'].get('InF_50m', 0)
            base_rma = results['Base_v3'].get('RMa_300m', 0)
            
            inf_improvements = []
            rma_improvements = []
            transfer_models = []
            
            for model_name in model_names:
                if 'Adapter' in model_name:
                    transfer_models.append(model_name.replace('_v3', ''))
                    inf_improvements.append(base_inf - results[model_name].get('InF_50m', 0))
                    rma_improvements.append(base_rma - results[model_name].get('RMa_300m', 0))
            
            if transfer_models:
                x2 = np.arange(len(transfer_models))
                ax2.bar(x2 - width/2, inf_improvements, width, label='InF Improvement', alpha=0.8, color='green')
                ax2.bar(x2 + width/2, rma_improvements, width, label='RMa Improvement', alpha=0.8, color='orange')
                
                ax2.set_xlabel('Transfer Learning Models')
                ax2.set_ylabel('NMSE Improvement (dB)')
                ax2.set_title('Adapter Transfer Learning Improvement\n(Compared to Base v3)')
                ax2.set_xticks(x2)
                ax2.set_xticklabels(transfer_models, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('v3_adapter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("[OK] Saved v3_adapter_comparison.png")

def main():
    tester = V3AdapterTester()
    results = tester.run_comparison()
    return results

if __name__ == "__main__":
    main()