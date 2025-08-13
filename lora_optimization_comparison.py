"""
LoRA Optimization Performance Comparison Tool

이 코드는 LoRA (Low-Rank Adaptation) 최적화 전후의 성능을 비교 분석하는 도구입니다.

주요 기능:
1. Base v4 모델과 LoRA 전이학습 모델들의 성능 비교
2. 최적화된 LoRA (rank=4, 3개 모듈) vs 기존 LoRA (rank=8, 6개 모듈) 성능 비교
3. 파라미터 효율성 분석 (114,688개 → 26,624개, 76.8% 감소)
4. InF/RMa 환경별 전이학습 효과 시각화

테스트 대상 모델:
- Base_v4_1k: 베이스 모델 (1k iteration 학습)
- InF_Transfer_Optimized: 최적화된 LoRA로 InF 전이학습된 모델
- RMa_Transfer_Optimized: 최적화된 LoRA로 RMa 전이학습된 모델
- InF_Transfer_Old: 기존 LoRA로 InF 전이학습된 모델 (비교용)
- RMa_Transfer_Old: 기존 LoRA로 RMa 전이학습된 모델 (비교용)

출력:
- optimized_lora_comparison.png: 모든 모델 성능 비교 차트
- lora_optimization_comparison.png: 최적화 전후 직접 비교 차트
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator_v4 import Estimator_v4

class SimpleModelTester:
    def __init__(self, device='cuda:0'):
        self.device = device
        
    def load_models(self):
        """학습된 모델들 로드"""
        models = {}
        saved_model_dir = Path(__file__).parent / 'saved_model'
        
        # Base v4 모델 로드 (Estimator_v4 구조)
        base_path = saved_model_dir / 'Large_estimator_v4_base_final.pt'
        if base_path.exists():
            try:
                # v4 베이스 모델도 전체 모델 객체로 저장되었으므로 직접 로드
                base_model = torch.load(base_path, map_location=self.device)
                base_model.eval()
                models['Base_v4_1k'] = base_model
                print("[OK] Loaded Base_v4_1k (Estimator_v4)")
            except Exception as e:
                print(f"[ERROR] Failed to load Base_v4_1k: {e}")
        else:
            print("[WARNING] Large_estimator_v4_base_final.pt not found")
            
        # InF Transfer 모델 로드 (v4 구조로 병합되어 저장됨) - 최적화된 버전
        transfer_inf_path = saved_model_dir / 'Large_estimator_v4_to_InF_optimized_v4.pt'
        if transfer_inf_path.exists():
            try:
                # InF Transfer 모델의 v4 구조 버전 로드 (최적화됨)
                transfer_inf_model = torch.load(transfer_inf_path, map_location=self.device)
                transfer_inf_model.eval()
                models['InF_Transfer_Optimized'] = transfer_inf_model
                print("[OK] Loaded InF_Transfer_Optimized (v4 structure)")
            except Exception as e:
                print(f"[ERROR] Failed to load InF_Transfer_Optimized: {e}")
        else:
            print("[WARNING] Large_estimator_v4_to_InF_optimized_v4.pt not found")
            
        # RMa Transfer 모델 로드 (v4 구조로 병합되어 저장됨) - 최적화된 버전
        transfer_rma_path = saved_model_dir / 'Large_estimator_v4_to_RMa_optimized_v4.pt'
        if transfer_rma_path.exists():
            try:
                # RMa Transfer 모델의 v4 구조 버전 로드 (최적화됨)
                transfer_rma_model = torch.load(transfer_rma_path, map_location=self.device)
                transfer_rma_model.eval()
                models['RMa_Transfer_Optimized'] = transfer_rma_model
                print("[OK] Loaded RMa_Transfer_Optimized (v4 structure)")
            except Exception as e:
                print(f"[ERROR] Failed to load RMa_Transfer_Optimized: {e}")
        else:
            print("[WARNING] Large_estimator_v4_to_RMa_optimized_v4.pt not found")
            
        # 기존 모델들도 로드 (비교용)
        old_inf_path = saved_model_dir / 'Large_estimator_v4_to_InF_test_v4.pt'
        if old_inf_path.exists():
            try:
                old_inf_model = torch.load(old_inf_path, map_location=self.device)
                old_inf_model.eval()
                models['InF_Transfer_Old'] = old_inf_model
                print("[OK] Loaded InF_Transfer_Old (for comparison)")
            except Exception as e:
                print(f"[ERROR] Failed to load InF_Transfer_Old: {e}")
        else:
            print("[WARNING] Large_estimator_v4_to_InF_test_v4.pt not found")
            
        old_rma_path = saved_model_dir / 'Large_estimator_v4_to_RMa_test_v4.pt'
        if old_rma_path.exists():
            try:
                old_rma_model = torch.load(old_rma_path, map_location=self.device)
                old_rma_model.eval()
                models['RMa_Transfer_Old'] = old_rma_model
                print("[OK] Loaded RMa_Transfer_Old (for comparison)")
            except Exception as e:
                print(f"[ERROR] Failed to load RMa_Transfer_Old: {e}")
        else:
            print("[WARNING] Large_estimator_v4_to_RMa_test_v4.pt not found")
        
        return models
    
    def load_test_data(self):
        """간단한 테스트 데이터 로드"""
        test_data_dir = Path(__file__).parent / 'simple_test_data'
        datasets = {}
        
        for dataset_name in ['InF_50m', 'RMa_300m']:
            input_path = test_data_dir / f'{dataset_name}_input.npy'
            true_path = test_data_dir / f'{dataset_name}_true.npy'
            
            if input_path.exists() and true_path.exists():
                rx_input = np.load(input_path)
                ch_true = np.load(true_path)
                datasets[dataset_name] = (rx_input, ch_true)
                print(f"[OK] Loaded {dataset_name}: input {rx_input.shape}, true {ch_true.shape}")
            else:
                print(f"[WARNING] Test data for {dataset_name} not found")
        
        return datasets
    
    def calculate_nmse(self, ch_est, ch_true):
        """NMSE 계산 (학습과 동일한 방식)"""
        # 복소수를 실수부/허수부로 분리
        ch_true = np.stack((np.real(ch_true), np.imag(ch_true)), axis=-1)
        
        # NMSE 계산
        ch_mse = np.sum(np.square(ch_true - ch_est), axis=(1, 2)) / ch_true.shape[-1]
        ch_var = np.sum(np.square(ch_true), axis=(1, 2)) / ch_true.shape[-1]
        ch_nmse = np.mean(ch_mse / ch_var)
        
        return ch_nmse
    
    def test_models(self):
        """모델 테스트 실행"""
        models = self.load_models()
        datasets = self.load_test_data()
        
        if not models or not datasets:
            print("Models or datasets not loaded properly!")
            return
        
        results = {}
        
        print("\n" + "="*60)
        print("Simple Model Testing Results")
        print("="*60)
        
        for dataset_name, (rx_input, ch_true) in datasets.items():
            print(f"\nTesting on {dataset_name}:")
            results[dataset_name] = {}
            
            # 입력 데이터를 텐서로 변환
            rx_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)
            
            for model_name, model in models.items():
                try:
                    with torch.no_grad():
                        # 모델 추론
                        ch_est, _ = model(rx_tensor)
                        ch_est_np = ch_est.cpu().numpy()
                        
                        # NMSE 계산
                        nmse = self.calculate_nmse(ch_est_np, ch_true)
                        nmse_db = 10 * np.log10(nmse)
                        
                        results[dataset_name][model_name] = nmse_db
                        
                        print(f"  {model_name:<15}: {nmse_db:.2f} dB")
                        
                except Exception as e:
                    print(f"  {model_name:<15}: ERROR - {e}")
                    results[dataset_name][model_name] = np.nan
        
        # 결과 요약
        self.print_summary(results)
        
        # 플롯 그리기
        self.plot_results(results)
        
        return results
    
    def print_summary(self, results):
        """결과 요약 출력"""
        print("\n" + "="*100)
        print("SUMMARY - LoRA Transfer Learning Performance Comparison")
        print("="*100)
        
        # 최적화된 모델 이름으로 수정
        print(f"{'Dataset':<15} {'Base_v4_1k':<18} {'InF_Optimized':<18} {'RMa_Optimized':<18} {'InF_Old':<12} {'RMa_Old':<12}")
        print("-" * 100)
        
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_1k', np.nan)
            inf_opt_nmse = results[dataset_name].get('InF_Transfer_Optimized', np.nan)
            rma_opt_nmse = results[dataset_name].get('RMa_Transfer_Optimized', np.nan)
            inf_old_nmse = results[dataset_name].get('InF_Transfer_Old', np.nan)
            rma_old_nmse = results[dataset_name].get('RMa_Transfer_Old', np.nan)
            
            # 포맷팅
            base_str = f"{base_nmse:.2f}" if not np.isnan(base_nmse) else "N/A"
            inf_opt_str = f"{inf_opt_nmse:.2f}" if not np.isnan(inf_opt_nmse) else "N/A"
            rma_opt_str = f"{rma_opt_nmse:.2f}" if not np.isnan(rma_opt_nmse) else "N/A"
            inf_old_str = f"{inf_old_nmse:.2f}" if not np.isnan(inf_old_nmse) else "N/A"
            rma_old_str = f"{rma_old_nmse:.2f}" if not np.isnan(rma_old_nmse) else "N/A"
            
            print(f"{dataset_name:<15} {base_str:<18} {inf_opt_str:<18} {rma_opt_str:<18} {inf_old_str:<12} {rma_old_str:<12}")
        
        print("="*100)
        
        # 개선량 분석
        print("\nPerformance Improvements (vs Base Model):")
        print("-" * 70)
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_1k', np.nan)
            inf_opt_nmse = results[dataset_name].get('InF_Transfer_Optimized', np.nan)
            rma_opt_nmse = results[dataset_name].get('RMa_Transfer_Optimized', np.nan)
            inf_old_nmse = results[dataset_name].get('InF_Transfer_Old', np.nan)
            rma_old_nmse = results[dataset_name].get('RMa_Transfer_Old', np.nan)
            
            print(f"\n{dataset_name}:")
            if not np.isnan(base_nmse):
                if not np.isnan(inf_opt_nmse):
                    inf_opt_improvement = base_nmse - inf_opt_nmse
                    print(f"  InF Optimized: {inf_opt_improvement:+.2f} dB")
                if not np.isnan(rma_opt_nmse):
                    rma_opt_improvement = base_nmse - rma_opt_nmse
                    print(f"  RMa Optimized: {rma_opt_improvement:+.2f} dB")
                if not np.isnan(inf_old_nmse):
                    inf_old_improvement = base_nmse - inf_old_nmse
                    print(f"  InF Old:       {inf_old_improvement:+.2f} dB")
                if not np.isnan(rma_old_nmse):
                    rma_old_improvement = base_nmse - rma_old_nmse
                    print(f"  RMa Old:       {rma_old_improvement:+.2f} dB")
        
        print("="*70)
    
    def plot_results(self, results):
        """결과 플롯 그리기"""
        if not results:
            print("No results to plot!")
            return
            
        # 첫 번째 플롯: 모든 모델 성능 비교
        plt.figure(figsize=(14, 8))
        
        # 데이터 준비 - 실제 로드된 모델 이름 사용
        datasets = list(results.keys())
        models = ['Base_v4_1k', 'InF_Transfer_Optimized', 'RMa_Transfer_Optimized', 'InF_Transfer_Old', 'RMa_Transfer_Old']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        x_positions = np.arange(len(datasets))
        bar_width = 0.15
        
        # 각 모델별로 바 그래프 그리기
        for i, model in enumerate(models):
            nmse_values = []
            for dataset in datasets:
                nmse = results[dataset].get(model, np.nan)
                nmse_values.append(nmse if not np.isnan(nmse) else 0)
            
            # 실제 값이 있는 경우에만 바 그리기
            if any(v != 0 for v in nmse_values):
                bars = plt.bar(x_positions + i * bar_width, nmse_values, 
                              bar_width, label=model.replace('_', ' '), color=colors[i], alpha=0.7)
                
                # 바 위에 값 표시
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height != 0:  # NaN이 아닌 경우만
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('LoRA Transfer Learning Performance Comparison (All Models)', fontsize=14)
        plt.xticks(x_positions + bar_width * 2, datasets)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 플롯 저장
        save_path = Path(__file__).parent / 'optimized_lora_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAll models comparison plot saved to: {save_path}")
        
        plt.show()
        
        # 두 번째 플롯: 최적화 전후 비교
        plt.figure(figsize=(12, 6))
        
        improvements_opt = []
        improvements_old = []
        dataset_labels = []
        
        for dataset in datasets:
            base_nmse = results[dataset].get('Base_v4_1k', np.nan)
            
            # InF 비교
            inf_opt_nmse = results[dataset].get('InF_Transfer_Optimized', np.nan)
            inf_old_nmse = results[dataset].get('InF_Transfer_Old', np.nan)
            
            if not np.isnan(base_nmse) and not np.isnan(inf_opt_nmse) and not np.isnan(inf_old_nmse):
                improvements_opt.append(base_nmse - inf_opt_nmse)
                improvements_old.append(base_nmse - inf_old_nmse)
                dataset_labels.append(f'{dataset}\n(InF)')
            
            # RMa 비교
            rma_opt_nmse = results[dataset].get('RMa_Transfer_Optimized', np.nan)
            rma_old_nmse = results[dataset].get('RMa_Transfer_Old', np.nan)
            
            if not np.isnan(base_nmse) and not np.isnan(rma_opt_nmse) and not np.isnan(rma_old_nmse):
                improvements_opt.append(base_nmse - rma_opt_nmse)
                improvements_old.append(base_nmse - rma_old_nmse)
                dataset_labels.append(f'{dataset}\n(RMa)')
        
        if improvements_opt and improvements_old:
            x_pos = np.arange(len(dataset_labels))
            width = 0.35
            
            bars1 = plt.bar(x_pos - width/2, improvements_opt, width, 
                           label='Optimized LoRA (76.8% fewer params)', color='lightblue', alpha=0.8)
            bars2 = plt.bar(x_pos + width/2, improvements_old, width,
                           label='Original LoRA', color='lightcoral', alpha=0.8)
            
            # 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.xlabel('Dataset and Transfer Type', fontsize=12)
            plt.ylabel('NMSE Improvement vs Base (dB)', fontsize=12)
            plt.title('LoRA Optimization Impact: Parameter Efficiency vs Performance', fontsize=14)
            plt.xticks(x_pos, dataset_labels)
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            # 최적화 비교 플롯 저장
            save_path_comp = Path(__file__).parent / 'lora_optimization_comparison.png'
            plt.savefig(save_path_comp, dpi=300, bbox_inches='tight')
            print(f"LoRA optimization comparison plot saved to: {save_path_comp}")
            
            plt.show()

if __name__ == "__main__":
    print("Simple LoRA Model Testing")
    print("=" * 40)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    tester = SimpleModelTester(device=device)
    results = tester.test_models()