import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator import Estimator
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
            
        # InF Transfer 모델 로드 (v4 구조로 병합되어 저장됨)
        transfer_inf_path = saved_model_dir / 'Large_estimator_v4_to_InF_test_v4.pt'
        if transfer_inf_path.exists():
            try:
                # InF Transfer 모델의 v4 구조 버전 로드
                transfer_inf_model = torch.load(transfer_inf_path, map_location=self.device)
                transfer_inf_model.eval()
                models['InF_Transfer'] = transfer_inf_model
                print("[OK] Loaded InF_Transfer (v4 structure)")
            except Exception as e:
                print(f"[ERROR] Failed to load InF_Transfer: {e}")
        else:
            print("[WARNING] Large_estimator_v4_to_InF_test_v4.pt not found")
            
        # RMa Transfer 모델 로드 (v4 구조로 병합되어 저장됨)
        transfer_rma_path = saved_model_dir / 'Large_estimator_v4_to_RMa_test_v4.pt'
        if transfer_rma_path.exists():
            try:
                # RMa Transfer 모델의 v4 구조 버전 로드
                transfer_rma_model = torch.load(transfer_rma_path, map_location=self.device)
                transfer_rma_model.eval()
                models['RMa_Transfer'] = transfer_rma_model
                print("[OK] Loaded RMa_Transfer (v4 structure)")
            except Exception as e:
                print(f"[ERROR] Failed to load RMa_Transfer: {e}")
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
        print("\n" + "="*80)
        print("SUMMARY - LoRA Transfer Learning Performance")
        print("="*80)
        
        print(f"{'Dataset':<15} {'Base_v4_1k':<15} {'InF_Transfer':<15} {'RMa_Transfer':<15} {'Best_Improvement':<15}")
        print("-" * 90)
        
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_1k', np.nan)
            inf_nmse = results[dataset_name].get('InF_Transfer', np.nan)
            rma_nmse = results[dataset_name].get('RMa_Transfer', np.nan)
            
            # 포맷팅
            base_str = f"{base_nmse:.2f}" if not np.isnan(base_nmse) else "N/A"
            inf_str = f"{inf_nmse:.2f}" if not np.isnan(inf_nmse) else "N/A"
            rma_str = f"{rma_nmse:.2f}" if not np.isnan(rma_nmse) else "N/A"
            
            # 최고 개선량 계산
            improvements = []
            if not np.isnan(base_nmse):
                if not np.isnan(inf_nmse):
                    improvements.append(base_nmse - inf_nmse)
                if not np.isnan(rma_nmse):
                    improvements.append(base_nmse - rma_nmse)
            
            if improvements:
                best_improvement = max(improvements)
                best_str = f"{best_improvement:.2f} dB"
            else:
                best_str = "N/A"
            
            print(f"{dataset_name:<15} {base_str:<15} {inf_str:<15} {rma_str:<15} {best_str:<15}")
        
        print("="*80)
    
    def plot_results(self, results):
        """결과 플롯 그리기"""
        if not results:
            print("No results to plot!")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 데이터 준비
        datasets = list(results.keys())
        models = ['Base_v4_1k', 'InF_Transfer', 'RMa_Transfer']
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        x_positions = np.arange(len(datasets))
        bar_width = 0.25
        
        # 각 모델별로 바 그래프 그리기
        for i, model in enumerate(models):
            nmse_values = []
            for dataset in datasets:
                nmse = results[dataset].get(model, np.nan)
                nmse_values.append(nmse if not np.isnan(nmse) else 0)
            
            bars = plt.bar(x_positions + i * bar_width, nmse_values, 
                          bar_width, label=model, color=colors[i], alpha=0.7)
            
            # 바 위에 값 표시
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height != 0:  # NaN이 아닌 경우만
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('LoRA Transfer Learning Performance Comparison', fontsize=14)
        plt.xticks(x_positions + bar_width, datasets)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 플롯 저장
        save_path = Path(__file__).parent / 'simple_test_results.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        
        # 개선량 플롯도 추가로 그리기
        plt.figure(figsize=(10, 6))
        
        improvements = []
        dataset_names = []
        
        for dataset in datasets:
            base_nmse = results[dataset].get('Base_v4_1k', np.nan)
            inf_nmse = results[dataset].get('InF_Transfer', np.nan)
            rma_nmse = results[dataset].get('RMa_Transfer', np.nan)
            
            if not np.isnan(base_nmse):
                # InF 개선량
                if not np.isnan(inf_nmse):
                    inf_improvement = base_nmse - inf_nmse
                    improvements.append(inf_improvement)
                    dataset_names.append(f'{dataset}\n(InF Transfer)')
                
                # RMa 개선량
                if not np.isnan(rma_nmse):
                    rma_improvement = base_nmse - rma_nmse
                    improvements.append(rma_improvement)
                    dataset_names.append(f'{dataset}\n(RMa Transfer)')
        
        if improvements:
            colors_imp = ['red' if 'InF' in name else 'green' for name in dataset_names]
            bars = plt.bar(range(len(improvements)), improvements, color=colors_imp, alpha=0.7)
            
            # 값 표시
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.xlabel('Transfer Learning Model', fontsize=12)
            plt.ylabel('NMSE Improvement (dB)', fontsize=12)
            plt.title('Transfer Learning Improvement over Base Model', fontsize=14)
            plt.xticks(range(len(dataset_names)), dataset_names, rotation=0)
            plt.grid(True, axis='y', alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            # 개선량 플롯 저장
            save_path_imp = Path(__file__).parent / 'transfer_improvements.png'
            plt.savefig(save_path_imp, dpi=300, bbox_inches='tight')
            print(f"Improvement plot saved to: {save_path_imp}")
            
            plt.show()

if __name__ == "__main__":
    print("Simple LoRA Model Testing")
    print("=" * 40)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    tester = SimpleModelTester(device=device)
    results = tester.test_models()