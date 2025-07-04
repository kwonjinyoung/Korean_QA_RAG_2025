#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
훈련 결과 분석 및 모니터링 스크립트
Training Results Analysis and Monitoring
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']


class TrainingAnalyzer:
    """훈련 결과 분석기"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.history_file = os.path.join(results_dir, "training_history.json")
        self.history_data = None
        
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history_data = json.load(f)
        else:
            print(f"훈련 기록 파일을 찾을 수 없습니다: {self.history_file}")
    
    def print_summary(self):
        """훈련 결과 요약 출력"""
        if not self.history_data:
            print("분석할 데이터가 없습니다.")
            return
        
        print("=" * 60)
        print("🎯 자동 훈련 결과 요약")
        print("=" * 60)
        
        # 기본 정보
        training_history = self.history_data.get("training_history", [])
        best_scores = self.history_data.get("best_scores", {})
        target_scores = self.history_data.get("target_scores", {})
        best_model_path = self.history_data.get("best_model_path", "")
        
        print(f"📊 총 훈련 반복 횟수: {len(training_history)}")
        print(f"🏆 최고 성능 모델: {best_model_path}")
        print(f"📅 마지막 업데이트: {self.history_data.get('timestamp', 'N/A')}")
        print()
        
        # 목표 vs 최고 성능 비교
        print("🎯 목표 점수 vs 최고 성능:")
        print("-" * 40)
        for metric, target in target_scores.items():
            best_score = best_scores.get(metric, 0.0)
            status = "✅ 달성" if best_score >= target else "❌ 미달성"
            print(f"{metric:15} | 목표: {target:.3f} | 최고: {best_score:.3f} | {status}")
        print()
        
        # 반복별 성능 개선
        if len(training_history) > 1:
            print("📈 성능 개선 추이:")
            print("-" * 40)
            first_scores = training_history[0]["scores"]
            last_scores = training_history[-1]["scores"]
            
            for metric in ["exact_match", "f1_score", "bleu_score", "rougeL"]:
                if metric in first_scores and metric in last_scores:
                    improvement = last_scores[metric] - first_scores[metric]
                    improvement_pct = (improvement / first_scores[metric]) * 100 if first_scores[metric] > 0 else 0
                    direction = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                    print(f"{metric:15} | {direction} {improvement:+.3f} ({improvement_pct:+.1f}%)")
        print()
    
    def create_performance_charts(self):
        """성능 차트 생성"""
        if not self.history_data or not self.history_data.get("training_history"):
            print("차트 생성할 데이터가 없습니다.")
            return
        
        training_history = self.history_data["training_history"]
        target_scores = self.history_data.get("target_scores", {})
        
        # 데이터 준비
        iterations = [item["iteration"] for item in training_history]
        metrics_data = {}
        
        for metric in ["exact_match", "f1_score", "bleu_score", "rouge1", "rouge2", "rougeL"]:
            metrics_data[metric] = [item["scores"].get(metric, 0.0) for item in training_history]
        
        # 차트 생성
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('자동 훈련 성능 추이 (Auto Training Performance Trends)', fontsize=16)
        
        metric_names = {
            "exact_match": "정확도 (Exact Match)",
            "f1_score": "F1 점수",
            "bleu_score": "BLEU 점수", 
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2",
            "rougeL": "ROUGE-L"
        }
        
        for idx, (metric, scores) in enumerate(metrics_data.items()):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            # 성능 추이 그래프
            ax.plot(iterations, scores, 'o-', linewidth=2, markersize=6, label=metric_names[metric])
            
            # 목표 선 그리기
            if metric in target_scores:
                ax.axhline(y=target_scores[metric], color='red', linestyle='--', 
                          label=f'목표: {target_scores[metric]:.3f}')
            
            # 최고 성능 표시
            if scores:
                max_score = max(scores)
                max_idx = scores.index(max_score)
                ax.plot(iterations[max_idx], max_score, 'r*', markersize=15, 
                       label=f'최고: {max_score:.3f}')
            
            ax.set_xlabel('반복 횟수')
            ax.set_ylabel('점수')
            ax.set_title(metric_names[metric])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 차트 저장
        chart_path = os.path.join(self.results_dir, "performance_trends.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 성능 차트 저장: {chart_path}")
        plt.close()
    
    def create_detailed_report(self):
        """상세 보고서 생성"""
        if not self.history_data:
            return
        
        report_path = os.path.join(self.results_dir, "detailed_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 자동 훈련 상세 보고서\n\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 기본 정보
            f.write("## 📊 기본 정보\n\n")
            training_history = self.history_data.get("training_history", [])
            f.write(f"- 총 훈련 반복 횟수: {len(training_history)}\n")
            f.write(f"- 최고 성능 모델: {self.history_data.get('best_model_path', 'N/A')}\n")
            f.write(f"- 마지막 업데이트: {self.history_data.get('timestamp', 'N/A')}\n\n")
            
            # 목표 vs 결과
            f.write("## 🎯 목표 점수 vs 최고 성능\n\n")
            f.write("| 메트릭 | 목표 점수 | 최고 성능 | 달성 여부 |\n")
            f.write("|--------|----------|----------|----------|\n")
            
            target_scores = self.history_data.get("target_scores", {})
            best_scores = self.history_data.get("best_scores", {})
            
            for metric, target in target_scores.items():
                best_score = best_scores.get(metric, 0.0)
                status = "✅ 달성" if best_score >= target else "❌ 미달성"
                f.write(f"| {metric} | {target:.3f} | {best_score:.3f} | {status} |\n")
            f.write("\n")
            
            # 반복별 상세 결과
            f.write("## 📈 반복별 상세 결과\n\n")
            for item in training_history:
                f.write(f"### 반복 {item['iteration']}\n\n")
                f.write(f"- 모델 경로: `{item['model_path']}`\n")
                f.write(f"- 완료 시간: {item['timestamp']}\n")
                f.write("- 성능 점수:\n")
                
                for metric, score in item["scores"].items():
                    f.write(f"  - {metric}: {score:.4f}\n")
                f.write("\n")
            
            # 권장사항
            f.write("## 💡 권장사항\n\n")
            
            # 목표 달성 여부에 따른 권장사항
            all_targets_met = all(
                best_scores.get(metric, 0.0) >= target 
                for metric, target in target_scores.items()
            )
            
            if all_targets_met:
                f.write("✅ **모든 목표 점수를 달성했습니다!**\n\n")
                f.write("- 최고 성능 모델을 프로덕션에 배포할 수 있습니다.\n")
                f.write("- 추가적인 성능 향상을 위해 더 많은 데이터나 다른 하이퍼파라미터를 고려해보세요.\n")
            else:
                f.write("❌ **일부 목표 점수가 미달성되었습니다.**\n\n")
                f.write("다음 사항들을 고려해보세요:\n")
                f.write("- 학습률 조정 (더 작은 학습률 시도)\n")
                f.write("- 더 많은 훈련 에포크\n")
                f.write("- 데이터 품질 검토\n")
                f.write("- 모델 아키텍처 변경\n")
        
        print(f"📝 상세 보고서 저장: {report_path}")
    
    def export_to_csv(self):
        """결과를 CSV로 내보내기"""
        if not self.history_data or not self.history_data.get("training_history"):
            return
        
        training_history = self.history_data["training_history"]
        
        # DataFrame 생성
        rows = []
        for item in training_history:
            row = {
                "iteration": item["iteration"],
                "model_path": item["model_path"],
                "timestamp": item["timestamp"]
            }
            row.update(item["scores"])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, "training_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"📊 CSV 파일 저장: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="훈련 결과 분석")
    parser.add_argument("--results_dir", type=str, 
                       default="./results/auto_training",
                       help="결과 디렉토리 경로")
    parser.add_argument("--create_charts", action="store_true",
                       help="성능 차트 생성")
    parser.add_argument("--create_report", action="store_true", 
                       help="상세 보고서 생성")
    parser.add_argument("--export_csv", action="store_true",
                       help="CSV로 내보내기")
    
    args = parser.parse_args()
    
    analyzer = TrainingAnalyzer(args.results_dir)
    
    # 기본 요약 출력
    analyzer.print_summary()
    
    # 옵션에 따른 추가 분석
    if args.create_charts:
        analyzer.create_performance_charts()
    
    if args.create_report:
        analyzer.create_detailed_report()
    
    if args.export_csv:
        analyzer.export_to_csv()
    
    # 모든 분석 수행 (기본값)
    if not any([args.create_charts, args.create_report, args.export_csv]):
        analyzer.create_performance_charts()
        analyzer.create_detailed_report()
        analyzer.export_to_csv()


if __name__ == "__main__":
    main()