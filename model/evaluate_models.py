#!/usr/bin/env python3
"""
모델 성능 비교 평가 스크립트
Model Performance Comparison Script
"""

import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime


def load_results(result_file):
    """추론 결과 파일을 로드합니다."""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models(baseline_results, improved_results, output_dir="./comparison_results"):
    """두 모델의 성능을 비교합니다."""
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)
    
    # 기본 통계 비교
    baseline_summary = baseline_results.get("summary", {})
    improved_summary = improved_results.get("summary", {})
    
    comparison_data = {
        "메트릭": ["정확도", "평균 생성 시간 (초)", "총 테스트 케이스"],
        "기본 모델": [
            f"{baseline_summary.get('accuracy', 0):.2%}",
            f"{baseline_summary.get('average_generation_time', 0):.2f}",
            str(baseline_summary.get('total_cases', 0))
        ],
        "개선 모델": [
            f"{improved_summary.get('accuracy', 0):.2%}",
            f"{improved_summary.get('average_generation_time', 0):.2f}",
            str(improved_summary.get('total_cases', 0))
        ]
    }
    
    # 성능 개선 계산
    accuracy_improvement = improved_summary.get('accuracy', 0) - baseline_summary.get('accuracy', 0)
    time_improvement = baseline_summary.get('average_generation_time', 0) - improved_summary.get('average_generation_time', 0)
    
    comparison_data["개선율"] = [
        f"{accuracy_improvement:+.2%}",
        f"{time_improvement:+.2f}초",
        "동일"
    ]
    
    # DataFrame 생성 및 저장
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + "="*60)
    print("모델 성능 비교 결과")
    print("="*60)
    print(df_comparison.to_string(index=False))
    print("="*60)
    
    # CSV로 저장
    df_comparison.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    # 질문 유형별 성능 분석
    analyze_by_question_type(baseline_results, improved_results, output_dir)
    
    # 시각화
    create_comparison_plots(baseline_summary, improved_summary, output_dir)
    
    return comparison_data


def analyze_by_question_type(baseline_results, improved_results, output_dir):
    """질문 유형별 성능을 분석합니다."""
    
    def extract_by_type(results):
        type_stats = {}
        for result in results.get("detailed_results", []):
            q_type = result.get("question_type", "알 수 없음")
            if q_type not in type_stats:
                type_stats[q_type] = {"total": 0, "correct": 0, "times": []}
            
            type_stats[q_type]["total"] += 1
            if result.get("evaluation", {}).get("exact_match", False):
                type_stats[q_type]["correct"] += 1
            type_stats[q_type]["times"].append(result.get("generation_time", 0))
        
        return type_stats
    
    baseline_by_type = extract_by_type(baseline_results)
    improved_by_type = extract_by_type(improved_results)
    
    # 질문 유형별 비교 데이터 생성
    all_types = set(baseline_by_type.keys()) | set(improved_by_type.keys())
    
    type_comparison = []
    for q_type in sorted(all_types):
        baseline_acc = baseline_by_type.get(q_type, {}).get("correct", 0) / max(baseline_by_type.get(q_type, {}).get("total", 1), 1)
        improved_acc = improved_by_type.get(q_type, {}).get("correct", 0) / max(improved_by_type.get(q_type, {}).get("total", 1), 1)
        
        baseline_time = sum(baseline_by_type.get(q_type, {}).get("times", [0])) / max(len(baseline_by_type.get(q_type, {}).get("times", [1])), 1)
        improved_time = sum(improved_by_type.get(q_type, {}).get("times", [0])) / max(len(improved_by_type.get(q_type, {}).get("times", [1])), 1)
        
        type_comparison.append({
            "질문_유형": q_type,
            "기본_정확도": f"{baseline_acc:.2%}",
            "개선_정확도": f"{improved_acc:.2%}",
            "정확도_개선": f"{improved_acc - baseline_acc:+.2%}",
            "기본_평균시간": f"{baseline_time:.2f}",
            "개선_평균시간": f"{improved_time:.2f}",
            "시간_개선": f"{baseline_time - improved_time:+.2f}"
        })
    
    df_type_comparison = pd.DataFrame(type_comparison)
    print("\n질문 유형별 성능 비교:")
    print(df_type_comparison.to_string(index=False))
    
    # CSV로 저장
    df_type_comparison.to_csv(f"{output_dir}/type_comparison.csv", index=False)


def create_comparison_plots(baseline_summary, improved_summary, output_dir):
    """비교 시각화를 생성합니다."""
    
    # 설정
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 정확도 비교 차트
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 정확도 비교
    models = ['기본 모델', '개선 모델']
    accuracies = [baseline_summary.get('accuracy', 0), improved_summary.get('accuracy', 0)]
    
    bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('정확도')
    ax1.set_title('모델 정확도 비교')
    ax1.set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2%}', ha='center', va='bottom')
    
    # 생성 시간 비교
    times = [baseline_summary.get('average_generation_time', 0), 
             improved_summary.get('average_generation_time', 0)]
    
    bars2 = ax2.bar(models, times, color=['lightgreen', 'orange'])
    ax2.set_ylabel('평균 생성 시간 (초)')
    ax2.set_title('모델 생성 시간 비교')
    
    # 막대 위에 값 표시
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n시각화 차트가 {output_dir}/model_comparison.png에 저장되었습니다.")


def generate_report(baseline_results, improved_results, output_dir):
    """상세한 비교 보고서를 생성합니다."""
    
    report_content = f"""
# 모델 성능 비교 보고서
생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 요약
### 기본 모델
- 정확도: {baseline_results.get('summary', {}).get('accuracy', 0):.2%}
- 평균 생성 시간: {baseline_results.get('summary', {}).get('average_generation_time', 0):.2f}초
- 총 테스트 케이스: {baseline_results.get('summary', {}).get('total_cases', 0)}개

### 개선 모델  
- 정확도: {improved_results.get('summary', {}).get('accuracy', 0):.2%}
- 평균 생성 시간: {improved_results.get('summary', {}).get('average_generation_time', 0):.2f}초
- 총 테스트 케이스: {improved_results.get('summary', {}).get('total_cases', 0)}개

## 개선 사항
- 정확도 개선: {improved_results.get('summary', {}).get('accuracy', 0) - baseline_results.get('summary', {}).get('accuracy', 0):+.2%}
- 시간 개선: {baseline_results.get('summary', {}).get('average_generation_time', 0) - improved_results.get('summary', {}).get('average_generation_time', 0):+.2f}초

## 권장사항
1. 개선된 모델이 더 나은 성능을 보이므로 프로덕션 환경에서 사용 권장
2. 추가 훈련 데이터로 더 향상된 성능 기대 가능
3. 특정 질문 유형에 대한 추가 최적화 고려
"""
    
    # 보고서 저장
    with open(f"{output_dir}/comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"상세 보고서가 {output_dir}/comparison_report.md에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="모델 성능 비교 평가")
    parser.add_argument("--baseline", type=str, required=True, help="기본 모델 결과 파일")
    parser.add_argument("--improved", type=str, required=True, help="개선 모델 결과 파일")
    parser.add_argument("--output_dir", type=str, default="./comparison_results", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    try:
        # 결과 파일 로드
        print("결과 파일들을 로딩중...")
        baseline_results = load_results(args.baseline)
        improved_results = load_results(args.improved)
        
        # 성능 비교
        comparison_data = compare_models(baseline_results, improved_results, args.output_dir)
        
        # 보고서 생성
        generate_report(baseline_results, improved_results, args.output_dir)
        
        print(f"\n모든 비교 결과가 {args.output_dir} 디렉토리에 저장되었습니다.")
        
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main() 