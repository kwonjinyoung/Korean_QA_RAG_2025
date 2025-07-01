"""
텍스트 평가를 위한 모듈
정답과 평가대상을 비교하여 다양한 평가 지표를 계산합니다.
"""

import re
import os
import json
import pandas as pd
import argparse
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# 필요한 라이브러리 import
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import bleurt
    from bleurt import score as bleurt_score
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False


def extract_quoted_text(text: str) -> List[str]:
    """
    큰 따옴표(") 안의 텍스트를 모두 추출합니다.
    
    Args:
        text: 입력 텍스트
        
    Returns:
        List[str]: 큰 따옴표 안의 모든 텍스트 리스트
    """
    # 큰 따옴표로 둘러싸인 텍스트를 모두 찾기
    quoted_texts = re.findall(r'"([^"]*)"', text)
    return quoted_texts


def korean_tokenize(text: str) -> List[str]:
    """
    한국어 텍스트를 위한 간단한 토큰화
    구두점을 분리하고 공백으로 나눕니다.
    """
    # 구두점을 공백으로 대체
    text = re.sub(r'[^\w\s]', ' ', text)
    # 여러 공백을 하나로 통합하고 토큰화
    tokens = text.split()
    return [token.strip() for token in tokens if token.strip()]


def manual_rouge1_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    수동으로 ROUGE-1 점수를 계산 (한국어 최적화)
    """
    ref_tokens = korean_tokenize(reference.lower())
    cand_tokens = korean_tokenize(candidate.lower())
    
    if not ref_tokens and not cand_tokens:
        return {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0}
    elif not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    
    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    common_tokens = ref_set.intersection(cand_set)
    
    precision = len(common_tokens) / len(cand_set) if len(cand_set) > 0 else 0.0
    recall = len(common_tokens) / len(ref_set) if len(ref_set) > 0 else 0.0
    
    if precision + recall > 0:
        fmeasure = 2 * precision * recall / (precision + recall)
    else:
        fmeasure = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "fmeasure": fmeasure
    }


class TextEvaluator:
    """텍스트 평가를 위한 클래스"""
    
    def __init__(self, bleurt_checkpoint: Optional[str] = None):
        """
        TextEvaluator 초기화
        
        Args:
            bleurt_checkpoint: BLEURT 모델 체크포인트 경로 (선택사항)
        """
        self.bleurt_scorer = None
        if BLEURT_AVAILABLE and bleurt_checkpoint:
            try:
                self.bleurt_scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)
            except Exception as e:
                print(f"BLEURT 모델 로딩 실패: {e}")
        
        # ROUGE scorer 초기화 (한국어에는 stemmer 사용 안 함)
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
        else:
            self.rouge_scorer = None
    
    def exact_match(self, reference: str, candidate: str, scale_100: bool = False) -> float:
        """
        Exact Match 점수 계산 (큰 따옴표 안의 텍스트만 비교)
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            scale_100: 100점 만점으로 변환할지 여부
            
        Returns:
            float: 1.0 (완전 일치) 또는 0.0 (불일치), scale_100=True이면 100.0 또는 0.0
        """
        # 큰 따옴표 안의 텍스트 추출
        ref_quoted = extract_quoted_text(reference)
        cand_quoted = extract_quoted_text(candidate)
        
        # 추출된 텍스트가 없는 경우 전체 텍스트 사용
        if not ref_quoted:
            ref_text = reference.strip()
        else:
            # 여러 개의 인용문이 있는 경우 첫 번째 것을 사용
            ref_text = ref_quoted[0].strip()
            
        if not cand_quoted:
            cand_text = candidate.strip()
        else:
            # 여러 개의 인용문이 있는 경우 첫 번째 것을 사용
            cand_text = cand_quoted[0].strip()
        
        # 텍스트 정규화 (공백 정리, 소문자 변환)
        ref_normalized = re.sub(r'\s+', ' ', ref_text.lower())
        cand_normalized = re.sub(r'\s+', ' ', cand_text.lower())
        
        score = 1.0 if ref_normalized == cand_normalized else 0.0
        return score * 100 if scale_100 else score
    
    def semantic_similarity(self, reference: str, candidate: str, scale_100: bool = False) -> float:
        """
        의미적 유사도 점수 계산 (BLEURT 대신 사용)
        토큰 레벨 겹침을 기반으로 한 간단한 의미적 유사도
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            scale_100: 100점 만점으로 변환할지 여부
            
        Returns:
            float: 의미적 유사도 점수
        """
        rouge_scores = manual_rouge1_score(reference, candidate)
        
        # ROUGE-1 F-measure를 기반으로 의미적 유사도 계산
        # 더 정교한 가중치 적용
        semantic_score = rouge_scores["fmeasure"]
        
        # 길이 패널티 적용 (너무 짧거나 긴 답변에 패널티)
        ref_len = len(korean_tokenize(reference))
        cand_len = len(korean_tokenize(candidate))
        
        if ref_len > 0:
            length_ratio = min(cand_len, ref_len) / max(cand_len, ref_len)
            semantic_score *= (0.7 + 0.3 * length_ratio)  # 길이 유사성 보너스
        
        return semantic_score * 100 if scale_100 else semantic_score
    
    def bleurt_score(self, reference: str, candidate: str, scale_100: bool = False) -> float:
        """
        BLEURT 점수 계산 (사용 불가시 의미적 유사도로 대체)
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            scale_100: 100점 만점으로 변환할지 여부
            
        Returns:
            float: BLEURT 점수 또는 의미적 유사도 점수
        """
        if not BLEURT_AVAILABLE or self.bleurt_scorer is None:
            # BLEURT 대신 의미적 유사도 사용
            return self.semantic_similarity(reference, candidate, scale_100)
        
        try:
            scores = self.bleurt_scorer.score(references=[reference], candidates=[candidate])
            score = float(scores[0])
            
            if scale_100:
                # BLEURT 점수는 일반적으로 -1에서 1 사이의 값을 가지므로 0-100으로 변환
                score = max(0, min(100, (score + 1) * 50))
            
            return score
        except Exception as e:
            print(f"BLEURT 점수 계산 오류: {e}")
            # 오류 시 의미적 유사도로 대체
            return self.semantic_similarity(reference, candidate, scale_100)
    
    def bert_score(self, reference: str, candidate: str, scale_100: bool = False) -> Dict[str, float]:
        """
        BERTScore 계산
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            scale_100: 100점 만점으로 변환할지 여부
            
        Returns:
            Dict[str, float]: precision, recall, f1 점수
        """
        if not BERT_SCORE_AVAILABLE:
            print("BERTScore를 사용할 수 없습니다. bert-score 라이브러리를 설치하세요.")
            return {"precision": -1.0, "recall": -1.0, "f1": -1.0}
        
        try:
            P, R, F1 = bert_score([candidate], [reference], lang="ko", verbose=False)
            
            precision = float(P[0])
            recall = float(R[0])
            f1 = float(F1[0])
            
            if scale_100:
                precision *= 100
                recall *= 100
                f1 *= 100
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        except Exception as e:
            print(f"BERTScore 계산 오류: {e}")
            return {"precision": -1.0, "recall": -1.0, "f1": -1.0}
    
    def rouge_1_score(self, reference: str, candidate: str, scale_100: bool = False) -> Dict[str, float]:
        """
        ROUGE-1 점수 계산 (한국어 최적화)
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            scale_100: 100점 만점으로 변환할지 여부
            
        Returns:
            Dict[str, float]: precision, recall, fmeasure 점수
        """
        # 수동 계산 사용 (한국어에 더 적합)
        try:
            scores = manual_rouge1_score(reference, candidate)
            
            precision = scores["precision"]
            recall = scores["recall"]
            fmeasure = scores["fmeasure"]
            
            if scale_100:
                precision *= 100
                recall *= 100
                fmeasure *= 100
            
            return {
                "precision": precision,
                "recall": recall,
                "fmeasure": fmeasure
            }
        except Exception as e:
            print(f"ROUGE-1 점수 계산 오류: {e}")
            return {"precision": -1.0, "recall": -1.0, "fmeasure": -1.0}
    
    def evaluate_all(self, reference: str, candidate: str, scale_100: bool = False) -> Dict[str, Any]:
        """
        모든 평가 지표를 한번에 계산
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            scale_100: 100점 만점으로 변환할지 여부
            
        Returns:
            Dict[str, Any]: 모든 평가 점수
        """
        results = {}
        
        # Exact Match
        results["exact_match"] = self.exact_match(reference, candidate, scale_100)
        
        # BLEURT (또는 의미적 유사도)
        results["bleurt"] = self.bleurt_score(reference, candidate, scale_100)
        
        # BERTScore
        results["bert_score"] = self.bert_score(reference, candidate, scale_100)
        
        # ROUGE-1
        results["rouge_1"] = self.rouge_1_score(reference, candidate, scale_100)
        
        return results
    
    def get_overall_score(self, reference: str, candidate: str, weights: Optional[Dict[str, float]] = None) -> float:
        """
        가중평균을 이용한 전체 점수 계산 (100점 만점)
        
        Args:
            reference: 정답 텍스트
            candidate: 평가대상 텍스트
            weights: 각 지표별 가중치 (기본값: 동일 가중치)
            
        Returns:
            float: 전체 점수 (0-100)
        """
        if weights is None:
            weights = {
                "exact_match": 0.15,
                "bert_score_f1": 0.35,
                "rouge_1_fmeasure": 0.35,
                "bleurt": 0.15
            }
        
        results = self.evaluate_all(reference, candidate, scale_100=True)
        
        total_score = 0.0
        total_weight = 0.0
        
        # Exact Match
        if "exact_match" in weights and results["exact_match"] >= 0:
            total_score += results["exact_match"] * weights["exact_match"]
            total_weight += weights["exact_match"]
        
        # BERTScore F1
        if "bert_score_f1" in weights and results["bert_score"]["f1"] >= 0:
            total_score += results["bert_score"]["f1"] * weights["bert_score_f1"]
            total_weight += weights["bert_score_f1"]
        
        # ROUGE-1 F-measure
        if "rouge_1_fmeasure" in weights and results["rouge_1"]["fmeasure"] >= 0:
            total_score += results["rouge_1"]["fmeasure"] * weights["rouge_1_fmeasure"]
            total_weight += weights["rouge_1_fmeasure"]
        
        # BLEURT (또는 의미적 유사도)
        if "bleurt" in weights and results["bleurt"] >= 0:
            total_score += results["bleurt"] * weights["bleurt"]
            total_weight += weights["bleurt"]
        
        # 가중치 정규화
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0


def evaluate_text(reference: str, candidate: str, bleurt_checkpoint: Optional[str] = None, scale_100: bool = True) -> Dict[str, Any]:
    """
    편의 함수: 텍스트 평가를 한번에 수행
    
    Args:
        reference: 정답 텍스트
        candidate: 평가대상 텍스트
        bleurt_checkpoint: BLEURT 모델 체크포인트 경로 (선택사항)
        scale_100: 100점 만점으로 변환할지 여부 (기본값: True)
        
    Returns:
        Dict[str, Any]: 모든 평가 점수
    """
    evaluator = TextEvaluator(bleurt_checkpoint)
    return evaluator.evaluate_all(reference, candidate, scale_100)


def get_overall_score(reference: str, candidate: str, weights: Optional[Dict[str, float]] = None, bleurt_checkpoint: Optional[str] = None) -> float:
    """
    편의 함수: 전체 점수를 한번에 계산 (100점 만점)
    
    Args:
        reference: 정답 텍스트
        candidate: 평가대상 텍스트
        weights: 각 지표별 가중치
        bleurt_checkpoint: BLEURT 모델 체크포인트 경로 (선택사항)
        
    Returns:
        float: 전체 점수 (0-100)
    """
    evaluator = TextEvaluator(bleurt_checkpoint)
    return evaluator.get_overall_score(reference, candidate, weights)


def load_evaluation_data(file_path: str = "eval_input.json", limit: int = None) -> List[Dict]:
    """
    평가 데이터 로드
    
    Args:
        file_path: 평가 데이터 파일 경로
        limit: 평가할 데이터 개수 제한
        
    Returns:
        List[Dict]: 평가 데이터
    """
    print(f"📚 평가 데이터 로드 중: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"평가 데이터 파일이 존재하지 않습니다: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit and limit > 0:
        data = data[:limit]
        print(f"✅ 평가 데이터 로드 완료: {len(data)}개 문항 (제한: {limit}개)")
    else:
        print(f"✅ 평가 데이터 로드 완료: {len(data)}개 문항")
    
    return data


def evaluate_results(data: List[Dict], bleurt_checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    모든 결과 평가
    
    Args:
        data: 평가 데이터
        bleurt_checkpoint: BLEURT 모델 체크포인트 경로 (선택사항)
        
    Returns:
        Dict[str, Any]: 평가 결과
    """
    print("\n📊 평가 시작...")
    
    evaluator = TextEvaluator(bleurt_checkpoint)
    results = []
    
    # 유형별 통계를 위한 딕셔너리
    type_stats = {}
    
    # 전체 통계
    total_items = len(data)
    processed_items = 0
    error_items = 0
    
    for item in data:
        try:
            question_id = item["id"]
            question = item["input"]["question"]
            question_type = item["input"]["question_type"]
            
            # 모델 답변과 참조 답변
            model_answer = item["output"]["answer"]
            reference_answer = item["output"]["reference"]
            
            # 답변 오류 확인
            if "답변 생성 시간 초과" in model_answer or "답변 생성 오류" in model_answer:
                error_items += 1
                continue
            
            # 유형별 통계 초기화
            if question_type not in type_stats:
                type_stats[question_type] = {
                    "count": 0,
                    "scores": [],
                    "exact_match": 0,
                    "total_score": 0.0
                }
            
            # 평가 수행
            scores = evaluator.evaluate_all(reference_answer, model_answer, scale_100=True)
            overall_score = evaluator.get_overall_score(reference_answer, model_answer)
            
            # 유형별 통계 업데이트
            type_stats[question_type]["count"] += 1
            type_stats[question_type]["scores"].append(overall_score)
            type_stats[question_type]["total_score"] += overall_score
            
            if scores["exact_match"] == 100:
                type_stats[question_type]["exact_match"] += 1
            
            # 결과 저장
            result = {
                "id": question_id,
                "question_type": question_type,
                "model_answer": model_answer,
                "reference_answer": reference_answer,
                "scores": scores,
                "overall_score": overall_score
            }
            results.append(result)
            
            processed_items += 1
            
            # 진행 상황 출력
            if processed_items % 10 == 0:
                print(f"  진행 중: {processed_items}/{total_items} 문항 평가 완료")
            
        except Exception as e:
            print(f"❌ 문항 {item.get('id', '알 수 없음')} 평가 중 오류: {e}")
            error_items += 1
    
    # 전체 평균 점수 계산
    all_scores = [result["overall_score"] for result in results]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # 유형별 평균 점수 계산
    for q_type in type_stats:
        if type_stats[q_type]["count"] > 0:
            type_stats[q_type]["avg_score"] = type_stats[q_type]["total_score"] / type_stats[q_type]["count"]
            type_stats[q_type]["exact_match_rate"] = type_stats[q_type]["exact_match"] / type_stats[q_type]["count"]
    
    # 종합 결과
    evaluation_result = {
        "total_items": total_items,
        "processed_items": processed_items,
        "error_items": error_items,
        "overall_avg_score": avg_score,
        "type_stats": type_stats,
        "detailed_results": results
    }
    
    print(f"✅ 평가 완료: {processed_items}/{total_items} 문항 평가됨 (오류: {error_items})")
    
    return evaluation_result


def print_evaluation_summary(result: Dict[str, Any]):
    """
    평가 결과 요약 출력
    
    Args:
        result: 평가 결과
    """
    print("\n" + "=" * 60)
    print("📊 한국어 QA RAG 시스템 평가 결과 요약")
    print("=" * 60)
    
    print(f"총 문항 수: {result['total_items']}개")
    print(f"평가된 문항 수: {result['processed_items']}개")
    print(f"오류 문항 수: {result['error_items']}개")
    print(f"전체 평균 점수: {result['overall_avg_score']:.2f}점 / 100점")
    
    print("\n📈 유형별 평가 결과:")
    for q_type, stats in result["type_stats"].items():
        print(f"  {q_type}:")
        print(f"    - 문항 수: {stats['count']}개")
        print(f"    - 평균 점수: {stats['avg_score']:.2f}점 / 100점")
        print(f"    - 정확히 일치: {stats['exact_match']}개 ({stats['exact_match_rate']:.1%})")
    
    print("\n🔍 평가 지표별 평균 점수:")
    
    # 지표별 평균 계산
    metrics = {
        "exact_match": [],
        "bleurt": [],
        "bert_score_f1": [],
        "rouge_1_fmeasure": []
    }
    
    for item in result["detailed_results"]:
        metrics["exact_match"].append(item["scores"]["exact_match"])
        metrics["bleurt"].append(item["scores"]["bleurt"])
        metrics["bert_score_f1"].append(item["scores"]["bert_score"]["f1"])
        metrics["rouge_1_fmeasure"].append(item["scores"]["rouge_1"]["fmeasure"])
    
    # 평균 출력
    print(f"  Exact Match: {sum(metrics['exact_match']) / len(metrics['exact_match']):.2f}점")
    print(f"  BLEURT/의미적유사도: {sum(metrics['bleurt']) / len(metrics['bleurt']):.2f}점")
    print(f"  BERTScore F1: {sum(metrics['bert_score_f1']) / len(metrics['bert_score_f1']):.2f}점")
    print(f"  ROUGE-1 F-measure: {sum(metrics['rouge_1_fmeasure']) / len(metrics['rouge_1_fmeasure']):.2f}점")
    
    print("\n" + "=" * 60)


def save_evaluation_results(result: Dict[str, Any], output_file: str = "evaluation_results.json"):
    """
    평가 결과를 JSON 파일로 저장
    
    Args:
        result: 평가 결과
        output_file: 출력 파일 경로
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 평가 결과가 저장되었습니다: {output_file}")
        print(f"   - 파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        # CSV 파일로도 저장 (요약 결과만)
        csv_file = output_file.replace('.json', '.csv')
        
        # 상세 결과를 DataFrame으로 변환
        data = []
        for item in result["detailed_results"]:
            data.append({
                "id": item["id"],
                "question_type": item["question_type"],
                "exact_match": item["scores"]["exact_match"],
                "bleurt": item["scores"]["bleurt"],
                "bert_score_f1": item["scores"]["bert_score"]["f1"],
                "rouge_1_fmeasure": item["scores"]["rouge_1"]["fmeasure"],
                "overall_score": item["overall_score"]
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 평가 결과 요약이 CSV 파일로 저장되었습니다: {csv_file}")
        
    except Exception as e:
        print(f"❌ 결과 파일 저장 실패: {e}")


def main():
    """
    메인 함수: 평가 데이터 로드, 평가 수행, 결과 출력 및 저장
    """
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="한국어 QA RAG 시스템 평가")
    parser.add_argument("--limit", type=int, default=None, help="평가할 데이터 개수 제한 (기본값: 전체)")
    args = parser.parse_args()
    
    try:
        print("🚀 한국어 QA RAG 시스템 평가 시작")
        print("=" * 60)
        
        # 평가 데이터 로드 (제한 적용)
        input_file = "eval_input.json"
        data = load_evaluation_data(input_file, limit=args.limit)
        
        # 평가 수행
        result = evaluate_results(data)
        
        # 결과 요약 출력
        print_evaluation_summary(result)
        
        # 결과 저장
        save_evaluation_results(result)
        
        print("\n✅ 평가가 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


# 사용 예시
if __name__ == "__main__":
    main()
