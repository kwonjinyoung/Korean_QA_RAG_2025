"""
텍스트 평가를 위한 모듈
정답과 평가대상을 비교하여 다양한 평가 지표를 계산합니다.
"""

import re
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


# 사용 예시
if __name__ == "__main__":
    # 예시 텍스트
    #reference_text = "안녕하세요, 오늘 날씨가 정말 좋습니다."
    #candidate_text = "안녕하세요, 오늘 날씨가 매우 좋네요."

    reference_text = """\"수백여 명의 군사들이 죽었다.\"가 옳다. '명'과 같이 단위를 나타내는 말은 의존 명사이든 자립 명사이든 하나의 단어로 인정되는 명사이므로 앞말과 띄어 써야 한다. 다만, 수 관형사 뒤에 단위 명사가 붙어서 차례 및 순서를 나타내는 경우나, 단위를 나타내는 명사가 아라비아 숫자 뒤에 붙는 경우에는 단위 명사를 앞말에 붙여 쓸 수 있다."""
    candidate_text = """\"수백여 명의 군사들이 죽었다.\"가 옳다. '두 개 이상의 단위를 묶어서 이룬 단위'를 표시할 때는 두 개의 단위 사이에 가운뎃점을 쓴다. '수백 명'은 '수십 개의 십'과 '명'이 결합하여 만들어진 단위이므로, 이를 표시할 때는 '수백.명'과 같이 가운뎃점을 쓴다. 다만, 이미 굳어진 말인 '수십 년', '수만 원' 등에는 쓰지 않는다."""

    # 평가 수행
    evaluator = TextEvaluator()
    
    print("=== 기본 점수 (0-1 범위) ===")
    results_basic = evaluator.evaluate_all(reference_text, candidate_text, scale_100=False)
    print(f"Exact Match: {results_basic['exact_match']:.3f}")
    print(f"BLEURT/의미적유사도: {results_basic['bleurt']:.3f}")
    print(f"BERTScore F1: {results_basic['bert_score']['f1']:.3f}")
    print(f"ROUGE-1 F-measure: {results_basic['rouge_1']['fmeasure']:.3f}")
    
    print("\n=== 100점 만점 점수 ===")
    results_100 = evaluator.evaluate_all(reference_text, candidate_text, scale_100=True)
    print(f"Exact Match: {results_100['exact_match']:.1f}점")
    print(f"BLEURT/의미적유사도: {results_100['bleurt']:.1f}점")
    print(f"BERTScore F1: {results_100['bert_score']['f1']:.1f}점")
    print(f"ROUGE-1 F-measure: {results_100['rouge_1']['fmeasure']:.1f}점")
    
    print("\n=== 전체 점수 (가중평균) ===")
    overall_score = evaluator.get_overall_score(reference_text, candidate_text)
    print(f"전체 점수: {overall_score:.1f}점 / 100점")
    
    print("\n=== 토큰화 디버깅 ===")
    ref_tokens = korean_tokenize(reference_text)
    cand_tokens = korean_tokenize(candidate_text)
    print(f"Reference 토큰: {ref_tokens}")
    print(f"Candidate 토큰: {cand_tokens}")
    common_tokens = set(ref_tokens) & set(cand_tokens)
    print(f"공통 토큰: {common_tokens}")
    print(f"공통 토큰 수: {len(common_tokens)}")
    
    print("\n=== 큰 따옴표 추출 디버깅 ===")
    ref_quoted = extract_quoted_text(reference_text)
    cand_quoted = extract_quoted_text(candidate_text)
    print(f"Reference에서 추출된 인용문: {ref_quoted}")
    print(f"Candidate에서 추출된 인용문: {cand_quoted}")
    
    if ref_quoted and cand_quoted:
        print(f"비교할 텍스트:")
        print(f"  Reference: '{ref_quoted[0]}'")
        print(f"  Candidate: '{cand_quoted[0]}'")
        print(f"  일치 여부: {ref_quoted[0].strip().lower() == cand_quoted[0].strip().lower()}")
