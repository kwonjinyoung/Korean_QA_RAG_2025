#!/bin/bash

# 스크립트 파일들에 실행 권한 부여
# Grant execution permissions to script files

echo "스크립트 파일들에 실행 권한을 부여합니다..."

chmod +x run_train_improved.sh
chmod +x run_inference_improved.sh  
chmod +x run_inference_test_current.sh
chmod +x run_validate_qwen3_8b.sh
chmod +x run_train.sh
chmod +x run_inference.sh

echo "권한 설정 완료!"
echo "다음 스크립트들을 실행할 수 있습니다:"
echo "- ./run_train_improved.sh      (Qwen3-8B 훈련)"
echo "- ./run_inference_improved.sh  (Qwen3-8B 추론)"
echo "- ./run_validate_qwen3_8b.sh   (Qwen3-8B 검증)"
echo "- ./run_inference_test_current.sh (현재 모델 테스트)"

ls -la *.sh | grep -E "run_|setup_" 