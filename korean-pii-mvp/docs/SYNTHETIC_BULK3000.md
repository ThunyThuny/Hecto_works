# B 합성 3,000문서 V3

데이터 경로는 `/workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3`입니다.
기존 V2 50문서는 `/workspace/nfs/datasets/dylee/PII/synthetic_data`에 그대로 유지합니다.
이 작업은 생성·자동 검사·AI 표본 검토까지이며 학습은 실행하지 않습니다.

| 분할 | 문서 | B 문맥 후보 | MASK | KEEP | 별도 NAME |
|---|---:|---:|---:|---:|---:|
| train | 2,000 | 10,500 | 5,775 | 4,725 | 400 |
| valid | 500 | 2,520 | 1,260 | 1,260 | 100 |
| test | 500 | 2,520 | 1,260 | 1,260 | 100 |
| 전체 | 3,000 | 15,540 | 8,295 | 7,245 | 600 |

7개 유형 PHONE/EMAIL/ADDRESS/WORKPLACE/SCHOOL/DEPARTMENT/POSITION을 균등하게
배정합니다. 각 유형은 train MASK825/KEEP675, valid/test 각각 MASK180/KEEP180입니다.
이름은 정책상 고정 MASK로 B 학습 대상에서 제외하며, 전체 후보는 16,140개입니다.

## 확대 시 보완한 사항

- 핵심 문장 448개: train 유형·역할당 16개, valid/test 각각 8개. 분할별 은행을 분리.
- 분할별 10개 제목 계열 모두 혼합60%/전체MASK20%/전체KEEP20%로 균등 배정.
  후보 수 구간에도 같은 비율 적용. 모드와 라벨 집계에서 NAME은 제외.
- 문서당 B 후보 3~8개, 문맥 유형 2~4개. NAME은 각 모드·제목 계열에 독립적으로
  문서20%에 추가하므로 전체 후보 수는 3~9개.
- 주소 조합을 8개 구×12개 도로×997개 번호로 늘리고 연락처 코드를 확장.
  개인·공용 연락처의 값은 전체 데이터에 걸쳐 분리.
- 승인된 값도 개인 용도 문맥에서는 MASK. 모든 유형·분할에 이 반례를20% 배정.
  승인 플래그만으로 KEEP/MASK를 정하는 단순 진단 규칙 정확도는80%.
  이는 학습한 B의 성능이 아니며 편향 점검용 수치.
- 모든 항목에 같은 중립 출처 번호를 붙여 여러 개인·기관 자료의 발췌임을 명시.
  같은 문서에서 여러 이력이 한 사람의 모순된 이력으로 읽히지 않도록 구성.
- 문서 내 같은 핵심 문장을 반복 선택하지 않도록 검사·대체.

각 유형의 세부 역할 수는 train에서 개인·승인300 / 개인·미승인375 /
공적·미승인150 / 공적·승인KEEP675이며, valid/test는72/72/36/180입니다.
개인·승인 연락처는 해당 값의 사적 회신·우편 수령 용도를 뜻합니다. 그 값을 KEEP
연락처로 재사용하지 않습니다. 모든 승인 목록은 가상 시뮬레이션입니다.

## 파일과 검수

`proposed_documents.jsonl`과 `proposals/{train,valid,test}.jsonl`이 생성 초안입니다.
`review.html`, `review.md`에서 원문·유형·구간·라벨을 확인하고 `review/mentions.csv`로
수정합니다. `review/reviews.jsonl`의 3,000개 선언은 모두 PENDING입니다.

`review_sample100/`에는 train60/valid20/test20의 검수 표본을 별도로 제공합니다.
7유형×4역할×3분할의84조합을 모두 포함합니다. 표본100의 AI 읽기 검토는 전체
3,000문서의 사람 검수 완료를 뜻하지 않습니다. 전체에는 자동 구간·라벨·분할·할당량·
연락처 충돌 검사를 적용합니다. 실제 토크나이저의512토큰 제한 검사 결과는
`bulk-validation.json`, AI 검토 기록은 `quality_review.json`을 확인합니다.

`generation_manifest.json`에 코드·정책·파일 해시를, `generation_plan.json`에 관계·
할당량을 기록합니다. 계획·라벨 사유·검수 상태는 B 입력에 포함하지 않습니다.
`preservation_record.json`은 기존50파일과 고정A가 유지됐음을 기록합니다.

새 경로에 같은 버전을 다시 만들 때의 명령입니다. 기존 경로에는 덮어쓸 수 없습니다.

```bash
cd /workspace/korean-pii-mvp
bash scripts/prepare_synthetic_bulk3000.sh \
  /workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3_new
```

## 사람 검수 후 단계

현재 생성 라벨을 그대로 사용하는 첫 합성 실험은 별도
[B_SYNTHETIC_EXPERIMENT.md](B_SYNTHETIC_EXPERIMENT.md)의 명시적 실험 경로를 사용합니다.
원본은 PENDING으로 유지하고 결과에 미검수 합성 실험임을 기록합니다.

검수·수정·새 지문 준비 절차는 [SYNTHETIC_WORKFLOW.md](SYNTHETIC_WORKFLOW.md)를
따릅니다. `DATA`에 이번3,000문서 경로를 사용합니다. 실제 검수자가 수정 완료한
내용을 승인한 뒤 다음처럼 분할 파일을 준비합니다.

```bash
DATA=/workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3
kpii prepare-synthetic-reviewed \
  --documents "$DATA/reviewed-documents.jsonl" \
  --reviews "$DATA/reviewed.reviews.jsonl" \
  --a-lock "$DATA/a_checkpoint.lock.json" \
  --output-dir "$DATA/ready-reviewed"
```

이 단계도 학습을 실행하지 않습니다. 일반 고정A·GPU5 B 스크립트가 준비된 검수
분할을 받을 수 있으며, 그 경로는 PENDING을 거부합니다. 별도 합성 실험 경로는
위 링크를 따릅니다. 아직 `ready-reviewed`나 B 학습 결과는 없습니다.

이번 test는 분리된 합성 문장 계열의 평가용입니다. 검수는 학습 전의 라벨·문장 품질
확인이며 모델 결과를 사용하지 않았습니다. 이후 test는 학습·체크포인트 선택·임계값
조정에 사용하지 않고 동결합니다. 실제 이력서·공고의 일반화 평가는 별도의 실제
데이터가 필요합니다.
