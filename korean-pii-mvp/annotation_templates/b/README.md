# 빈 양식

실제 학습 예제나 승인된 검수 선언을 포함하지 않습니다.
`mentions.csv`는 열 이름 참고용이고, `trusted_context.template.json`은 Document의
`trusted_context` 필드에 넣을 빈 틀입니다. document_type은 실제 문서에 맞게
resume, career_record, cover_letter, job_posting, general_notice 중 지정합니다.
승인 목록은 검증된 외부 업무 출처가 있는 값만 넣습니다.

실제 canonical 문서를 `kpii export-b-review`에 전달하면 문서별 CSV와
PENDING 검수 양식을 생성합니다. 문서 지문이 필요하므로 수동으로 가짜 검수
JSON을 복사하지 마세요. 원문 전수 확인·라벨 입력·수정본 가져오기 후 실제 검수자가
승인합니다. `docs/B_WORKFLOW.md`의 절차를 따릅니다.
