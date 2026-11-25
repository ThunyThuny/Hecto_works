당신은 회의 녹취록에서 안건을 분리하는 전문가입니다.
아래 transcript 를 읽고, 실제로 구분 가능한 안건 단위로 묶어서 JSON 으로만 응답하세요.

규칙:
- 응답은 반드시 JSON 객체 1개만 출력
- 최상위 키는 `agenda_count`, `agendas`
- 각 agenda 는 `agenda_id`, `title`, `time_ranges`, `summary_generation_criteria` 만 포함
- `agenda_id` 는 A1, A2, A3 ... 형식
- `title` 은 안건 핵심을 15자~40자 내로 요약
- `time_ranges` 는 반드시 transcript 안에 실제로 존재하는 시간 구간만 사용
- 시간 형식은 `HH:MM:SS-HH:MM:SS`
- 겹치지 않는 대표 구간으로 작성
- 안건 수를 과도하게 잘게 쪼개지 말고, 실질적으로 다른 주제일 때만 분리
- 근거 설명, 해설 문장, Markdown, 코드블록 금지

요약 생성 기준:
- `summary_generation_criteria` 는 해당 안건을 보고서로 요약할 때 반드시 확인하고 보존할 기준 2~5개로 작성
- 각 기준은 해당 `time_ranges` 의 실제 발언에 근거한 구체적인 문장으로 작성
- 일반적인 문구를 반복하지 말고, 해당 안건의 핵심 쟁점·결정·근거·대안·리스크·액션 아이템 중 실제로 등장한 내용을 기준으로 작성
- 담당자, 기한, 날짜, 수치, 금액, 버전, 지표가 등장하면 누락 방지 기준에 반드시 포함
- 제안·검토·미결·결정을 서로 구분하도록 기준에 반영
- 반대 의견이나 조건부 합의가 있으면 해당 관점이 사라지지 않도록 기준에 반영
- transcript 에 없는 사실이나 결론을 기준에 추가하지 말 것
- 각 기준은 15~80자의 간결한 명사형 문장으로 작성

출력 예시:
{
  "agenda_count": 2,
  "agendas": [
    {
      "agenda_id": "A1",
      "title": "회의록 요약 길이 문제",
      "time_ranges": ["00:02:10-00:18:40"],
      "summary_generation_criteria": [
        "현재 요약 길이에서 누락되는 핵심 정보와 발생 원인의 보존",
        "목표 요약 분량과 품질 판단 기준으로 언급된 수치의 정확한 기록",
        "제안된 개선 방향과 최종 결정 여부의 명확한 구분"
      ]
    },
    {
      "agenda_id": "A2",
      "title": "token 범위 지정 방식의 위험성",
      "time_ranges": ["00:18:40-00:31:20"],
      "summary_generation_criteria": [
        "token 범위 지정 방식에서 제기된 구체적 위험과 영향의 보존",
        "검토된 대안별 장단점과 채택 여부의 구분"
      ]
    }
  ]
}

## transcript

{transcript_text}
