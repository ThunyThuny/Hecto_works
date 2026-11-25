"""
파이프라인 공용 유틸리티
- Diarization 후처리 (collar 병합 + 단편 제거 + A-b-A 스무딩)
- 단어 단위 화자 매칭 (merge) + 초단문 화자 흡수 + 인접 동일화자 병합
- 한국어 텍스트 정규화
- LLM 프롬프트 (회의록 생성)
"""

import bisect
###dylee : agenda JSON 파싱 버그 수정을 위해 json import 추가
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("saerok.utils")

# merge 후처리 튜닝 (화자 과분리 완화) — env 로 조정, 기본 ON
MERGE_SAME_SPEAKER_MAX_GAP = float(os.getenv("MERGE_SAME_SPEAKER_MAX_GAP", "1e9"))  # 인접 동일화자 병합 gap 상한(s), 기본 사실상 무제한(항상 병합)
WORD_SMOOTH_MAX_WORDS = int(os.getenv("WORD_SMOOTH_MAX_WORDS", "2"))                # 초단문 화자흡수: 이 단어수 이하 run(0=off)
WORD_SMOOTH_MAX_DUR = float(os.getenv("WORD_SMOOTH_MAX_DUR", "1.0"))                # 초단문 화자흡수: 이 duration 이하(s)

# 회의 종류별 프롬프트 카탈로그
PROMPTS_DIR = Path(__file__).parent / "prompts"
MEETING_TYPES = ("seminar", "meeting", "team_report", "counsel", "interview")


def load_prompt_template(meeting_type: str) -> str:
    """prompts/<meeting_type>.md 를 매번 읽어서 반환 (재시작 없이 프롬프트 수정 반영)"""
    if meeting_type not in MEETING_TYPES:
        raise ValueError(f"Unknown meeting_type: {meeting_type!r}. Available: {MEETING_TYPES}")
    path = PROMPTS_DIR / f"{meeting_type}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Diarization 후처리
# ─────────────────────────────────────────────────────────────
def postprocess_diarization(segments, collar=0.5, min_duration=0.3, smooth_max_duration=0.0):
    """pyannote 화자분류 결과 후처리

    1. 같은 화자 칼라 병합: 인접 구간이 collar초 이내면 하나로 병합
    2. 짧은 구간 흡수: min_duration초 미만 구간을 인접 구간에 흡수
    3. 병합 후 같은 화자 인접 정리
    4. (옵션) A-b-A short-flip 스무딩: smooth_max_duration>0 이면 짧은 spurious 화자 전환 흡수
    """
    if len(segments) < 2:
        return segments

    # Step 1: 같은 화자 칼라 병합
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= collar:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    if len(merged) < 2:
        return merged

    # Step 2: 짧은 구간 흡수
    filtered = []
    for i, seg in enumerate(merged):
        duration = seg["end"] - seg["start"]

        if duration < min_duration:
            prev = filtered[-1] if filtered else None
            nxt = merged[i + 1] if i + 1 < len(merged) else None

            if prev and prev["speaker"] == seg["speaker"]:
                prev["end"] = max(prev["end"], seg["end"])
                continue
            if nxt and nxt["speaker"] == seg["speaker"]:
                nxt["start"] = min(nxt["start"], seg["start"])
                continue

            if prev and nxt:
                prev_gap = seg["start"] - prev["end"]
                next_gap = nxt["start"] - seg["end"]
                if prev_gap <= next_gap:
                    prev["end"] = max(prev["end"], seg["end"])
                else:
                    nxt["start"] = min(nxt["start"], seg["start"])
                continue
            elif prev:
                prev["end"] = max(prev["end"], seg["end"])
                continue
            elif nxt:
                nxt["start"] = min(nxt["start"], seg["start"])
                continue

        filtered.append(seg)

    # Step 3: 병합 후 같은 화자 인접 정리
    if len(filtered) < 2:
        return filtered

    final = [filtered[0]]
    for seg in filtered[1:]:
        prev = final[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = seg["end"]
        else:
            final.append(seg)

    # Step 4: A-b-A short-flip 스무딩 (옵션, smooth_max_duration>0 일 때만)
    #   짧은 구간의 양옆이 같은 화자면 spurious flip 으로 보고 흡수 → 과분리 완화
    if smooth_max_duration > 0 and len(final) >= 3:
        final = _smooth_short_flips(final, smooth_max_duration)
        merged2 = [final[0]]
        for seg in final[1:]:
            if seg["speaker"] == merged2[-1]["speaker"]:
                merged2[-1]["end"] = seg["end"]
            else:
                merged2.append(seg)
        final = merged2

    logger.info(f"  [Diar 후처리] {len(segments)}개 → {len(final)}개 구간 (collar={collar}s, min_dur={min_duration}s, smooth={smooth_max_duration}s)")
    return final


def _smooth_short_flips(segments, max_duration):
    """A-b-A 패턴 스무딩: 짧은 구간 b(<=max_duration)의 양옆이 같은 화자(A≠b)면
    b 를 흡수해 A 하나로 병합. pyannote 의 spurious 한 짧은 화자 전환 제거용."""
    if max_duration <= 0 or len(segments) < 3:
        return segments
    out = [segments[0].copy()]
    i = 1
    while i < len(segments):
        seg = segments[i]
        prev = out[-1]
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        dur = seg["end"] - seg["start"]
        if (nxt is not None and dur <= max_duration
                and prev["speaker"] == nxt["speaker"]
                and seg["speaker"] != prev["speaker"]):
            prev["end"] = nxt["end"]   # prev + b + nxt → prev 화자로 병합
            i += 2
        else:
            out.append(seg.copy())
            i += 1
    return out


# ─────────────────────────────────────────────────────────────
# 단어 단위 화자 매칭 (Merge)
# ─────────────────────────────────────────────────────────────
def _build_diarization_index(diarization):
    starts = [seg["start"] for seg in diarization]
    ends = [seg["end"] for seg in diarization]
    speakers = [seg["speaker"] for seg in diarization]
    return starts, ends, speakers


def _find_speaker(word_start, word_end, diar_index):
    starts, ends, speakers = diar_index
    if not starts:
        return "UNKNOWN"

    best_speaker = None
    best_overlap = 0.0

    right = bisect.bisect_right(starts, word_end)
    left = max(0, bisect.bisect_right(starts, word_start) - 1)

    for i in range(left, min(right, len(starts))):
        overlap = min(ends[i], word_end) - max(starts[i], word_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speakers[i]

    if best_speaker:
        return best_speaker

    midpoint = (word_start + word_end) / 2
    idx = bisect.bisect_right(starts, midpoint)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(starts):
        candidates.append(idx)

    closest_dist = float("inf")
    closest_speaker = "UNKNOWN"
    for i in candidates:
        dist = min(abs(starts[i] - midpoint), abs(ends[i] - midpoint))
        if dist < closest_dist:
            closest_dist = dist
            closest_speaker = speakers[i]

    return closest_speaker


def _normalize_korean_text(text):
    text = re.sub(r'(\d) \.(\d)', r'\1.\2', text)
    text = re.sub(r'(\d) %', r'\1%', text)
    text = re.sub(r'(\d) -(\d)', r'\1-\2', text)
    text = re.sub(r'(\w) &(\w)', r'\1&\2', text)
    return text


def _remove_repeated_phrases(text):
    text = re.sub(r'(.{4,}?)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\S+)( \1){3,}\b', r'\1', text)
    text = re.sub(r'((?:\S+ ){0,2}\S+)(?: \1){2,}', r'\1', text)
    return text


def _add_korean_punctuation(text):
    if not text or not text.strip():
        return text

    stripped = text.rstrip()
    if stripped and stripped[-1] in '.?!…':
        return text

    statement_endings = (
        '습니다', '입니다', '됩니다', '겠습니다', '있습니다', '없습니다',
        '했습니다', '됐습니다', '었습니다', '였습니다',
        '거든요', '잖아요', '네요', '군요', '더라고요', '대요',
        '해요', '돼요', '져요', '셔요',
        '었고요', '였고요', '했고요', '이에요', '예요',
        '합니다', '같아요',
        '거예요', '거에요', '는데요', '인데요', '던데요',
        '다고요', '라고요', '냐고요',
    )

    connective_endings = (
        '하고', '되고', '지고', '니까', '면서', '때문에',
        '해서', '되서', '져서', '라서', '이라',
        '는데', '인데', '던데',
    )

    for ending in connective_endings:
        if stripped.endswith(ending):
            return text

    for ending in statement_endings:
        if stripped.endswith(ending):
            return stripped + '.'

    return text


def _format_time(seconds):
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _merge_short_fragments(segments, max_chars=4, max_duration=1.0):
    if len(segments) < 2:
        return segments

    result = [segments[0]]
    for seg in segments[1:]:
        text_len = len(seg["text"].strip())
        duration = seg["end"] - seg["start"]
        is_fragment = text_len <= max_chars and duration < max_duration

        if is_fragment and result:
            prev = result[-1]
            gap = seg["start"] - prev["end"]
            if gap < 2.0 and prev["speaker"] == seg["speaker"]:
                prev["end"] = seg["end"]
                prev["text"] = prev["text"] + " " + seg["text"]
                prev["end_time"] = _format_time(prev["end"])
                if "words" in prev and "words" in seg:
                    prev["words"].extend(seg["words"])
                continue

        result.append(seg)

    return result


def _smooth_word_speakers(words, max_words=2, max_dur=1.0):
    """단어 단위 A-b-A 스무딩: 짧은 run(<=max_words 개 & <=max_dur초)의 양옆 화자가
    같으면 그 화자로 재라벨. _find_speaker 노이즈로 생긴 초단문 화자 flip 제거."""
    if max_words <= 0 or len(words) < 3:
        return words
    # 같은 화자 연속 run 경계
    runs = []
    i, n = 0, len(words)
    while i < n:
        j = i
        spk = words[i]["speaker"]
        while j < n and words[j]["speaker"] == spk:
            j += 1
        runs.append((i, j, spk))
        i = j
    for k in range(1, len(runs) - 1):
        s, e, spk = runs[k]
        prev_spk, next_spk = runs[k - 1][2], runs[k + 1][2]
        dur = words[e - 1]["end"] - words[s]["start"]
        if (prev_spk == next_spk and spk != prev_spk
                and (e - s) <= max_words and dur <= max_dur):
            for idx in range(s, e):
                words[idx]["speaker"] = prev_spk
    return words


def _merge_adjacent_same_speaker(segments, max_gap):
    """인접한 같은 화자 세그먼트 병합 (사이에 다른 화자 없음, gap<=max_gap).
    pause 로 쪼개진 연속 발화를 한 덩어리로 — 화자 과분리 완화."""
    if len(segments) < 2:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) <= max_gap:
            prev["end"] = seg["end"]
            prev["words"].extend(seg["words"])
            prev["text"] = (prev.get("text", "") + " " + seg.get("text", "")).strip()
        else:
            out.append(seg)
    return out


def merge_results(diarization, transcription):
    """단어 단위 화자 매칭 + 한국어 텍스트 정규화"""
    if not transcription:
        return {"segments": [], "speakers": []}

    diar_index = _build_diarization_index(diarization)

    # 모든 단어에 화자 할당
    labeled_words = []
    for seg in transcription:
        for word in seg.get("words", []):
            speaker = _find_speaker(word["start"], word["end"], diar_index)
            labeled_words.append({
                "start": word["start"],
                "end": word["end"],
                "word": word["word"],
                "speaker": speaker,
            })

    # 단어가 없으면 세그먼트 단위 폴백
    if not labeled_words:
        for seg in transcription:
            speaker = _find_speaker(seg["start"], seg["end"], diar_index)
            labeled_words.append({
                "start": seg["start"],
                "end": seg["end"],
                "word": seg["text"],
                "speaker": speaker,
            })

    # 초단문 화자 흡수 (단어 단위 A-b-A) — _find_speaker 노이즈 제거
    labeled_words = _smooth_word_speakers(labeled_words, WORD_SMOOTH_MAX_WORDS, WORD_SMOOTH_MAX_DUR)

    # 연속 같은 화자 단어 → 발화 병합
    merged = []
    if labeled_words:
        cur = {
            "speaker": labeled_words[0]["speaker"],
            "start": labeled_words[0]["start"],
            "end": labeled_words[0]["end"],
            "words": [{"word": labeled_words[0]["word"], "start": labeled_words[0]["start"], "end": labeled_words[0]["end"]}],
        }

        for w in labeled_words[1:]:
            if w["speaker"] == cur["speaker"] and w["start"] - cur["end"] < 2.0:
                cur["end"] = w["end"]
                cur["words"].append({"word": w["word"], "start": w["start"], "end": w["end"]})
            else:
                cur["text"] = " ".join(wd["word"] for wd in cur["words"])
                merged.append(cur)
                cur = {
                    "speaker": w["speaker"],
                    "start": w["start"],
                    "end": w["end"],
                    "words": [{"word": w["word"], "start": w["start"], "end": w["end"]}],
                }

        cur["text"] = " ".join(wd["word"] for wd in cur["words"])
        merged.append(cur)

    # 인접 동일화자 세그먼트 병합 (pause 로 쪼개진 연속 발화 합치기)
    merged = _merge_adjacent_same_speaker(merged, MERGE_SAME_SPEAKER_MAX_GAP)

    # 텍스트 후처리
    for seg in merged:
        seg["text"] = _normalize_korean_text(seg["text"])
        seg["text"] = _remove_repeated_phrases(seg["text"])
        seg["text"] = _add_korean_punctuation(seg["text"])
        seg["start_time"] = _format_time(seg["start"])
        seg["end_time"] = _format_time(seg["end"])

    merged = _merge_short_fragments(merged)
    speakers = sorted(set(s["speaker"] for s in merged))

    return {"segments": merged, "speakers": speakers}


# ─────────────────────────────────────────────────────────────
# LLM 프롬프트
# ─────────────────────────────────────────────────────────────
# 전 타입 공용 system prompt (meeting/seminar/team_report/counsel).
# 개별 persona·출력 형식은 prompts/<type>.md(user 메시지)가 담당하고,
# 여기서는 모든 타입에 공통인 불변식(사실 충실성 + Markdown 전용)만 강제한다.
MINUTES_SYSTEM_PROMPT = (
    "당신은 녹취록을 바탕으로 구조화된 문서를 작성하는 전문가입니다. "
    "녹취록에 실제로 언급된 내용만 기록하고, 명시되지 않은 사실·인과관계·결정을 지어내지 마세요. "
    "응답은 반드시 Markdown 으로만 작성하고 코드블록으로 감싸지 마세요."
)


def build_transcript_text(result, include_speakers: bool = True):
    """merge_results 결과를 LLM 입력 텍스트로 변환

    include_speakers=False 시 화자 라벨을 제거 — STT-only 시뮬레이션용
    """
    lines = []
    for seg in result["segments"]:
        start = seg.get("start_time", _format_time(seg["start"]))
        if include_speakers and "speaker" in seg:
            lines.append(f"{seg['speaker']} [{start}]: {seg['text']}")
        else:
            lines.append(f"[{start}] {seg['text']}")
    return "\n".join(lines)


def parse_minutes_response(response_text):
    """LLM 응답에서 clean Markdown 추출"""
    if not response_text:
        return "(회의록 생성 실패: 빈 응답)"
    text = response_text.strip()
    if not text:
        return "(회의록 생성 실패: 빈 응답)"
    code_block = re.search(r"```(?:markdown)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()
    return text

###dylee : v2 프롬프트 파일을 meeting_type 외 임의 이름으로도 읽을 수 있게 추가

def load_prompt_file(prompt_name: str) -> str:
    path = PROMPTS_DIR / f"{prompt_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


###dylee : 안건 분리 단계용 공통 system prompt 추가
AGENDA_SYSTEM_PROMPT = (
    "당신은 회의 녹취록에서 안건을 구조화하는 전문가입니다. "
    "응답은 반드시 JSON 으로만 작성하고, 명시된 키 외의 설명 문장은 쓰지 마세요. "
    "시간 범위는 transcript 에 실제로 존재하는 구간만 사용하세요."
)


###dylee : 안건 분리용 시간 포함 transcript 문자열 생성 함수 추가

def build_agenda_transcript_text(result, include_speakers: bool = True):
    lines = []
    for seg in result["segments"]:
        start = seg.get("start_time", _format_time(seg["start"]))
        end = seg.get("end_time", _format_time(seg["end"]))
        if include_speakers and "speaker" in seg:
            lines.append(f"{seg['speaker']} [{start}-{end}]: {seg['text']}")
        else:
            lines.append(f"[{start}-{end}] {seg['text']}")
    return "\n".join(lines)


###dylee : 안건 분리 LLM 응답(JSON) 파싱 함수 추가

def parse_agenda_response(response_text):
    if not response_text:
        return {"agenda_count": 0, "agendas": []}
    text = response_text.strip()
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        return {"agenda_count": 0, "agendas": []}

    agendas = parsed.get("agendas", []) if isinstance(parsed, dict) else []
    normalized = []
    for idx, agenda in enumerate(agendas, start=1):
        if not isinstance(agenda, dict):
            continue
        normalized.append({
            "agenda_id": str(agenda.get("agenda_id") or f"A{idx}"),
            "title": str(agenda.get("title") or f"안건 {idx}"),
            "time_ranges": [str(x) for x in agenda.get("time_ranges", []) if str(x).strip()],
            "summary_generation_criteria": [
                str(x).strip()
                for x in agenda.get("summary_generation_criteria", [])
                if str(x).strip()
            ],
        })
    return {"agenda_count": len(normalized), "agendas": normalized}


###dylee : 안건 time range를 기준으로 transcript 원문을 복원하는 후처리 함수 추가

def _parse_hms_to_seconds(value: str) -> float | None:
    try:
        parts = [int(x) for x in value.split(":")]
        if len(parts) == 2:
            m, s = parts
            return float(m * 60 + s)
        if len(parts) == 3:
            h, m, s = parts
            return float(h * 3600 + m * 60 + s)
    except Exception:
        return None
    return None


def _parse_time_range(range_text: str):
    if not range_text or "-" not in range_text:
        return None
    start_text, end_text = [x.strip() for x in range_text.split("-", 1)]
    start_sec = _parse_hms_to_seconds(start_text)
    end_sec = _parse_hms_to_seconds(end_text)
    if start_sec is None or end_sec is None:
        return None
    return start_sec, end_sec


def build_agenda_transcripts_from_ranges(segments, agenda_payload, include_speakers: bool = True):
    agendas = agenda_payload.get("agendas", []) if isinstance(agenda_payload, dict) else []
    for agenda in agendas:
        lines = []
        for range_text in agenda.get("time_ranges", []):
            parsed_range = _parse_time_range(range_text)
            if parsed_range is None:
                continue
            range_start, range_end = parsed_range
            for seg in segments:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                if seg_end < range_start or seg_start > range_end:
                    continue
                start = seg.get("start_time", _format_time(seg_start))
                end = seg.get("end_time", _format_time(seg_end))
                if include_speakers and "speaker" in seg:
                    lines.append(f"{seg['speaker']} [{start}-{end}]: {seg['text']}")
                else:
                    lines.append(f"[{start}-{end}] {seg['text']}")
        agenda["agenda_transcript"] = "\n".join(lines).strip()
    agenda_payload["agenda_count"] = len(agendas)
    return agenda_payload
