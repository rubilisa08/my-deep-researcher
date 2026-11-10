"""지표 — 정답표도 판정 모델도 쓰지 않는다. 만들어진 글(report.md)과 실제로 읽은 자료(trace.json)만 본다.

  signal: 실행끼리 높낮이를 견주는 신호. 순위표가 아니라 '어느 장치를 고칠지' 가리키는 용도.
  alarm : 0이어야 하는 경보. 0이 아니면 그 실행은 비교에서 빼고 원인부터 고친다.
지표마다 보는 장치(device)와 한 줄 설명(desc)을 단다. 한 줄로 설명이 안 되는 지표는 두지 않는다.

실행:  python metrics.py output/reports/<run>
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from graph import CITE, CONFIG, FORMAT_SECTIONS, ROOT, Corpus, Settings, tokenize


# ---------------------------------------------------------------- 문장 분리 (버전 고정)
def split_sentences(md: str) -> list[str]:
    text = re.sub(r"\n## 참고 자료.*\Z", "", md, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ">")) or set(line) <= set("-|: "):
            continue
        line = re.sub(r"^([-*+]|\d+\.)\s+", "", line)
        parts = re.split(r"(?<=[.!?。])((?:\s*\[D:[^\]]+\])*)\s+", line)  # 마침표 뒤 인용은 앞 문장에 붙인다
        buf = ""
        for k, p in enumerate(parts):
            buf += p
            if k % 2 == 1:
                out.append(buf.strip())
                buf = ""
        if buf.strip():
            out.append(buf.strip())
    return [x for x in out if len(x) >= 8]


# 문장 분리 규칙이 바뀌면 지표 분모가 바뀐다 → 버전을 기록하고 섞인 비교를 거부한다
SPLITTER_VERSION = hashlib.sha1(inspect.getsource(split_sentences).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- 지표 정의
@dataclass
class Metric:
    name: str
    kind: str      # signal | alarm
    device: str
    desc: str
    better: str = ""


METRICS = [
    Metric("citation_rate", "signal", "리서처 원고 규칙", "본문 문장 중 [D:] 인용이 붙은 비율", "high"),
    Metric("lexical_support", "signal", "리서처 인용 습관", "인용 문장의 핵심어가 인용 문서 원문에 절반 이상 나오는 문장의 비율", "high"),
    Metric("unique_cited", "signal", "리서처 탐색 폭", "보고서가 인용한 서로 다른 문서 수", "high"),
    Metric("read_to_cite", "signal", "리서처 원고 작성", "읽은 문서 중 인용으로 이어진 비율", "high"),
    Metric("seed_overlap", "signal", "코디네이터 목차·배정", "절끼리 시작 자료가 겹치는 정도(평균 자카드)", "low"),
    Metric("read_overlap", "signal", "코디네이터 목차(내용 vs 형식)", "절 담당자들이 읽은 문서 집합의 평균 자카드", "low"),
    Metric("text_overlap", "signal", "남의 구역 알려 주기", "절 본문끼리 겹치는 4-gram 비율(평균 자카드)", "low"),
    Metric("budget_use_min", "signal", "예산 배정·대조군 공정성", "에이전트별 (읽은 글자 ÷ 배정 예산)의 최솟값", "high"),
    Metric("orphan_read_rate", "signal", "고립 문서 배정", "링크로 닿지 않는 문서 중 누군가 읽은 비율", "high"),
    Metric("recheck_new_cites", "signal", "③ 점검·재파견", "재파견 뒤 최종 원고에 새로 생긴 유효 인용 수", "high"),
    Metric("isolation_ratio", "signal", "격리", "코디네이터가 본 글자 ÷ 서브에이전트들이 본 글자", "low"),
    Metric("cite_not_in_corpus", "alarm", "리서처 원고 규칙", "코퍼스에 없는 문서를 인용한 수"),
    Metric("cite_not_read", "alarm", "리서처 원고 규칙", "그 절 담당자가 읽지 않은 문서를 인용한 수"),
    Metric("invalid_assignment", "alarm", "코디네이터 배정 검사", "되물은 뒤에도 목록·명단에 없는 것을 가리킨 배정 수"),
    Metric("coordinator_raw_leak", "alarm", "격리", "읽은 원문 조각이 코디네이터 입력(원고 제외)에 나온 수"),
    Metric("synth_new_cites", "alarm", "④ 종합", "종합 단계가 원고에 없던 인용을 만든 수"),
    Metric("unsubmitted", "alarm", "리서처 루프", "턴 한도 안에 원고를 못 낸 에이전트 수"),
    Metric("model_mismatch", "alarm", "대조군 공정성", "요청과 다른 모델이 응답한 호출 수"),
    Metric("switch_audit", "alarm", "실험 스위치", "설정과 실제 실행 경로가 어긋난 항목 수"),
]
BY_NAME = {m.name: m for m in METRICS}


# ---------------------------------------------------------------- 도우미
def load(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    r = {k: json.loads((run_dir / f"{k}.json").read_text(encoding="utf-8")) for k in ("trace", "ledger", "config")}
    r["report"] = (run_dir / "report.md").read_text(encoding="utf-8")
    r["dir"] = run_dir
    return r


def is_single(trace: dict) -> bool:
    return bool(trace["agents"]) and trace["agents"][0]["agent_id"] == "single"


def section_reads(trace: dict) -> dict[str, set]:
    out: dict[str, set] = {}
    for a in trace["agents"]:
        out.setdefault(a["section"], set()).update(a["read_ids"])
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def mean_pairwise(sets: list[set]) -> float | None:
    ps = list(itertools.combinations(sets, 2))
    return sum(jaccard(a, b) for a, b in ps) / len(ps) if ps else None


def report_sections(md: str) -> dict[str, str]:
    body = re.sub(r"\n## 참고 자료.*\Z", "", md, flags=re.S)
    parts = re.split(r"\n## ", "\n" + body)[1:]
    return {p.splitlines()[0].strip(): p for p in parts if p.strip() and not p.startswith(("서론", "결론"))}


def ngrams(text: str, n: int = 4) -> set:
    t = [x for x in tokenize(CITE.sub("", text)) if not ("가" <= x[0] <= "힣" and len(x) == 2)]
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def lexical(sentence: str, ids: list[str], corpus: Corpus) -> float:
    w = set(tokenize(CITE.sub("", sentence)))
    if not w:
        return 0.0
    src = set()
    for i in ids:
        if i in corpus.text:
            src |= set(tokenize(corpus.text[i]))
    return len(w & src) / len(w)


def citation_rows(r: dict, corpus: Corpus) -> list[dict]:
    """인용 문장 하나하나 — 사람이 원문과 대조할 때 쓰는 표(데모의 인용 대조 탭)."""
    trace = r["trace"]
    reads = section_reads(trace)
    if is_single(trace):
        groups = {"all": r["report"]}
    else:
        secs = report_sections(r["report"])
        titles = {x["title"]: x["id"] for x in trace["plan"]}
        groups = {titles[t]: v for t, v in secs.items() if t in titles}
        groups["coord"] = "\n".join(v for t, v in secs.items() if t not in titles)
    allread = set().union(*reads.values()) if reads else set()
    rows = []
    for sid, text in groups.items():
        allowed = allread if sid == "coord" else reads.get(sid, set())
        for s in split_sentences(text):
            ids = CITE.findall(s)
            if ids:
                rows.append({"section": sid, "sentence": s, "cites": ids,
                             "not_in_corpus": [i for i in ids if i not in corpus.text],
                             "not_read": [i for i in ids if i in corpus.text and i not in allowed],
                             "lexical": round(lexical(s, ids, corpus), 2)})
    return rows


# ---------------------------------------------------------------- 스위치 감사
def audit_switches(r: dict) -> list[str]:
    """설정값이 아니라 실행 흔적을 본다 — '스위치를 껐다고 로그만 찍고 경로는 그대로'인 사고를 잡는다."""
    c, t = r["config"], r["trace"]
    ev = [e["event"] for e in t["events"]]
    roles = [x["role"] for x in r["ledger"]["calls"]]
    bad = []
    if not c["multi_agent"]:
        if [a["agent_id"] for a in t["agents"]] != ["single"]:
            bad.append("single인데 에이전트가 1개가 아님")
        if any(x.startswith(("coord", "researcher")) for x in roles):
            bad.append("single인데 코디네이터/리서처가 호출됨")
        if t["agents"] and t["agents"][0]["budget"] != Settings(**c).total_budget:
            bad.append("단일 에이전트 예산이 멀티 최대치와 다름")
        return bad
    plan = t["plan"]
    r1 = [a for a in t["agents"] if a["round"] == 1]
    if len(r1) != len(plan):
        bad.append(f"1차 파견 {len(r1)} != 절 {len(plan)}")
    if sum(x["budget_chars"] for x in plan) > Settings(**c).base_budget:
        bad.append("1차 배정 예산 합이 총예산을 넘음")
    peer = ["## 다른 리서처들의 구역" in a["prompt"] for a in r1]
    if c["peer_scopes"] != all(peer) or (not c["peer_scopes"] and any(peer)):
        bad.append("peer_scopes 설정과 프롬프트가 어긋남")
    checks = [e for e in t["events"] if e["event"] == "check"]
    n_re = ev.count("recheck_dispatch")
    if not c["recheck"] and n_re:
        bad.append("recheck=off인데 재파견됨")
    if n_re != sum(len(e["weak"]) for e in checks if e["dispatched"]):
        bad.append("재파견 수가 부족 신고 수와 다름")
    if sum(e["dispatched"] for e in checks) > c["max_rounds"]:
        bad.append("재파견 바퀴가 상한을 넘음")
    fmt = [x for x, _ in FORMAT_SECTIONS[: len(plan)]]
    if (c["split_by"] == "format") != ([x["title"] for x in plan] == fmt):
        bad.append("split_by 설정과 목차가 어긋남")
    if not c["assign_orphans"] and any(x.get("assigned_orphans") for x in plan):
        bad.append("assign_orphans=off인데 배정됨")
    for x in plan:
        p = next((a["prompt"] for a in r1 if a["section"] == x["id"]), "")
        if any(f"[D:{i}]" not in p for i in x["seed_doc_ids"]):
            bad.append(f"{x['id']}: 배정된 시작 자료가 리서처 프롬프트에 없음")
    synth_called = "coord:synth" in roles
    if (c["synth"] == "bridge") != (synth_called and "\n## 서론" in r["report"]):
        bad.append("synth 설정과 종합 호출이 어긋남")
    return bad


# ---------------------------------------------------------------- 계산
def compute(r: dict, corpus: Corpus) -> dict:
    t, md = r["trace"], r["report"]
    single = is_single(t)
    sents = split_sentences(md)
    rows = citation_rows(r, corpus)
    cited = {c for x in rows for c in x["cites"]}
    allread = set().union(*[set(a["read_ids"]) for a in t["agents"]]) if t["agents"] else set()
    reads = section_reads(t)
    orphans = t["corpus"]["orphans"]
    iso = t["isolation"]
    new_cites = 0
    if t.get("decisions"):
        first = {a["section"]: a["submitted"]["markdown"] for a in t["agents"] if a["round"] == 1}
        for sid, m in first.items():
            fresh = set(CITE.findall(t["drafts"].get(sid, ""))) - set(CITE.findall(m))
            new_cites += len({c for c in fresh if c in corpus.text and c in reads.get(sid, set())})
    return {
        "citation_rate": sum(1 for s in sents if CITE.search(s)) / max(1, len(sents)),
        "lexical_support": sum(x["lexical"] >= 0.5 for x in rows) / max(1, len(rows)),
        "unique_cited": len(cited & set(corpus.text)),
        "read_to_cite": len(cited & allread) / max(1, len(allread)),
        "seed_overlap": None if single else mean_pairwise([set(x["seed_doc_ids"]) for x in t["plan"]]),
        "read_overlap": None if single else mean_pairwise(list(reads.values())),
        "text_overlap": mean_pairwise([ngrams(v) for v in report_sections(md).values()]),
        "budget_use_min": min((a["used"] / a["budget"] for a in t["agents"] if a["budget"]), default=None),
        "orphan_read_rate": len(set(orphans) & allread) / len(orphans) if orphans else None,
        "recheck_new_cites": None if single else new_cites,
        "isolation_ratio": iso["coordinator_chars"] / iso["subagent_chars_total"] if iso["subagent_chars_total"] else None,
        "cite_not_in_corpus": sum(len(x["not_in_corpus"]) for x in rows),
        "cite_not_read": sum(len(x["not_read"]) for x in rows),
        "invalid_assignment": t.get("plan_info", {}).get("invalid_final", 0),
        "coordinator_raw_leak": iso["coordinator_raw_leak"],
        "synth_new_cites": len(t.get("synth", {}).get("new_cites", [])),
        "unsubmitted": sum(1 for a in t["agents"] if a["submitted"].get("unsubmitted")),
        "model_mismatch": sum(1 for x in r["ledger"]["calls"] if not str(x["served_model"]).startswith(x["requested_model"])),
        "switch_audit": len(audit_switches(r)),
    }


def evaluate(run_dir: Path, corpus: Corpus | None = None) -> dict:
    r = load(run_dir)
    corpus = corpus or Corpus()
    v = compute(r, corpus)
    m = {"run": Path(run_dir).name, "splitter_version": SPLITTER_VERSION,
         "question_id": r["trace"]["question"]["id"], "arm": Settings(**r["config"]).arm(), "seed": r["config"]["seed"],
         "backend": r["config"]["backend"],
         "signals": {k: v[k] for k in v if BY_NAME[k].kind == "signal"},
         "alarms": {k: v[k] for k in v if BY_NAME[k].kind == "alarm"},
         "audit": audit_switches(r),
         "budget": [{"agent": a["agent_id"], "budget": a["budget"], "used": a["used"],
                     "context_limited": a["context_limited"], "nudges": a["nudges"]} for a in r["trace"]["agents"]],
         "tokens": r["ledger"]["total"], "elapsed_s": r["trace"]["elapsed_s"]}
    m["alarms_ok"] = not any(m["alarms"].values())
    (Path(run_dir) / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def record_run(run_dir: Path, corpus: Corpus | None = None) -> dict:
    """평가하고 output/runs.jsonl에 한 줄 추가한다."""
    m = evaluate(run_dir, corpus)
    line = {k: m[k] for k in ("run", "question_id", "arm", "seed", "backend", "alarms_ok", "signals", "alarms",
                              "splitter_version", "elapsed_s")}
    line["report"] = str(Path(run_dir).relative_to(ROOT) / "report.md").replace("\\", "/")
    with open(ROOT / CONFIG["output_dir"] / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return m


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(evaluate(Path(d)), ensure_ascii=False, indent=1))
