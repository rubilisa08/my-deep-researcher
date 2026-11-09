"""한국어 위키백과에서 딥리서처용 코퍼스를 모으고, 컨텍스트 윈도우를 넘는지 숫자로 보인다.

시드 → 1홉 → 2홉으로 넓혀 data/corpus.json 하나로 저장:
  {"docs": {"제목": "본문"}, "links": {"제목": ["docs 안의 다른 제목", ...]}}

  python collect_corpus.py                 # 수집 (중단해도 .kowiki_cache/에서 이어 받음)
  python collect_corpus.py --stats         # 통계만 (data/corpus_stats.json)
  python collect_corpus.py --stats --count # count_tokens API로 실제 토큰 수까지 (키 필요)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import requests

# ---------------------------------------------------------------- 설정
SEEDS = [
    "대형 언어 모델",
    "검색 증강 생성",
    "벡터 데이터베이스",
    "트랜스포머 (기계 학습)",
    "자연어 처리",
    "정보 검색",
]
API = "https://ko.wikipedia.org/w/api.php"
USER_AGENT = "deep-researcher-corpus/0.1 (educational project; python-requests)"  # 아스키만
MIN_CHARS = 3000      # 이보다 짧은 토막글은 저장하지 않는다
HOP1_MAX = 60         # 1홉에서 받아 볼 후보 수
HOP2_MAX = 400        # 2홉에서 받아 볼 후보 수 상한 (보통 TARGET_CHARS에서 먼저 멈춘다)
TARGET_DOCS = 30      # 최소 목표 문서 수
TARGET_CHARS = 2_200_000  # 컨텍스트 윈도우(1M 토큰)를 확실히 넘기는 양
SLEEP = 0.3

EXCLUDE = re.compile(
    r"^\d+년|^\d+년대|^\d+세기|^기원전|\d+월 \d+일|목록|^틀:|^분류:|^위키백과:|^포털:|^도움말:|동음이의|일람"
)

S = requests.Session()
S.headers["User-Agent"] = USER_AGENT
assert USER_AGENT.isascii(), "User-Agent는 아스키만"


CACHE = Path(".kowiki_cache")
_delay = [SLEEP]  # 현재 간격
_floor = [SLEEP]  # 429를 한 번이라도 받으면 1.5초로 올라가는 하한 (벌칙 대기 50초를 피하는 편이 빠르다)


def get(params: dict) -> dict:
    """한 번의 API 호출. 디스크 캐시 + retry-after 준수 + 적응형 간격."""
    key = hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    p = CACHE / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    for attempt in range(12):
        time.sleep(_delay[0])
        r = S.get(API, params=params, timeout=30)
        if r.status_code in (429, 503):
            wait = int(r.headers.get("retry-after", 0) or 0) or 10 * (attempt + 1)
            _floor[0] = max(_floor[0], 1.5)
            _delay[0] = min(max(_delay[0] * 2, _floor[0]), 5.0)
            print(f"  [429] {wait}s 대기, 이후 간격 {_delay[0]:.1f}s", file=sys.stderr, flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        _delay[0] = max(_floor[0], _delay[0] * 0.9)  # 잘 되면 간격을 천천히 되돌린다
        CACHE.mkdir(exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
    raise RuntimeError(f"429가 계속됨: {params.get('titles', '')[:60]}")


def api(params: dict) -> dict:
    """continue가 있으면 이어 받아 pages를 합친다."""
    params = {**params, "action": "query", "format": "json", "formatversion": 2}
    merged: dict = {"pages": {}, "redirects": [], "normalized": []}
    cont: dict = {}
    while True:
        data = get({**params, **cont})
        q = data.get("query", {})
        merged["redirects"] += q.get("redirects", [])
        merged["normalized"] += q.get("normalized", [])
        for p in q.get("pages", []):
            cur = merged["pages"].get(p["title"])
            if cur is None:  # 처음 본 문서
                merged["pages"][p["title"]] = {**p, "links": list(p.get("links", []))}
            else:            # continue로 이어 받은 링크를 붙인다
                cur["links"] += p.get("links", [])
                if p.get("extract") and not cur.get("extract"):
                    cur["extract"] = p["extract"]
        if "continue" not in data:
            return merged
        cont = data["continue"]


def fetch_extract(title: str) -> tuple[str, str] | None:
    """extracts는 한 번에 한 문서만. (정식 제목, 본문)"""
    d = api({"prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title})
    for t, p in d["pages"].items():
        if p.get("missing") or not p.get("extract"):
            return None
        return t, p["extract"]
    return None


def fetch_links(title: str) -> list[str]:
    d = api({"prop": "links", "plnamespace": 0, "pllimit": "max", "titles": title})
    out = []
    for p in d["pages"].values():
        out += [l["title"] for l in p.get("links", [])]
    return [t for t in dict.fromkeys(out) if not EXCLUDE.search(t)]


def page_lengths(titles: list[str]) -> dict[str, int]:
    """요청 제목 → 문서 길이(위키문법 바이트). 50개씩 묶어 조회(prop=info). 넘겨주기는 따라간다."""
    out: dict[str, int] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = api({"prop": "info", "redirects": 1, "titles": "|".join(chunk)})
        norm = {n["from"]: n["to"] for n in d["normalized"]}
        red = {r["from"]: r["to"] for r in d["redirects"]}
        for t in chunk:
            x = norm.get(t, t)
            x = red.get(x, x)
            p = d["pages"].get(x)
            out[t] = 0 if (not p or p.get("missing")) else int(p.get("length", 0))
    return out


def resolve(titles: list[str]) -> dict[str, str]:
    """링크 제목 → 정식 제목(넘겨주기 해소). 50개씩 묶어 조회."""
    m: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = api({"redirects": 1, "titles": "|".join(chunk)})
        for n in d["normalized"]:
            m[n["from"]] = n["to"]
        red = {r["from"]: r["to"] for r in d["redirects"]}
        for t in chunk:
            x = m.get(t, t)
            m[t] = red.get(x, x)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.json")
    ap.add_argument("--stats", action="store_true", help="수집하지 않고 통계만")
    ap.add_argument("--count", action="store_true", help="--stats와 함께: count_tokens로 실제 토큰 수")
    ap.add_argument("--target-chars", type=int, default=TARGET_CHARS,
                    help="총 글자 수가 이만큼 차면 멈춘다 (기본: 한국어 0.5토큰/자로 잡아도 1M 토큰을 넘는 양)")
    ap.add_argument("--hop1-max", type=int, default=HOP1_MAX)
    ap.add_argument("--hop2-max", type=int, default=HOP2_MAX)
    a = ap.parse_args()
    if a.stats:
        return corpus_stats(a.out, a.count)

    docs: dict[str, str] = {}
    raw_links: dict[str, list[str]] = {}
    skipped_short: list[str] = []

    def take(title: str, hop: int) -> bool:
        if title in docs or title in skipped_short:
            return False
        got = fetch_extract(title)
        if not got:
            print(f"  [없음] {title}")
            return False
        canon, text = got
        if canon in docs:
            return False
        if len(text) < MIN_CHARS:
            skipped_short.append(canon)
            print(f"  [토막글 {len(text)}자] {canon}")
            return False
        docs[canon] = text
        raw_links[canon] = fetch_links(canon)
        print(f"  hop{hop} {canon}: {len(text):,}자, 링크 {len(raw_links[canon])}")
        return True

    print("[hop0] 시드")
    for s in SEEDS:
        take(s, 0)
    seeds = list(docs)

    def candidates(sources: list[str], limit: int) -> list[str]:
        # 여러 출처가 함께 가리키는 제목부터. 같은 순위면 긴 문서부터.
        c = Counter(t for s in sources for t in set(raw_links[s]) if t not in docs and t not in skipped_short)
        ranked = [t for t, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))][:limit]
        size = page_lengths(ranked)
        # 위키문법 바이트 수가 MIN_CHARS 미만이면 평문도 MIN_CHARS 미만 → 본문을 받아 보지 않고 거른다
        kept = [t for t in ranked if size.get(t, 0) >= MIN_CHARS]
        print(f"  후보 {len(ranked)}개 중 길이 사전 필터 통과 {len(kept)}개")
        return sorted(kept, key=lambda t: (-c[t], -size.get(t, 0)))

    def total_chars() -> int:
        return sum(len(t) for t in docs.values())

    print("[hop1]")
    tried = 0
    for t in candidates(seeds, a.hop1_max * 3):
        if tried >= a.hop1_max:
            break
        tried += 1
        take(t, 1)
    hop1 = [d for d in docs if d not in seeds]
    print(f"  hop1까지 {len(docs)}개, {total_chars():,}자")

    print("[hop2]")
    tried = 0
    for t in candidates(seeds + hop1, a.hop2_max * 3):
        if total_chars() >= a.target_chars and len(docs) >= TARGET_DOCS:
            break
        if tried >= a.hop2_max:
            break
        tried += 1
        if take(t, 2) and len(docs) % 20 == 0:
            print(f"  ... {len(docs)}개, {total_chars():,}자", flush=True)

    # links: docs 안에 실제로 있는 제목만 (넘겨주기 해소 후)
    all_targets = sorted({t for ls in raw_links.values() for t in ls})
    canon = resolve(all_targets)
    links = {}
    for title, ls in raw_links.items():
        mapped = [canon.get(t, t) for t in ls]
        links[title] = sorted({t for t in mapped if t in docs and t != title})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"docs": docs, "links": links}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 보고
    total = sum(len(t) for t in docs.values())
    n_links = [len(v) for v in links.values()]
    inbound = Counter(t for v in links.values() for t in v)
    zero_out = [t for t, v in links.items() if not v]
    zero_in = [t for t in docs if inbound[t] == 0]
    print("\n========== corpus.json ==========")
    print(f"저장: {out}")
    print(f"문서 수: {len(docs)}  (토막글로 제외 {len(skipped_short)}건)")
    print(f"총 글자 수: {total:,}  (대략 {total // 2:,}~{total:,} 토큰 — 한국어는 글자당 0.5~1 토큰)")
    print(f"문서당 링크 수 중앙값: {statistics.median(n_links) if n_links else 0}")
    print(f"링크가 0개인 문서 ({len(zero_out)}): {zero_out}")
    print(f"(참고) 아무도 가리키지 않는 문서 — 탐색으로 닿지 않음 ({len(zero_in)}): {zero_in}")
    if total < a.target_chars:
        print(f"경고: 총 {total:,}자로 목표 {a.target_chars:,}자에 못 미침. --hop2-max를 올리거나 SEEDS를 늘리세요.",
              file=sys.stderr)
    if len(docs) < TARGET_DOCS:
        print(f"경고: 목표 {TARGET_DOCS}건 미만. SEEDS를 늘리거나 HOP 상한을 올리세요.", file=sys.stderr)

MODEL = "claude-opus-5-5"
CONTEXT = 1_000_000          # claude-opus-5-5 컨텍스트 윈도우
PER_RESEARCHER = 120_000     # 리서처 1명의 기본 읽기 예산(자, config.json의 section_budget_chars)


def corpus_stats(path: str, count: bool = False):
    """문서 수 · 총 글자 수 · 토큰(추정 또는 실측) · 컨텍스트 윈도우 대비 · 링크 통계."""
    from dotenv import load_dotenv
    load_dotenv()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs, links = data["docs"], data["links"]
    chars = sum(len(t) for t in docs.values())
    inbound = Counter(t for v in links.values() for t in v)
    st = {
        "docs": len(docs),
        "total_chars": chars,
        "est_tokens_low": int(chars * 0.5),   # 한국어 글자당 0.5~1 토큰(보수적 범위)
        "est_tokens_high": chars,
        "context_window": CONTEXT,
        "link_edges": sum(len(v) for v in links.values()),
        "links_per_doc_median": statistics.median(len(v) for v in links.values()) if links else 0,
        "zero_link_docs": [t for t, v in links.items() if not v],
        "no_inbound_docs": [t for t in docs if inbound[t] == 0],
        "per_researcher_budget_chars": PER_RESEARCHER,
        "share_one_researcher_can_read": round(PER_RESEARCHER / chars, 3),
    }
    if count:
        from anthropic import Anthropic
        client = Anthropic()
        base = client.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": "."}]).input_tokens
        total = 0
        for i, (t, text) in enumerate(docs.items()):
            n = client.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": text}]).input_tokens
            total += n - base
            print(f"\r  토큰 세는 중 {i + 1}/{len(docs)}", end="", file=sys.stderr)
        print(file=sys.stderr)
        st["counted_tokens"] = total
        st["tokens_per_char"] = round(total / chars, 3)
    out = Path(path).with_name("corpus_stats.json")
    out.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"문서 {st['docs']}개 · 총 {chars:,}자 · 링크 {st['link_edges']}개(문서당 중앙값 {st['links_per_doc_median']})")
    if "counted_tokens" in st:
        tok = st["counted_tokens"]
        print(f"실제 토큰(count_tokens, {MODEL}): {tok:,} ({st['tokens_per_char']} 토큰/자)")
        print(f"컨텍스트 윈도우 {CONTEXT:,} 대비 {tok / CONTEXT:.2f}배 → {'넘는다' if tok > CONTEXT else '넘지 않는다'}")
    else:
        lo, hi = st["est_tokens_low"], st["est_tokens_high"]
        print(f"추정 토큰 {lo:,}~{hi:,} (글자당 0.5~1) · 컨텍스트 윈도우 {CONTEXT:,}")
        print("→ " + ("낮게 잡아도 넘는다" if lo > CONTEXT else "높게 잡으면 넘는다 — --count로 확인 필요" if hi > CONTEXT
                      else "넘지 않는다 — 이 구조를 쓸 이유가 약하다"))
    print(f"리서처 1명 예산 {PER_RESEARCHER:,}자로 읽을 수 있는 몫: {st['share_one_researcher_can_read']:.1%}")
    print(f"링크 0개 문서 {len(st['zero_link_docs'])}개 · 아무도 가리키지 않는 문서 {len(st['no_inbound_docs'])}개")


if __name__ == "__main__":
    main()
