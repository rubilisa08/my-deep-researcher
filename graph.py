"""딥리서처 그래프: 기획 → 배치 → 서브에이전트 → 점검 → 종합 (LangGraph).

    START → plan ─(Send × 절 수)→ researcher ─→ check ─(부족 신고 절만 Send)→ researcher …
                                                  └─(없거나 바퀴 상한)→ synth → END

- 코디네이터(plan·check·synth)는 문서 id·제목 목록과 리서처 원고만 본다. 원문은 리서처 컨텍스트에만 있다.
- 누가 몇 글자를 봤는지는 Meter가 센다(격리를 숫자로).
- 산출물: output/reports/<run>/ (report.md · trace.json · plan.json · ledger.json · config.json)

실행:  python graph.py --question 3 [--arm full|no_peer|no_recheck|format_split|concat|no_orphan] [--backend mock]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import operator
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CITE = re.compile(r"\[D:([^\]]+)\]")


# ================================================================ 설정 · 실험 스위치
@dataclass
class Settings:
    question_id: int = 3
    question: str = ""               # 직접 입력 질문 (있으면 question_id 대신)
    backend: str = "api"             # api(config.json의 provider) | mock(오프라인 배선 확인용)
    model: str = CONFIG["model"]
    effort: str = CONFIG["effort"]
    context_tokens: int = CONFIG["context_tokens"]
    n_sections: int = CONFIG["n_sections"]
    section_budget_chars: int = CONFIG["section_budget_chars"]
    section_turns: int = CONFIG["section_turns"]
    max_rounds: int = CONFIG["max_rounds"]
    recheck_budget_ratio: float = CONFIG["recheck_budget_ratio"]
    min_budget_use: float = CONFIG["min_budget_use"]
    max_parallel: int = CONFIG["max_parallel"]
    # --- ablation 스위치 (ablation.py가 하나씩 끈다) ---
    multi_agent: bool = True         # False → baseline.py 단일 에이전트
    split_by: str = "content"        # content | format
    peer_scopes: bool = True         # 남의 구역 알려 주기
    recheck: bool = True             # ③ 점검·재파견
    synth: str = "bridge"            # bridge(서론·결론만 LLM) | concat
    assign_orphans: bool = True      # 링크로 닿지 않는 문서 강제 배정
    seed: int = 0                    # 시행 번호
    tag: str = ""

    @property
    def base_budget(self) -> int:
        return self.section_budget_chars * self.n_sections

    @property
    def recheck_pool(self) -> int:
        return int(self.base_budget * self.recheck_budget_ratio)

    @property
    def total_budget(self) -> int:
        """단일 에이전트 대조군 예산 = 멀티가 쓸 수 있는 최대치 (1차 + 재파견 풀)."""
        return self.base_budget + self.recheck_pool

    @property
    def total_turns(self) -> int:
        base = self.section_turns * self.n_sections
        return base + int(base * self.recheck_budget_ratio)

    def arm(self) -> str:
        if not self.multi_agent:
            return "single"
        off = [n for n, cond in [("format_split", self.split_by == "format"), ("no_peer", not self.peer_scopes),
                                 ("no_recheck", not self.recheck), ("concat", self.synth == "concat"),
                                 ("no_orphan", not self.assign_orphans)] if cond]
        return "-".join(off) or "full"

    def run_name(self) -> str:
        q = f"qx{hashlib.sha1(self.question.encode()).hexdigest()[:6]}" if self.question else f"q{self.question_id}"
        return f"{q}__{self.arm()}__{self.backend}__s{self.seed}" + (f"__{self.tag}" if self.tag else "")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "Settings":
        return dataclasses.replace(self, **kw)


ARMS: dict[str, dict] = {
    "full": {},
    "no_peer": {"peer_scopes": False},
    "no_recheck": {"recheck": False},
    "format_split": {"split_by": "format"},
    "concat": {"synth": "concat"},
    "no_orphan": {"assign_orphans": False},
    "single": {"multi_agent": False},
}


def load_questions() -> list[dict]:
    return json.loads((ROOT / CONFIG["questions_path"]).read_text(encoding="utf-8"))["questions"]


def resolve_question(s: Settings) -> dict:
    if s.question:
        return {"id": None, "kind": "custom", "title": s.question[:60], "question": s.question, "why": ""}
    return load_questions()[s.question_id]


# ================================================================ LLM (공급자별 어댑터)
@dataclass
class Ledger:
    calls: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw):
        with self._lock:
            self.calls.append(kw)

    def totals(self) -> dict:
        return {"calls": len(self.calls), "input_tokens": sum(c["input_tokens"] for c in self.calls),
                "cached_input_tokens": sum(c.get("cached_tokens", 0) for c in self.calls),
                "output_tokens": sum(c["output_tokens"] for c in self.calls),
                "served_models": sorted({c["served_model"] for c in self.calls})}


class Blocked(RuntimeError):
    pass


class ContextFull(RuntimeError):
    """모델 컨텍스트 윈도우를 넘어 요청이 거절됨."""


@dataclass
class Call:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    """공급자와 무관한 응답. 에이전트 루프는 이것만 본다."""
    text: str
    calls: list[Call]
    stop: str           # end | tool | pause | length
    n_input: int        # 이번 요청의 입력 토큰(=컨텍스트 크기) — 85% 한도 판단용
    raw: Any = None     # 공급자 원본(대화에 되붙일 때 사용)


class OpenAILLM:
    """OpenAI Responses API. 도구 호출(function calling) + 추론 강도(reasoning.effort)를 함께 쓸 수 있다.
    (gpt-5.4-mini 등은 Chat Completions에서 도구와 추론 강도를 같이 쓰지 못한다.)
    대화는 입력 항목 목록으로 잇는다: 응답의 출력 항목(추론·함수 호출)을 그대로 되붙이고, 도구 결과는 function_call_output으로."""

    def __init__(self, s: Settings, ledger: Ledger):
        import openai
        self.openai = openai
        self.client = openai.OpenAI(api_key=_api_key("OPENAI_API_KEY"), max_retries=4)
        self.s, self.ledger = s, ledger

    @staticmethod
    def _tools(tools):
        return [{"type": "function", "name": t["name"], "description": t["description"],
                 "parameters": t["input_schema"], "strict": False} for t in tools]

    def chat(self, *, role, system, messages, tools=None, max_tokens=16000, json_schema=None) -> Reply:
        kw = dict(model=self.s.model, instructions=system, input=messages, max_output_tokens=max_tokens)
        if self.s.effort:
            kw["reasoning"] = {"effort": self.s.effort}
        if tools:
            kw["tools"] = self._tools(tools)
        if json_schema:
            kw["text"] = {"format": {"type": "json_schema", "name": "output", "schema": json_schema, "strict": True}}
        t0 = time.time()
        try:
            r = self.client.responses.create(**kw)
        except self.openai.BadRequestError as e:
            msg = str(e)
            if "context_length" in msg or "context window" in msg or "maximum context" in msg:
                raise ContextFull(msg[:200]) from e
            raise
        u = r.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        stop_raw = r.status if r.status != "incomplete" else f"incomplete:{getattr(r.incomplete_details, 'reason', '')}"
        self.ledger.add(role=role, requested_model=self.s.model, served_model=r.model, input_tokens=u.input_tokens,
                        cached_tokens=cached, output_tokens=u.output_tokens, stop_reason=stop_raw,
                        seconds=round(time.time() - t0, 1))
        text, calls, items = [], [], []
        for it in r.output:
            items.append(it.model_dump(exclude_none=True))
            if it.type == "message":
                for c in it.content:
                    if c.type == "output_text":
                        text.append(c.text)
                    elif c.type == "refusal":
                        raise Blocked(f"{role}: refusal")
            elif it.type == "function_call":
                try:
                    args = json.loads(it.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_invalid_json": it.arguments}
                calls.append(Call(it.call_id, it.name, args))
        if calls:
            stop = "tool"
        elif r.status == "incomplete":
            stop = "length"
        else:
            stop = "end"
        return Reply("".join(text), calls, stop, u.input_tokens, items)

    def assistant_msg(self, rep: Reply) -> list[dict]:
        # 응답 출력 항목(추론 항목 포함)을 그대로 되붙인다 — 추론을 이어 가려면 필요하다
        out = []
        for it in rep.raw:
            it = {k: v for k, v in it.items() if k not in ("status",) or it.get("type") == "message"}
            out.append(it)
        return out

    def result_msgs(self, results: list[tuple[str, str, bool]]) -> list[dict]:
        return [{"type": "function_call_output", "call_id": i, "output": ("[오류] " if err else "") + out}
                for i, out, err in results]


class AnthropicLLM:
    """Anthropic Messages API (Claude). sk-ant- 키를 쓸 때 config.json의 provider를 anthropic으로."""

    def __init__(self, s: Settings, ledger: Ledger):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=_api_key("ANTHROPIC_API_KEY"), max_retries=4)
        self.s, self.ledger = s, ledger

    def chat(self, *, role, system, messages, tools=None, max_tokens=16000, json_schema=None) -> Reply:
        oc: dict = {"effort": self.s.effort}
        if json_schema:
            oc["format"] = {"type": "json_schema", "schema": json_schema}
        kw = dict(model=self.s.model, max_tokens=max_tokens, system=system, messages=messages,
                  output_config=oc, cache_control={"type": "ephemeral"})
        if tools:
            kw["tools"] = tools
        t0 = time.time()
        try:
            with self.client.messages.stream(**kw) as st:
                r = st.get_final_message()
        except self.anthropic.BadRequestError as e:
            if "prompt is too long" in str(e):
                raise ContextFull(str(e)[:200]) from e
            raise
        u = r.usage
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        n_in = (u.input_tokens or 0) + cached + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        self.ledger.add(role=role, requested_model=self.s.model, served_model=r.model, input_tokens=n_in,
                        cached_tokens=cached, output_tokens=u.output_tokens or 0, stop_reason=r.stop_reason,
                        seconds=round(time.time() - t0, 1))
        if r.stop_reason == "refusal":
            raise Blocked(f"{role}: refusal")
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [Call(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        stop = {"tool_use": "tool", "pause_turn": "pause", "max_tokens": "length"}.get(r.stop_reason, "end")
        return Reply(text, calls, stop, n_in, r)

    def assistant_msg(self, rep: Reply) -> dict:
        return {"role": "assistant", "content": rep.raw.content}

    def result_msgs(self, results: list[tuple[str, str, bool]]) -> list[dict]:
        return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": out,
                                              **({"is_error": True} if err else {})} for i, out, err in results]}]


class MockLLM:
    """API 없이 그래프 배선을 확인하는 가짜 응답. 내용은 의미 없다."""

    def __init__(self, s: Settings, ledger: Ledger):
        self.s, self.ledger, self._n, self._lock = s, ledger, 0, threading.Lock()

    def _id(self):
        with self._lock:
            self._n += 1
            return f"call_mock_{self._n}"

    def chat(self, *, role, system, messages, tools=None, json_schema=None, **_) -> Reply:
        blob = json.dumps(messages, ensure_ascii=False, default=str)
        self.ledger.add(role=role, requested_model=self.s.model, served_model=self.s.model,
                        input_tokens=len(blob) // 2, cached_tokens=0, output_tokens=200, stop_reason="end", seconds=0)
        ids = list(dict.fromkeys(CITE.findall(blob)))
        if json_schema:  # 기획
            n = int(re.search(r"절 수: (\d+)", blob).group(1))
            roles = re.findall(r"- \[역할\] ([^:]+):", blob) or ["?"]
            secs = [{"title": f"모의 절 {k + 1}", "role": roles[k % len(roles)], "scope": f"모의 범위 {k + 1}",
                     "questions": ["무엇이 바뀌었나?"], "seed_doc_ids": ids[k::n][:3], "search_hints": ["모델"],
                     "budget_weight": 1 + k % 3} for k in range(n)]
            return Reply(json.dumps({"sections": secs}, ensure_ascii=False), [], "end", 0)
        if tools:
            turns = sum(1 for m in messages if m["role"] == "assistant")
            if turns == 0:
                return Reply("", [Call(self._id(), "search_corpus", {"query": "모델 학습 신경망"})], "tool", 0)
            read = set(re.findall(r'"doc_id": "([^"]+)"', blob))
            todo = [i for i in ids if i not in read]
            if turns < 4 and todo:
                return Reply("", [Call(self._id(), "read_doc", {"doc_id": todo[0]})], "tool", 0)
            cites = " ".join(f"[D:{i}]" for i in sorted(read)[:3])
            ok = "재조사" in blob or "절: s1 「" not in blob
            return Reply("", [Call(self._id(), "submit_section", {
                "markdown": f"모의 원고의 첫 문장이다 {cites}.\n\n둘째 문장에는 인용이 없다.",
                "sufficient": ok, "gaps": [] if ok else ["모의 공백"]})], "tool", 0)
        return Reply("### 서론\n모의 서론이다.\n### 결론\n모의 결론이다.", [], "end", 0)

    def assistant_msg(self, rep: Reply) -> dict:
        return {"role": "assistant", "content": rep.text,
                "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in rep.calls]}

    def result_msgs(self, results):
        return [{"role": "tool", "tool_call_id": i, "content": out} for i, out, _ in results]


def _api_key(preferred: str) -> str | None:
    """.env의 키 이름이 공급자와 달라도(예: OpenAI 키를 ANTHROPIC_API_KEY에 넣음) 형식으로 알아본다."""
    import os
    want = "sk-ant-" if preferred.startswith("ANTHROPIC") else "sk-"
    for name in (preferred, "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        v = os.environ.get(name, "")
        if v.startswith(want) and (want == "sk-ant-" or not v.startswith("sk-ant-")):
            return v
    return os.environ.get(preferred)


def make_llm(s: Settings, ledger: Ledger):
    if s.backend == "mock":
        return MockLLM(s, ledger)
    return {"openai": OpenAILLM, "anthropic": AnthropicLLM}[CONFIG["provider"]](s, ledger)


# ================================================================ 코퍼스 · 검색 · 도구
TOKEN = re.compile(r"[a-z0-9_]+|[가-힣]+")


def tokenize(s: str) -> list[str]:
    out = []
    for t in TOKEN.findall(s.lower()):
        if len(t) < 2:
            continue
        out.append(t)
        if "가" <= t[0] <= "힣" and len(t) > 2:  # 조사 붙은 어절도 맞도록 글자 바이그램
            out += [t[i:i + 2] for i in range(len(t) - 1)]
    return out


class Corpus:
    """data/corpus.json = {"docs": {제목: 본문}, "links": {제목: [제목]}}"""

    def __init__(self, path: Path | None = None):
        data = json.loads(Path(path or ROOT / CONFIG["corpus_path"]).read_text(encoding="utf-8"))
        self.text: dict[str, str] = data["docs"]
        self.links: dict[str, list[str]] = {k: [t for t in v if t in self.text] for k, v in data["links"].items()}
        self.ids = list(self.text)
        self.backlinks: dict[str, list[str]] = {i: [] for i in self.ids}
        for i, ls in self.links.items():
            for t in ls:
                self.backlinks[t].append(i)
        df: Counter = Counter()
        self._tf = []
        for i in self.ids:
            toks = tokenize(i) * 3 + tokenize(self.text[i][:20000])
            tf = Counter(toks)
            self._tf.append((tf, len(toks)))
            df.update(tf.keys())
        n = len(self.ids)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._avg = sum(l for _, l in self._tf) / max(1, n)

    def url(self, i: str) -> str:
        return f"https://{CONFIG['wiki_lang']}.wikipedia.org/wiki/{i.replace(' ', '_')}"

    def total_chars(self) -> int:
        return sum(len(t) for t in self.text.values())

    def orphans(self) -> list[str]:
        """링크만 따라가서는 닿지 못하는 문서: 나가는 링크가 0개이거나 들어오는 링크가 0개."""
        return [i for i in self.ids if not self.links.get(i) or not self.backlinks[i]]

    def search(self, q: str, k: int = 8) -> list[tuple[str, float]]:
        qt = tokenize(q)
        out = []
        for idx, i in enumerate(self.ids):
            tf, dl = self._tf[idx]
            s = sum(self._idf.get(t, 0) * tf[t] * 2.4 / (tf[t] + 1.4 * (0.25 + 0.75 * dl / self._avg)) for t in qt if t in tf)
            if s > 0:
                out.append((i, s))
        return sorted(out, key=lambda x: -x[1])[:k]

    def catalog(self) -> str:
        """코디네이터가 보는 목록: id(=제목)와 길이·링크 수뿐, 본문은 없다."""
        return "\n".join(f"[D:{i}] ({len(self.text[i]):,}자, 링크 {len(self.links.get(i, []))})" for i in self.ids)


TOOLS = [
    {"name": "search_corpus", "description": "코퍼스를 검색해 문서 id·길이·짧은 미리보기를 돌려준다. 읽기 예산을 쓰지 않는다.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                      "additionalProperties": False}},
    {"name": "read_doc", "description": "문서 원문을 offset부터 최대 6000자 읽는다. 읽은 글자 수만큼 읽기 예산이 준다.",
     "input_schema": {"type": "object", "properties": {"doc_id": {"type": "string"}, "offset": {"type": "integer"}},
                      "required": ["doc_id"], "additionalProperties": False}},
    {"name": "follow_links", "description": "이 문서가 가리키는 문서와 이 문서를 가리키는 문서 목록. 예산을 쓰지 않는다.",
     "input_schema": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"],
                      "additionalProperties": False}},
    {"name": "submit_section", "description": "원고를 제출하고 작업을 끝낸다. 마지막에 한 번 호출한다.",
     "input_schema": {"type": "object", "properties": {
         "markdown": {"type": "string", "description": "본문. 사실 문장마다 [D:문서id] 인용"},
         "sufficient": {"type": "boolean", "description": "읽은 자료로 맡은 질문에 충분히 답했는가"},
         "gaps": {"type": "array", "items": {"type": "string"}, "description": "부족하면 무엇이 부족한지"}},
         "required": ["markdown", "sufficient", "gaps"], "additionalProperties": False}},
]
CHUNK = 6000


class Toolbox:
    """에이전트 1명 전용 도구. 읽은 곳(read_log)과 예산 사용을 기록한다."""

    def __init__(self, corpus: Corpus, budget: int, agent_id: str):
        self.c, self.budget, self.used, self.agent_id = corpus, budget, 0, agent_id
        self.read_log: list[dict] = []
        self.search_log: list[str] = []
        self.submitted: dict | None = None

    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def read_ids(self) -> list[str]:
        return sorted({r["doc_id"] for r in self.read_log})

    def run(self, name: str, a: dict) -> tuple[str, bool]:
        try:
            if name == "search_corpus":
                self.search_log.append(a["query"])
                hits = self.c.search(a["query"])
                if not hits:
                    return "결과 없음. 다른 검색어를 써라.", False
                return f"남은 읽기 예산 {self.remaining():,}자\n" + "\n".join(
                    f"[D:{i}] ({len(self.c.text[i]):,}자){' (이미 읽음)' if i in self.read_ids else ''}\n  "
                    + " ".join(self.c.text[i][:180].split()) for i, _ in hits), False
            if name in ("read_doc", "follow_links"):
                i = a["doc_id"].strip().strip("[]").removeprefix("D:")
                if i not in self.c.text:
                    return f"없는 문서 id: {i}", True
                if name == "follow_links":
                    f = lambda ids: "\n".join(f"  [D:{x}]" for x in ids[:25]) or "  (없음)"
                    return f"가리키는 문서:\n{f(self.c.links.get(i, []))}\n가리켜지는 문서:\n{f(self.c.backlinks[i])}", False
                if self.remaining() <= 0:
                    return "읽기 예산 소진. 지금까지 읽은 자료로 submit_section 하라.", True
                off = int(a.get("offset") or 0)
                n = min(CHUNK, self.remaining(), max(0, len(self.c.text[i]) - off))
                self.used += n
                self.read_log.append({"doc_id": i, "offset": off, "n": n})
                more = len(self.c.text[i]) - off - n
                return (f"[D:{i}] 남은 예산 {self.remaining():,}자\n\n{self.c.text[i][off:off + n]}"
                        + (f"\n\n[… {more:,}자 더 있음: offset={off + n}]" if more > 0 else "\n\n[문서 끝]")), False
            if name == "submit_section":
                self.submitted = {"markdown": str(a.get("markdown", "")), "sufficient": bool(a.get("sufficient")),
                                  "gaps": list(a.get("gaps") or [])}
                return "제출 완료.", False
            return f"알 수 없는 도구 {name}", True
        except (KeyError, TypeError, ValueError) as e:
            return f"입력 오류: {e!r}", True


class Meter:
    """격리를 숫자로: 코디네이터에게 들어간 글(종류별)과 서브에이전트마다 들어간 글자 수를 따로 센다."""

    def __init__(self):
        self.coord: list[tuple[str, str]] = []
        self.sub: dict[str, int] = {}
        self._lock = threading.Lock()

    def see_coord(self, kind: str, text: str):
        with self._lock:
            self.coord.append((kind, text))

    def see_sub(self, agent_id: str, text: str):
        with self._lock:
            self.sub[agent_id] = self.sub.get(agent_id, 0) + len(text)

    def summary(self) -> dict:
        by: dict[str, int] = {}
        for k, t in self.coord:
            by[k] = by.get(k, 0) + len(t)
        return {"coordinator_chars": sum(by.values()), "coordinator_by_kind": by,
                "subagent_chars": dict(self.sub), "subagent_chars_total": sum(self.sub.values())}


# ================================================================ 에이전트 루프 (리서처 · 단일 에이전트 공용)
RESEARCHER_SYSTEM = """너는 딥리서치 팀의 리서처 서브에이전트다. 장문 보고서 중 '네 절 하나'만 맡는다.
너의 역할: {role} — {focus}

작업 방식
- search_corpus / follow_links로 자료를 찾고 read_doc으로 원문을 읽는다. 예산이 한정되어 있으니 골라 읽어라.
- 문서끼리 가리키는 링크를 따라가며 맥락을 넓힌다.
- 마지막에 submit_section을 호출한다. 원문은 위로 올라가지 않고 원고만 올라간다.

원고 규칙
- 한국어 마크다운. 절 제목은 쓰지 말고 본문만(소제목 ### 가능).
- 다룰 것: 맡은 대상이 지금 모양이 된 경위, 경쟁하거나 밀려난 대안과 그 이유, 남은 한계.
- 사실 문장마다 끝에 [D:문서id]를 붙인다. read_doc으로 실제로 읽은 문서만, 그 문장의 내용이 적힌 문서에만 붙인다.
- 자료에 없는 것은 추측하지 말고 "자료에서 확인되지 않음"이라고 쓴다.
- sufficient는 정직하게: 근거를 못 찾은 질문이 있으면 false와 gaps를 적는다."""

SINGLE_SYSTEM = (RESEARCHER_SYSTEM.split("\n\n", 1)[1]
                 .replace("절 제목은 쓰지 말고 본문만(소제목 ### 가능).", "보고서 전체를 쓴다. 스스로 목차를 정해 ## 절 제목을 붙인다."))
SINGLE_SYSTEM = "너는 딥리서치 에이전트다. 장문 보고서 전체를 혼자 조사하고 쓴다.\n\n" + SINGLE_SYSTEM


def _trim_old_results(msgs: list, keep_last: int = 6) -> int:
    """컨텍스트 초과 시 응급 처치: 오래된 도구 결과 본문을 비운다(마지막 keep_last개는 남김). 비운 개수를 돌려준다."""
    slots = []
    for m in msgs:
        if m.get("role") == "tool":
            slots.append((m, "content"))
        elif m.get("type") == "function_call_output":
            slots.append((m, "output"))
        elif m.get("role") == "user" and isinstance(m.get("content"), list):
            slots += [(b, "content") for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
    cut = slots[:-keep_last] if len(slots) > keep_last else []
    for d, k in cut:
        d[k] = "[컨텍스트 한도로 생략됨 — 이미 읽은 내용]"
    return len(cut)


def agent_loop(llm, system: str, prompt: str, tb: Toolbox, max_turns: int, role: str, max_tokens: int,
               meter: Meter, s: Settings) -> dict:
    """도구 루프 (공급자 무관 — llm은 Reply를 돌려주고 assistant_msg/result_msgs로 대화를 잇는다).
    모든 에이전트에 같은 규칙:
    - 예산의 min_budget_use 미만으로 제출하면 최대 2번 되돌려 더 읽게 한다(대조군을 묶어 두지 않도록).
    - 컨텍스트가 85%를 넘으면 더 읽지 못하게 막고 context_limited로 기록한다.
    - 그래도 컨텍스트를 넘으면 오래된 도구 결과를 비우고 한 번만 다시 시도해 제출을 받는다(trimmed로 기록)."""
    meter.see_sub(tb.agent_id, system + prompt)
    msgs: list = [{"role": "user", "content": prompt}]
    turns = nudges = trimmed = 0
    told = limited = False
    rep = None
    while True:
        try:
            rep = llm.chat(role=role, system=system, messages=msgs, tools=TOOLS, max_tokens=max_tokens)
        except ContextFull:
            limited = True
            if trimmed:
                break
            trimmed = _trim_old_results(msgs)
            msgs.append({"role": "user", "content": "컨텍스트 한도에 닿았다. 더 읽지 말고 지금 submit_section 하라."})
            continue
        am = llm.assistant_msg(rep)
        msgs += am if isinstance(am, list) else [am]
        limited = limited or rep.n_input > 0.85 * s.context_tokens
        if rep.stop == "pause":
            continue
        if not rep.calls:
            if tb.submitted or told:
                break
            msgs.append({"role": "user", "content": "submit_section 도구로 원고를 제출해 작업을 끝내라."})
            told = True
            continue
        turns += 1
        results = []
        for c in rep.calls:
            if c.name == "read_doc" and limited:
                out, err = "컨텍스트 한도에 가까워 더 읽을 수 없다. submit_section 하라.", True
            elif (c.name == "submit_section" and nudges < 2 and not limited and turns < max_turns
                  and tb.used < s.min_budget_use * tb.budget and tb.remaining() > CHUNK):
                nudges += 1
                out, err = (f"아직 읽기 예산의 {tb.used / tb.budget:.0%}만 썼다(남은 {tb.remaining():,}자). "
                            "관련 자료를 더 읽고 보강한 뒤 다시 제출하라."), True
            else:
                out, err = tb.run(c.name, c.input)
            if c.name != "submit_section" and turns >= max_turns - 1:
                out += f"\n\n[알림] 도구 라운드 {turns}/{max_turns}. 다음 응답에서 submit_section을 호출하라."
            meter.see_sub(tb.agent_id, out)
            results.append((c.id, out, err))
        msgs += llm.result_msgs(results)
        if tb.submitted or turns > max_turns:
            break
    sub = tb.submitted or {"markdown": (rep.text if rep else "") or "(원고 미제출)", "sufficient": False,
                           "gaps": ["턴 한도 내 미제출"], "unsubmitted": True}
    return {"turns": turns, "read_log": tb.read_log, "read_ids": tb.read_ids, "search_log": tb.search_log,
            "budget": tb.budget, "used": tb.used, "submitted": sub, "nudges": nudges, "context_limited": limited,
            "context_trimmed": trimmed, "chars_seen": meter.sub.get(tb.agent_id, 0)}


# ================================================================ ① 기획
FORMAT_SECTIONS = [("개요", "주제 전체의 개요와 핵심 개념"), ("연표", "시간 순서로 본 변화"),
                   ("기술 분석", "현재 구조의 기술적 분석"), ("대안 비교", "경쟁하거나 밀려난 대안 비교"),
                   ("전망과 시사점", "남은 문제와 앞으로의 방향")]

PLAN_SCHEMA = {"type": "object", "properties": {"sections": {"type": "array", "items": {
    "type": "object", "properties": {
        "title": {"type": "string"},
        "role": {"type": "string", "description": "역할 명단의 이름 중 하나"},
        "scope": {"type": "string", "description": "이 절이 맡는 내용 단위와 경계"},
        "questions": {"type": "array", "items": {"type": "string"}},
        "seed_doc_ids": {"type": "array", "items": {"type": "string"}},
        "search_hints": {"type": "array", "items": {"type": "string"}},
        "budget_weight": {"type": "integer", "description": "1~5, 읽을 자료가 많을수록 크게"}},
    "required": ["title", "role", "scope", "questions", "seed_doc_ids", "search_hints", "budget_weight"],
    "additionalProperties": False}}}, "required": ["sections"], "additionalProperties": False}


def _norm(i: str) -> str:
    return i.strip().strip("[]").removeprefix("D:").strip()


def validate_plan(corpus: Corpus, secs: list[dict], roster: set[str]) -> list[str]:
    """배정이 실제 자료와 명단을 가리키는지 코드가 검사한다 (모델은 그럴듯한 제목을 지어낸다)."""
    p = []
    for s in secs:
        bad = [i for i in s["seed_doc_ids_raw"] if _norm(i) not in corpus.text]
        if bad:
            p.append(f"「{s['title']}」: 목록에 없는 문서 {bad}")
        if not s["seed_doc_ids"]:
            p.append(f"「{s['title']}」: 유효한 시작 자료가 없음")
        if s["role"] not in roster:
            p.append(f"「{s['title']}」: 명단에 없는 역할 '{s['role']}'")
    return p


def plan_sections(llm, corpus: Corpus, q: dict, s: Settings, meter: Meter) -> tuple[list[dict], dict]:
    roster = {r["name"]: r["focus"] for r in CONFIG["roles"]}
    if s.split_by == "format":
        rule = "목차는 아래 형식 절을 그대로 쓴다:\n" + "\n".join(f"- {t}: {d}" for t, d in FORMAT_SECTIONS[: s.n_sections])
    else:
        rule = ("목차는 '내용 단위'(서로 다른 대상·갈래·단계)로 나눈다. 절마다 서로 다른 문서 묶음을 1차 자료로 가져야 한다"
                "(시작 자료가 겹치지 않게). '개요/연표/분석/전망' 같은 형식 단위로 나누지 마라 — 그러면 모두가 같은 자료를 읽는다. "
                "한 대상을 따라가는 질문이면 단계·측면별로 나눈다.")
    head = (f"질문:\n{q['question']}\n\n절 수: {s.n_sections}\n\n{rule}\n\n역할 명단(절마다 하나를 고른다):\n"
            + "\n".join(f"- [역할] {n}: {f}" for n, f in roster.items())
            + "\n\n절마다 시작 자료 2~6개를 아래 목록의 id 그대로(대괄호·D: 없이), 검색 힌트 2~4개, 예산 가중치(1~5)를 준다.\n\n"
              "코퍼스 목록 (원문 아님 — id·길이·링크 수만):\n")
    cat = corpus.catalog()
    meter.see_coord("지시·질문", head)
    meter.see_coord("목록", cat)
    msgs = [{"role": "user", "content": head + cat}]
    info = {"attempts": 0, "problems": []}
    for _ in range(2):  # 지어낸 배정이 있으면 짚어서 한 번 되묻는다
        info["attempts"] += 1
        rep = llm.chat(role="coord:plan", system="너는 딥리서치 코디네이터다. 목차를 짜고 분업을 배정한다.",
                        messages=msgs, json_schema=PLAN_SCHEMA, max_tokens=16000)
        secs = json.loads(rep.text)["sections"][: s.n_sections]
        for k, sec in enumerate(secs):
            sec["id"] = f"s{k + 1}"
            sec["seed_doc_ids_raw"] = list(sec["seed_doc_ids"])
            sec["seed_doc_ids"] = list(dict.fromkeys(_norm(i) for i in sec["seed_doc_ids"] if _norm(i) in corpus.text))
        problems = validate_plan(corpus, secs, set(roster))
        info["problems"].append(problems)
        if not problems:
            break
        fix = "다음 배정이 목록·명단에 없는 것을 가리킨다. 목록의 id와 명단의 역할만 써서 다시 답하라:\n" + "\n".join(problems)
        meter.see_coord("지시·질문", fix)
        am = llm.assistant_msg(rep)
        msgs += (am if isinstance(am, list) else [am]) + [{"role": "user", "content": fix}]
    info["invalid_final"] = len(validate_plan(corpus, secs, set(roster)))
    for sec in secs:
        if sec["role"] not in roster:
            sec["role"] = next(iter(roster))
    if s.split_by == "format":
        for sec, (t, d) in zip(secs, FORMAT_SECTIONS):
            sec["title"], sec["scope"] = t, d
    w = [min(5, max(1, int(x.get("budget_weight", 3)))) for x in secs]
    for sec, wi in zip(secs, w):  # 1차 총예산을 가중치대로 나눈다
        sec["budget_weight"] = wi
        sec["budget_chars"] = s.base_budget * wi // sum(w)
        sec["budget_turns"] = max(6, s.section_turns * len(secs) * wi // sum(w))
    if s.assign_orphans:  # 링크로 닿지 않는 문서 → 가장 관련 높은 절에 강제 배정
        have = {i for x in secs for i in x["seed_doc_ids"]}
        scores = {x["id"]: dict(corpus.search(" ".join([x["title"], x["scope"], *x["questions"]]), k=len(corpus.ids)))
                  for x in secs}
        for k, doc in enumerate(o for o in corpus.orphans() if o not in have):
            best = max(secs, key=lambda x: scores[x["id"]].get(doc, 0))
            if not scores[best["id"]].get(doc):
                best = secs[k % len(secs)]
            best["seed_doc_ids"].append(doc)
            best.setdefault("assigned_orphans", []).append(doc)
    return secs, info


# ================================================================ ② 서브에이전트
def researcher_prompt(q: dict, sec: dict, secs: list[dict], s: Settings, budget: int, turns: int,
                      prior: dict | None) -> str:
    orph = set(sec.get("assigned_orphans", []))
    seeds = "\n".join(f"- [D:{i}]" + ("  ← 링크로는 닿지 않는 문서, 직접 확인할 것" if i in orph else "")
                      for i in sec["seed_doc_ids"]) or "- (없음)"
    p = (f"## 보고서 전체 질문\n{q['question']}\n\n## 네가 맡은 절: {sec['id']} 「{sec['title']}」\n범위: {sec['scope']}\n"
         "맡은 질문:\n" + "\n".join(f"- {x}" for x in sec["questions"])
         + f"\n\n## 시작 자료\n{seeds}\n검색 힌트: {', '.join(sec['search_hints'])}")
    if s.peer_scopes:
        p += ("\n\n## 다른 리서처들의 구역 (남의 구역)\n"
              + "\n".join(f"- {x['id']} 「{x['title']}」({x['role']}): {x['scope']}" for x in secs if x["id"] != sec["id"])
              + "\n이 구역들은 해당 절이 다룬다. 접점은 한두 문장으로만 언급하라.")
    p += f"\n\n## 예산\n읽기 {budget:,}자, 도구 라운드 최대 {turns}회."
    if prior:
        p += (f"\n\n## 재조사 요청 ({prior['round']}번째 바퀴)\n부족하다고 신고된 점:\n"
              + "\n".join(f"- {g}" for g in prior["gaps"])
              + f"\n이 절이 이미 읽은 문서(다시 안 읽어도 인용 가능): {', '.join(prior['read_ids'][:60])}\n"
              f"이전 원고:\n<<<\n{prior['markdown']}\n>>>\n부족한 점을 채워 절 전체를 다시 써서 제출하라. 이전 인용은 빠뜨리지 마라.")
    return p


def valid_cites(md: str, read: set[str], corpus: Corpus) -> tuple[set[str], set[str]]:
    c = set(CITE.findall(md))
    ok = {x for x in c if x in corpus.text and x in read}
    return ok, c - ok


def choose_draft(old_md: str, new_md: str, read: set[str], corpus: Corpus) -> dict:
    """두 번째 원고 처리. 무조건 덮어쓰면 인용을 빠뜨린 원고가 멀쩡한 원고를 지운다.
    교체 조건(모두): 이전 유효 인용 70% 이상 유지 · 유효 인용 수가 줄지 않음 · 무효 인용이 늘지 않음.
    아니면 이전 원고를 지키고, 새 원고에서 새로 생긴 유효 인용이 든 줄만 '재조사 보충'으로 덧붙인다."""
    ov, oi = valid_cites(old_md, read, corpus)
    nv, ni = valid_cites(new_md, read, corpus)
    kept = len(ov & nv) / len(ov) if ov else 1.0
    d = {"old_valid": len(ov), "new_valid": len(nv), "old_invalid": len(oi), "new_invalid": len(ni), "kept": round(kept, 2)}
    if kept >= 0.7 and len(nv) >= len(ov) and len(ni) <= len(oi):
        return {**d, "decision": "replace", "markdown": new_md, "reason": "이전 유효 인용을 지키며 늘렸고 무효 인용이 늘지 않음"}
    extra = [ln for ln in new_md.splitlines() if any(f"[D:{c}]" in ln for c in nv - ov)]
    md = old_md.rstrip() + ("\n\n### 재조사 보충\n\n" + "\n".join(extra) if extra else "")
    return {**d, "decision": "keep+append" if extra else "keep", "markdown": md,
            "reason": f"이전 유효 인용 유지 {kept:.0%} · 유효 {len(ov)}→{len(nv)} · 무효 {len(oi)}→{len(ni)}이라 교체하지 않음"
                      + (f"; 새 유효 인용 {len(nv - ov)}개가 든 줄만 덧붙임" if extra else "")}


# ================================================================ ④ 종합
def synthesize(llm, q: dict, secs: list[dict], drafts: dict[str, str], s: Settings, corpus: Corpus,
               meter: Meter) -> tuple[str, dict]:
    body = "\n".join(f"## {x['title']}\n\n{drafts[x['id']].strip()}\n" for x in secs)
    info = {"mode": s.synth, "new_cites": []}
    if s.synth == "bridge":
        instr = (f"질문:\n{q['question']}\n\n아래는 리서처들의 절 원고다. 원고는 고치지 않는다. 너는 '### 서론'(질문과 절들의 범위를 잇는 "
                 "지도)과 '### 결론'(절들을 가로지르는 공통 패턴과 대안 비교)만 쓴다. 원고에 있는 [D:id] 인용만 재사용하고 새 사실을 만들지 마라.\n\n")
        meter.see_coord("지시·질문", instr)
        meter.see_coord("원고", body)
        bridge = (llm.chat(role="coord:synth", system="너는 장문 보고서의 편집자다.",
                                  messages=[{"role": "user", "content": instr + body}], max_tokens=16000)).text
        info["new_cites"] = sorted(set(CITE.findall(bridge)) - set(CITE.findall(body)))
        intro, _, concl = bridge.partition("### 결론")
        report = (f"# {q['title']}\n\n> {q['question']}\n\n## 서론\n\n{intro.replace('### 서론', '').strip()}\n\n"
                  f"{body}\n## 결론\n\n{concl.strip()}\n")
    else:
        report = f"# {q['title']}\n\n> {q['question']}\n\n{body}"
    return report + "\n\n## 참고 자료\n\n" + references(report, corpus), info


def references(md: str, corpus: Corpus) -> str:
    return "\n".join(f"- [D:{i}] {corpus.url(i)}" if i in corpus.text else f"- [D:{i}] (코퍼스에 없음)"
                     for i in dict.fromkeys(CITE.findall(md)))


# ================================================================ LangGraph
def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class State(TypedDict, total=False):
    question: dict
    plan: list[dict]
    plan_info: dict
    round: int                                        # 지금 막 끝난 파견 바퀴 (1 = 1차)
    pool_left: int                                    # 남은 재파견 예산 풀
    pending: list[dict]                               # check가 고른 재파견 과제
    traces: Annotated[list[dict], operator.add]       # 에이전트별 작업 기록 (위로 올라가는 것은 원고뿐)
    drafts: Annotated[dict, _merge]                   # 절 id → 현재 원고
    read_sets: Annotated[dict, _merge]                # 절 id → 그 절이 지금까지 읽은 문서
    decisions: Annotated[list[dict], operator.add]    # 두 번째 원고 처리 기록
    events: Annotated[list[dict], operator.add]
    report: str
    synth_info: dict


class Task(TypedDict):
    question: dict
    peers: list[dict]
    sec: dict
    budget: int
    turns: int
    round: int
    prior: dict | None


def build_graph(s: Settings, corpus: Corpus, llm, meter: Meter, log=print):
    def plan_node(st: State) -> dict:
        secs, info = plan_sections(llm, corpus, st["question"], s, meter)
        for x in secs:
            log(f"① {x['id']} 「{x['title']}」 {x['role']} · 시작 자료 {len(x['seed_doc_ids'])} · 예산 {x['budget_chars']:,}자")
        return {"plan": secs, "plan_info": info, "round": 1, "pool_left": s.recheck_pool,
                "events": [{"event": "plan", "split_by": s.split_by, "attempts": info["attempts"],
                            "orphans_assigned": sum(len(x.get("assigned_orphans", [])) for x in secs)}]}

    def fan_out(st: State) -> list[Send]:
        log(f"② 리서처 {len(st['plan'])}명 동시 파견")
        return [Send("researcher", Task(question=st["question"], peers=st["plan"], sec=x, budget=x["budget_chars"],
                                        turns=x["budget_turns"], round=1, prior=None)) for x in st["plan"]]

    def researcher_node(t: Task) -> dict:
        sec, r = t["sec"], t["round"]
        aid = f"{sec['id']}-r{r}"
        role = next(x for x in CONFIG["roles"] if x["name"] == sec["role"])
        tb = Toolbox(corpus, t["budget"], aid)
        prompt = researcher_prompt(t["question"], sec, t["peers"], s, t["budget"], t["turns"], t["prior"])
        tr = agent_loop(llm, RESEARCHER_SYSTEM.format(role=role["name"], focus=role["focus"]), prompt, tb, t["turns"],
                        f"researcher:{aid}", 32000, meter, s)
        tr.update(agent_id=aid, section=sec["id"], round=r, prompt=prompt)
        read = set(tr["read_ids"]) | set((t["prior"] or {}).get("read_ids", []))
        out = {"traces": [tr], "read_sets": {sec["id"]: sorted(read)},
               "events": [{"event": "dispatch" if r == 1 else "recheck_dispatch", "section": sec["id"], "round": r}]}
        if t["prior"] is None:
            out["drafts"] = {sec["id"]: tr["submitted"]["markdown"]}
        else:
            d = choose_draft(t["prior"]["markdown"], tr["submitted"]["markdown"], read, corpus)
            out["drafts"] = {sec["id"]: d.pop("markdown")}
            out["decisions"] = [{"section": sec["id"], "round": r, **d}]
            log(f"③ {sec['id']} {r}번째 원고: {d['decision']} — {d['reason']}")
        log(f"② {aid} {len(tr['read_ids'])}문서/{tr['used']:,}자 읽음 · 충분={tr['submitted']['sufficient']}")
        return out

    def check_node(st: State) -> dict:
        latest: dict[str, dict] = {}
        for tr in st["traces"]:
            latest[tr["section"]] = tr if tr["round"] >= latest.get(tr["section"], {"round": 0})["round"] else latest[tr["section"]]
        weak = [x for x in st["plan"] if not latest[x["id"]]["submitted"]["sufficient"]]
        done_rounds = st["round"] - 1
        used_last = sum(tr["used"] for tr in st["traces"] if tr["round"] == st["round"] and st["round"] > 1)
        pool = st["pool_left"] - used_last
        go = s.recheck and bool(weak) and done_rounds < s.max_rounds and pool > 0
        pending = []
        if go:
            left = s.max_rounds - done_rounds
            per = pool // left // len(weak)
            tn = max(4, (s.total_turns - s.section_turns * s.n_sections) // s.max_rounds // len(weak))
            log(f"③ {st['round'] + 1}번째 바퀴: {[x['id'] for x in weak]} 재파견 (각 {per:,}자 / {tn}턴)")
            for x in weak:
                pending.append(Task(question=st["question"], peers=st["plan"], sec=x, budget=per, turns=tn, round=st["round"] + 1, prior={
                    "gaps": latest[x["id"]]["submitted"]["gaps"], "markdown": st["drafts"][x["id"]],
                    "read_ids": st["read_sets"][x["id"]], "round": st["round"] + 1}))
        return {"pending": pending, "pool_left": pool, "round": st["round"] + (1 if go else 0),
                "events": [{"event": "check", "after_round": st["round"], "weak": [x["id"] for x in weak], "dispatched": go}]}

    def route_check(st: State):
        return [Send("researcher", t) for t in st["pending"]] if st["pending"] else "synth"

    def synth_node(st: State) -> dict:
        report, info = synthesize(llm, st["question"], st["plan"], st["drafts"], s, corpus, meter)
        log("④ 종합 완료")
        return {"report": report, "synth_info": info, "events": [{"event": "synth", "mode": s.synth}]}

    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("researcher", researcher_node)
    g.add_node("check", check_node)
    g.add_node("synth", synth_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["researcher"])
    g.add_edge("researcher", "check")
    g.add_conditional_edges("check", route_check, ["researcher", "synth"])
    g.add_edge("synth", END)
    app = g.compile()

    def invoke(q: dict) -> State:
        return app.invoke({"question": q}, config={"max_concurrency": s.max_parallel, "recursion_limit": 50})

    invoke.graph = app
    return invoke


def mermaid() -> str:
    """REPORT.md에 넣는 구조도."""
    s = Settings(backend="mock")
    return build_graph(s, None, None, Meter()).graph.get_graph().draw_mermaid()


# ================================================================ 실행 · 기록
def leak_check(corpus: Corpus, traces: list[dict], meter: Meter) -> dict:
    """리서처가 읽은 원문 조각이 코디네이터 입력(원고 제외)에 들어갔는가 → 0이어야 한다."""
    other = "\n".join(t for k, t in meter.coord if k != "원고")
    ms = "\n".join(t for k, t in meter.coord if k == "원고")
    n = leak = verb = 0
    for tr in traces:
        for r in tr["read_log"]:
            mid = r["offset"] + r["n"] // 2
            snip = " ".join(corpus.text[r["doc_id"]][mid: mid + 80].split())
            if len(snip) >= 40:
                n += 1
                leak += snip in other
                verb += snip in ms
    return {"snippets_checked": n, "coordinator_raw_leak": leak, "manuscript_verbatim": verb}


def save_run(s: Settings, q: dict, corpus: Corpus, ledger: Ledger, meter: Meter, result: dict, elapsed: float) -> Path:
    out = ROOT / CONFIG["output_dir"] / "reports" / s.run_name()
    out.mkdir(parents=True, exist_ok=True)
    trace = {"run": s.run_name(), "question": q, "elapsed_s": elapsed, "plan": result.get("plan", []),
             "plan_info": result.get("plan_info", {}), "agents": result["traces"], "drafts": result.get("drafts", {}),
             "decisions": result.get("decisions", []), "events": result.get("events", []),
             "synth": result.get("synth_info", {}),
             "isolation": {**meter.summary(), **leak_check(corpus, result["traces"], meter)},
             "corpus": {"docs": len(corpus.ids), "chars": corpus.total_chars(), "orphans": corpus.orphans()}}
    (out / "report.md").write_text(result["report"], encoding="utf-8")
    (out / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "ledger.json").write_text(json.dumps({"calls": ledger.calls, "total": ledger.totals()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(s.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def run(s: Settings, corpus: Corpus | None = None, log=print) -> Path:
    """멀티 에이전트 실행. 단일 에이전트는 baseline.run_single."""
    if not s.multi_agent:
        from baseline import run_single
        return run_single(s, corpus, log)
    corpus = corpus or Corpus()
    q = resolve_question(s)
    ledger, meter = Ledger(), Meter()
    t0 = time.time()
    log(f"코퍼스 {len(corpus.ids)}문서 · {corpus.total_chars():,}자 · 질문: {q['title']}")
    result = build_graph(s, corpus, make_llm(s, ledger), meter, log)(q)
    out = save_run(s, q, corpus, ledger, meter, result, round(time.time() - t0, 1))
    iso = json.loads((out / "trace.json").read_text(encoding="utf-8"))["isolation"]
    log(f"완료 → {out.relative_to(ROOT)} · 코디네이터 {iso['coordinator_chars']:,}자 / 서브에이전트 "
        f"{iso['subagent_chars_total']:,}자 · 원문 누출 {iso['coordinator_raw_leak']}")
    return out


def cli_settings(ap: argparse.ArgumentParser | None = None) -> tuple[Settings, argparse.Namespace]:
    load_dotenv()
    ap = ap or argparse.ArgumentParser()
    ap.add_argument("--question", type=int, default=3, help="data/questions.json의 id")
    ap.add_argument("--ask", default="", help="직접 입력하는 질문")
    ap.add_argument("--arm", default="full", choices=list(ARMS))
    ap.add_argument("--backend", default="api", choices=["api", "mock"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = Settings(question_id=a.question, question=a.ask, backend=a.backend, seed=a.seed).replace(**ARMS[a.arm])
    return s, a


if __name__ == "__main__":
    s, _ = cli_settings()
    d = run(s)
    from metrics import record_run
    print(json.dumps(record_run(d)["alarms"], ensure_ascii=False))
