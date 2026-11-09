"""단일 에이전트 대조군 — 예산을 맞춘다.

같은 자료(Corpus) · 같은 모델(Settings.model) · 같은 도구 4종(graph.TOOLS) · 같은 에이전트 루프(graph.agent_loop),
그리고 멀티 에이전트가 쓸 수 있는 최대치와 같은 읽기 예산(1차 총예산 + 재파견 풀)과 총 턴을 준다.
'예산의 80% 미만으로 제출하면 되돌려 더 읽게 하는' 규칙도 리서처와 똑같이 적용된다.
컨텍스트가 차서 더 못 읽은 경우는 trace의 context_limited로 따로 표시한다 — 묶어 둔 것과 구별하기 위해.

실행:  python baseline.py --question 3 [--backend mock]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from graph import (SINGLE_SYSTEM, Corpus, Ledger, Meter, Settings, Toolbox, agent_loop, cli_settings, leak_check,
                   make_llm, references, resolve_question, save_run)


def run_single(s: Settings, corpus: Corpus | None = None, log=print) -> Path:
    s = s.replace(multi_agent=False)
    corpus = corpus or Corpus()
    q = resolve_question(s)
    ledger, meter = Ledger(), Meter()
    llm = make_llm(s, ledger)
    budget, turns = s.total_budget, s.total_turns
    log(f"단일 에이전트: 예산 {budget:,}자 / {turns}턴 (멀티 최대치와 같음)")
    t0 = time.time()
    tb = Toolbox(corpus, budget, "single")
    prompt = (f"## 질문\n{q['question']}\n\n## 목표\n약 {s.n_sections}개 절의 장문 보고서. 절은 내용 단위로 나눈다.\n\n"
              f"## 예산\n읽기 {budget:,}자, 도구 라운드 최대 {turns}회. 마지막에 submit_section(markdown=보고서 전체).")
    tr = agent_loop(llm, SINGLE_SYSTEM, prompt, tb, turns, "single", min(128000, 32000 * s.n_sections), meter, s)
    tr.update(agent_id="single", section="all", round=1, prompt=prompt)
    log(f"단일 에이전트 완료: {tr['used']:,}/{budget:,}자 ({tr['used'] / budget:.0%}) · 문서 {len(tr['read_ids'])}개"
        + (" · 컨텍스트 한도로 중단" if tr["context_limited"] else ""))
    body = tr["submitted"]["markdown"]
    report = f"# {q['title']}\n\n> {q['question']}\n\n{body}\n\n## 참고 자료\n\n{references(body, corpus)}"
    result = {"report": report, "traces": [tr], "events": [{"event": "single_dispatch", "budget": budget, "turns": turns}]}
    return save_run(s, q, corpus, ledger, meter, result, round(time.time() - t0, 1))


if __name__ == "__main__":
    s, _ = cli_settings()
    d = run_single(s)
    from metrics import record_run
    print(json.dumps(record_run(d)["alarms"], ensure_ascii=False))
