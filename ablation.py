"""스위치를 하나씩 끄고 평가하는 실험.

같은 질문으로 보고서 여러 편: 기본(full) · 장치 하나 끈 것 · 단일 에이전트(single).
- 경보가 0이 아닌 실행은 비교에서 뺀다(명백한 실패 거르기). 지표로 순위를 매기지 않는다.
- 같은 설정을 trials번 돌려 시행별 값을 그대로 남긴다(같은 설정도 시행마다 흔들린다).
- 문장 분리 버전이 섞이면 거부한다.
- 어느 쪽이 나은지는 사람이 나란히 읽고 output/compare_notes.md(데모의 비교 탭)에 적는다.

실행:  python ablation.py --question 3 --arms full,no_peer,single --trials 2 [--backend mock]
결과:  output/ablation.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st

from dotenv import load_dotenv

from graph import ARMS, CONFIG, ROOT, Corpus, Settings, run
from metrics import METRICS, record_run


def summarize(results: dict[str, list[dict]]) -> dict:
    out = {}
    for arm, ms in results.items():
        kept = [m for m in ms if m["alarms_ok"]]
        sig = {}
        for met in METRICS:
            if met.kind != "signal":
                continue
            xs = [m["signals"][met.name] for m in kept if m["signals"][met.name] is not None]
            sig[met.name] = {"values": xs, "mean": st.mean(xs) if xs else None,
                             "sd": st.stdev(xs) if len(xs) > 1 else None}
        out[arm] = {"runs": [m["run"] for m in ms], "excluded": [m["run"] for m in ms if not m["alarms_ok"]],
                    "alarms": {m["run"]: {k: v for k, v in m["alarms"].items() if v} for m in ms},
                    "budget_use": {m["run"]: [round(b["used"] / b["budget"], 2) for b in m["budget"]] for m in ms},
                    "context_limited": {m["run"]: any(b["context_limited"] for b in m["budget"]) for m in ms},
                    "signals": sig}
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=int, default=3)
    ap.add_argument("--arms", default="full,no_peer,single")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--backend", default="anthropic", choices=["anthropic", "mock"])
    a = ap.parse_args()
    corpus = Corpus()
    arms = a.arms.split(",")
    results: dict[str, list[dict]] = {x: [] for x in arms}
    for seed in range(a.trials):
        for arm in arms:
            s = Settings(question_id=a.question, backend=a.backend, seed=seed).replace(**ARMS[arm])
            d = ROOT / CONFIG["output_dir"] / "reports" / s.run_name()
            if not (d / "report.md").exists():  # 중단 후 재개
                d = run(s, corpus)
            results[arm].append(record_run(d, corpus))
    versions = {m["splitter_version"] for ms in results.values() for m in ms}
    if len(versions) > 1:
        raise SystemExit(f"문장 분리 버전이 섞여 있음 {versions} — metrics.py로 다시 평가한 뒤 비교하라.")

    path = ROOT / CONFIG["output_dir"] / "ablation.json"
    allx = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = f"q{a.question}__{a.backend}"
    allx[key] = {"question_id": a.question, "arms": arms, "trials": a.trials, "splitter_version": versions.pop(),
                 "summary": summarize(results)}
    path.write_text(json.dumps(allx, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n== {key} — 경보로 거른 뒤 신호(평균, 시행별 값). 순위가 아니다 ==")
    for arm, v in allx[key]["summary"].items():
        print(f"[{arm}] 제외 {len(v['excluded'])}건 · 예산 사용률 {list(v['budget_use'].values())}")
        for name, x in v["signals"].items():
            if x["mean"] is not None:
                print(f"   {name:18s} {x['mean']:.3g}  {['%.3g' % y for y in x['values']]}")


if __name__ == "__main__":
    main()
