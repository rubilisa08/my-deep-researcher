"""딥리서처 데모 — streamlit run app.py

보고서만이 아니라 '절마다 누가 무엇을 읽고 무엇을 썼는지'를 보여 준다.
API 키는 환경변수 OPENAI_API_KEY(또는 ANTHROPIC_API_KEY)나 .env로만 받는다(저장소에 올리지 않는다).
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from graph import ARMS, CITE, CONFIG, ROOT, Corpus, Settings, load_questions, run, tokenize
from metrics import BY_NAME, METRICS, citation_rows, evaluate, load, record_run, report_sections

load_dotenv()
st.set_page_config(page_title="딥리서처", layout="wide")
REPORTS = ROOT / CONFIG["output_dir"] / "reports"


@st.cache_resource
def get_corpus() -> Corpus:
    return Corpus()


def runs() -> list[str]:
    if not REPORTS.exists():
        return []
    ds = [d for d in REPORTS.iterdir() if (d / "trace.json").exists()]
    return [d.name for d in sorted(ds, key=lambda d: d.stat().st_mtime, reverse=True)]


def metrics_of(d: Path) -> dict:
    p = d / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else evaluate(d, get_corpus())


corpus = get_corpus()
QS = load_questions()
KIND = {"single": "한 건", "scatter": "여러 갈래", "trace": "한 대상", "custom": "직접 입력"}

# ================================================================ 사이드바: 질문하기
with st.sidebar:
    st.header("질문하기")
    st.caption(f"코퍼스: 한국어 위키백과 {len(corpus.ids)}문서 · {corpus.total_chars():,}자 · "
               f"링크 {sum(len(v) for v in corpus.links.values()):,}개")
    mode = st.radio("질문", ["질문 세트", "직접 입력"], horizontal=True)
    qid, ask = 3, ""
    if mode == "질문 세트":
        qid = st.selectbox("질문 세트", range(len(QS)), index=3,
                           format_func=lambda i: f"[{i}] {KIND[QS[i]['kind']]} · {QS[i]['title']}")
        st.info(QS[qid]["question"])
        st.caption(f"왜 나눌 만한가: {QS[qid]['why']}")
    else:
        ask = st.text_area("질문", placeholder="코퍼스(인공지능·기계 학습 위키 문서)로 답할 수 있는 질문")
    arm = st.selectbox("실험 팔", list(ARMS), help="full=기본 · 나머지=장치 하나 끈 것 · single=단일 에이전트 대조군")
    has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    backend = st.selectbox("백엔드", ["api", "mock"] if has_key else ["mock", "api"],
                           help=f"api = {CONFIG['provider']} {CONFIG['model']} · mock = API 없이 배선만 확인하는 가짜 응답")
    seed = st.number_input("시행 번호", 0, 99, 0)
    if st.button("실행", type="primary", width="stretch"):
        if mode == "직접 입력" and not ask.strip():
            st.warning("질문을 입력하세요")
        else:
            s = Settings(question_id=qid, question=ask.strip(), backend=backend, seed=int(seed)).replace(**ARMS[arm])
            with st.status("실행 중…", expanded=True) as box:
                d = run(s, corpus, log=st.write)
                record_run(d, corpus)
                box.update(label=f"완료: {d.name}", state="complete")
            st.session_state["run"] = d.name

# ================================================================ 본문
rs = runs()
if not rs:
    st.title("딥리서처")
    st.write("왼쪽에서 질문을 골라 실행하세요.")
    st.stop()
name = st.selectbox("실행 결과", rs, index=rs.index(st.session_state["run"]) if st.session_state.get("run") in rs else 0)
R = load(REPORTS / name)
T, C = R["trace"], R["config"]
M = metrics_of(REPORTS / name)
single = T["agents"][0]["agent_id"] == "single"

st.title(T["question"]["title"])
st.caption(f"> {T['question']['question']}")
st.caption(f"`{name}` · 팔 **{Settings(**C).arm()}** · 질문 성격 {KIND.get(T['question']['kind'], '-')} · {T['elapsed_s']}s")
if C["backend"] == "mock":
    st.warning("mock 백엔드 결과입니다 — 배선 확인용 가짜 응답이라 글 내용은 의미가 없습니다.")
iso = T["isolation"]
bad = {k: v for k, v in M["alarms"].items() if v}
c = st.columns(5)
c[0].metric("경보", "모두 0" if not bad else f"{len(bad)}개 켜짐")
c[1].metric("코디네이터가 본 글자", f"{iso['coordinator_chars']:,}")
c[2].metric("서브에이전트들이 본 글자", f"{iso['subagent_chars_total']:,}")
c[3].metric("원문 누출", iso["coordinator_raw_leak"])
c[4].metric("인용한 문서", M["signals"]["unique_cited"])
if bad:
    st.error("켜진 경보: " + " · ".join(f"{k}={v} ({BY_NAME[k].desc})" for k, v in bad.items()))

tab = st.tabs(["절별 작업", "보고서", "격리·예산", "지표", "인용 대조", "비교"])

# ---------------------------------------------------------------- 절별 작업 (이 데모의 요점)
with tab[0]:
    st.caption("절마다 누가(역할) 무엇을 배정받아 무엇을 읽고 무엇을 썼는지. 리서처가 올린 원고와 최종본을 나란히 놓는다.")
    by: dict[str, list] = {}
    for a in T["agents"]:
        by.setdefault(a["section"], []).append(a)
    finals = report_sections(R["report"])
    dec = {(d["section"], d["round"]): d for d in T.get("decisions", [])}
    secs = [{"id": "all", "title": "단일 에이전트(보고서 전체)", "role": "-", "scope": "전체", "questions": [],
             "seed_doc_ids": []}] if single else T["plan"]
    for x in secs:
        ags = by.get(x["id"], [])
        with st.expander(f"{x['id']} 「{x['title']}」 · {x['role']} · 파견 {len(ags)}회 · "
                         f"{sum(a['used'] for a in ags):,}자 읽음", expanded=x is secs[0]):
            left, right = st.columns([2, 3])
            with left:
                st.markdown(f"**범위** {x['scope']}")
                if x["questions"]:
                    st.markdown("**맡은 질문**\n" + "\n".join(f"- {q}" for q in x["questions"]))
                if x["seed_doc_ids"]:
                    orph = set(x.get("assigned_orphans", []))
                    st.markdown("**배정된 시작 자료**\n" + "\n".join(
                        f"- {i}" + (" · *링크로 닿지 않아 강제 배정*" if i in orph else "") for i in x["seed_doc_ids"]))
                    st.caption(f"배정 예산 {x['budget_chars']:,}자 (가중치 {x['budget_weight']}) · {x['budget_turns']}턴")
                for a in ags:
                    st.markdown(f"---\n**{a['agent_id']}** · {a['round']}번째 바퀴 · {a['used']:,}/{a['budget']:,}자 "
                                f"({a['used'] / max(1, a['budget']):.0%}) · 도구 {a['turns']}라운드 · 본 글자 {a['chars_seen']:,}"
                                + (" · **컨텍스트 한도로 중단**" if a["context_limited"] else "")
                                + (f" · 되돌림 {a['nudges']}회" if a["nudges"] else ""))
                    if a["search_log"]:
                        st.markdown("검색어: " + " · ".join(f"`{q}`" for q in a["search_log"]))
                    if a["read_log"]:
                        st.dataframe(pd.DataFrame([{"읽은 문서": r["doc_id"], "offset": r["offset"], "글자": r["n"]}
                                                   for r in a["read_log"]]), hide_index=True, width="stretch")
                    sub = a["submitted"]
                    st.markdown("제출: " + ("충분" if sub["sufficient"] else f"**부족 신고** — {'; '.join(sub['gaps'])}"))
                    d = dec.get((x["id"], a["round"]))
                    if d:
                        st.info(f"두 번째 원고 처리: **{d['decision']}** — {d['reason']}")
            with right:
                if single:
                    st.markdown(ags[0]["submitted"]["markdown"])
                else:
                    l2, r2 = st.columns(2)
                    with l2:
                        st.markdown("**리서처가 올린 원고**")
                        for a in ags:
                            st.caption(a["agent_id"])
                            st.markdown(a["submitted"]["markdown"])
                    with r2:
                        st.markdown("**최종본의 이 절**")
                        st.markdown(finals.get(x["title"], "(없음)").split("\n", 1)[-1])

with tab[1]:
    st.markdown(R["report"])

# ---------------------------------------------------------------- 격리·예산
with tab[2]:
    a, b = st.columns(2)
    with a:
        st.markdown("**코디네이터가 본 글자 (종류별)**")
        if iso["coordinator_by_kind"]:
            st.bar_chart(pd.Series(iso["coordinator_by_kind"], name="글자"), horizontal=True)
        else:
            st.caption("단일 에이전트 실행에는 코디네이터가 없다.")
        st.caption("목록 = 문서 id·길이·링크 수, 원고 = 리서처가 올린 글. 원문 본문은 여기 없어야 한다.")
    with b:
        st.markdown("**서브에이전트별로 본 글자 (프롬프트 + 도구 결과)**")
        st.bar_chart(pd.Series(iso["subagent_chars"], name="글자"), horizontal=True)
    st.write(f"리서처가 읽은 원문 조각 {iso['snippets_checked']}개를 대조: 코디네이터 입력(원고 제외)에서 발견 "
             f"**{iso['coordinator_raw_leak']}**개 · 원고에 그대로 옮겨진 것 {iso['manuscript_verbatim']}개")
    st.markdown("**예산: 배정만큼 썼는가** — 대조군이 예산을 남기고 멈췄다면 이긴 것이 아니라 묶어 둔 것")
    st.dataframe(pd.DataFrame(M["budget"]).assign(사용률=lambda df: (df.used / df.budget).map("{:.0%}".format)),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- 지표
with tab[3]:
    st.markdown("**경보 — 0이어야 한다** (정답표·판정 모델 없이, 글과 읽은 자료만 본다)")
    st.dataframe(pd.DataFrame([{"경보": m.name, "값": M["alarms"][m.name], "보는 장치": m.device, "설명": m.desc}
                               for m in METRICS if m.kind == "alarm"]), hide_index=True, width="stretch")
    st.markdown("**신호 — 순위가 아니라 어디를 고칠지 가리킨다**")
    st.dataframe(pd.DataFrame([{"신호": m.name, "값": None if M["signals"][m.name] is None else round(M["signals"][m.name], 3),
                                "방향": {"high": "높을수록", "low": "낮을수록"}.get(m.better, ""), "보는 장치": m.device,
                                "설명": m.desc} for m in METRICS if m.kind == "signal"]), hide_index=True, width="stretch")
    if M["audit"]:
        st.error("스위치 감사: " + "; ".join(M["audit"]))

# ---------------------------------------------------------------- 인용 대조
with tab[4]:
    st.caption("읽지 않은 문서 인용은 코드가 잡는다. 읽은 문서를 엉뚱한 문장에 붙인 것은 사람이 원문과 대조해 잡는다.")
    rows = citation_rows(R, corpus)
    if rows:
        df = pd.DataFrame([{"#": k, "절": r["section"], "문장": CITE.sub("", r["sentence"])[:90], "인용": ", ".join(r["cites"]),
                            "어휘 대조": r["lexical"], "안 읽음": len(r["not_read"])} for k, r in enumerate(rows)])
        st.dataframe(df, hide_index=True, width="stretch", height=240)
        k = st.number_input("대조할 문장 # (기본: 어휘 대조가 가장 낮은 문장)", 0, len(rows) - 1, int(df["어휘 대조"].idxmin()))
        row = rows[k]
        st.markdown(f"> {row['sentence']}")
        words = {w for w in tokenize(CITE.sub("", row["sentence"])) if len(w) >= 2 and not w.isascii()}
        for cid in row["cites"]:
            if cid not in corpus.text:
                st.error(f"{cid}: 코퍼스에 없음")
                continue
            text = corpus.text[cid]
            start = max(range(0, max(1, len(text) - 1500), 750),
                        key=lambda p: len(words & set(tokenize(text[p:p + 1500]))))
            win = html.escape(text[start:start + 1500])
            for w in sorted(words, key=len, reverse=True):
                win = win.replace(w, f"<mark>{w}</mark>")
            st.markdown(f"**{cid}** ({'이 절이 읽음' if cid not in row['not_read'] else '이 절이 읽지 않음'}) — {corpus.url(cid)}")
            st.markdown(f"<div style='font-size:.9em;border-left:3px solid #999;padding-left:8px'>…{win}…</div>",
                        unsafe_allow_html=True)
        p = REPORTS / name / "citation_notes.json"
        notes = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        opts = ["미확인", "제자리", "엉뚱한 곳에 붙음", "일부만"]
        v = st.radio("원문에 그 내용이 있는가", opts, index=opts.index(notes.get(str(k), {}).get("verdict", "미확인")), horizontal=True)
        memo = st.text_input("메모", notes.get(str(k), {}).get("memo", ""))
        if st.button("판정 기록"):
            notes[str(k)] = {"sentence": row["sentence"], "cites": row["cites"], "verdict": v, "memo": memo}
            p.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
            st.success("저장했습니다")
    else:
        st.info("인용 문장이 없습니다.")

# ---------------------------------------------------------------- 비교
with tab[5]:
    same = [r for r in rs if r.split("__")[0] == name.split("__")[0]]
    picks = st.multiselect("같은 질문의 실행을 나란히 읽기 (기본 · 장치 하나 끈 것 · 단일 에이전트)", same, default=same[:3])
    if picks:
        cols = st.columns(len(picks))
        for col, p in zip(cols, picks):
            with col:
                m = metrics_of(REPORTS / p)
                st.markdown(f"**{m['arm']}**  \n`{p}`")
                st.caption(("경보 모두 0" if m["alarms_ok"] else "경보 켜짐 → 비교 제외") + f" · 인용 문서 {m['signals']['unique_cited']}"
                           f" · 최소 예산 사용률 {(m['signals']['budget_use_min'] or 0):.0%}")
                with st.container(height=650):
                    st.markdown((REPORTS / p / "report.md").read_text(encoding="utf-8"))
    p = ROOT / CONFIG["output_dir"] / "compare_notes.md"
    txt = st.text_area("어느 쪽이 나은지, 그렇게 판단한 이유 (자기 말로)", p.read_text(encoding="utf-8") if p.exists() else "", height=180)
    if st.button("비교 기록 저장"):
        p.write_text(txt, encoding="utf-8")
        st.success("output/compare_notes.md에 저장했습니다")
