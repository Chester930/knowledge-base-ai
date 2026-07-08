#!/usr/bin/env python
"""
SVO 品質驗證機制 — 互動式逐階段檢驗工具

給人工把關用：輸入一句話（或一段文字），逐階段印出「抽取 → 審查 → 重試 → 本體擴充」
每一步的完整輸入輸出，讓使用者可以親自檢查每個模型的判斷品質，而不是只看最終結果
或翻 log。對應 docs/THEORETICAL_ARCHITECTURE.md 第4.1節的三模型迴圈。

用法：
  # 互動模式（可連續輸入多句，每句都跑一次完整流程）
  python run_svo_pipeline_debug.py

  # 單次模式
  python run_svo_pipeline_debug.py --text "臺灣湯淺電池股份有限公司宜蘭廠..."

  # 指定要用哪個 KG 的擴充類型清單（會影響抽取 prompt 與本體擴充的持久化對象）
  # 不指定時使用一個隨機、不影響任何真實 KG 的臨時 kg_id
  python run_svo_pipeline_debug.py --kg <kg_id> --text "..."

  # 停用驗證機制，只看單純抽取結果（等同 svo_verify_enabled=False 的舊行為）
  python run_svo_pipeline_debug.py --no-verify --text "..."
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from uuid import UUID, uuid4

sys.path.append(".")

from core.database import connect, disconnect
from core.providers.factory import init_providers
from services.svo_service import (
    extract_svo_from_text,
    verify_svo_extraction,
    propose_ontology_extension,
)
from services import ontology_service


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_triples(triples, label: str) -> None:
    print(f"\n[{label}]（{len(triples)} 條）")
    if not triples:
        print("  （無）")
    for i, t in enumerate(triples):
        print(f"  {i}. {t.subject}({t.subject_type}) -[{t.rel_type}:{t.verb}]-> {t.object}({t.object_type})")


def _print_verdicts(verdicts: list[dict]) -> None:
    for v in verdicts:
        mark = "✅ 通過" if v["accepted"] else "❌ 拒絕"
        print(f"  [{v['index']}] {mark} — {v['reason']}")


async def run_pipeline(text: str, kg_id: str, use_verify: bool, max_retries: int) -> None:
    from core.config import settings

    kg_id_str = str(kg_id)
    extra_et = ontology_service.get_extra_entity_types(kg_id_str)
    extra_rt = ontology_service.get_extra_rel_types(kg_id_str)

    _print_header(f"原文（KG: {kg_id_str}）")
    print(text)
    if extra_et or extra_rt:
        print(f"\n此 KG 已擴充的實體類型：{extra_et or '（無）'}")
        print(f"此 KG 已擴充的關係類型：{extra_rt or '（無）'}")

    # ── Stage 1：初次抽取 ──────────────────────────────────────────────
    _print_header("Stage 1｜抽取模型")
    t0 = time.time()
    triples = await extract_svo_from_text(text, extra_entity_types=extra_et, extra_rel_types=extra_rt)
    print(f"（耗時 {time.time()-t0:.2f}s）")
    _print_triples(triples, "抽取結果")

    if not use_verify:
        print("\n[--no-verify] 已停用審查機制，流程在此結束。")
        return
    if not triples:
        print("\n抽取結果為空，視為無需審查，流程結束。")
        return

    # ── Stage 2：審查 ──────────────────────────────────────────────────
    attempt = 0
    while True:
        _print_header(f"Stage 2｜審查模型（第 {attempt + 1} 次審查）")
        t0 = time.time()
        accepted, verdicts = await verify_svo_extraction(text, triples)
        print(f"（耗時 {time.time()-t0:.2f}s）")
        _print_verdicts(verdicts)
        print(f"\n整體判定：{'✅ 全數通過' if accepted else '❌ 有三元組被拒絕'}")

        if accepted:
            _print_header("最終結果（將寫入知識圖譜）")
            _print_triples(triples, "最終三元組")
            return

        if attempt >= max_retries:
            break
        attempt += 1

        # ── Stage 1b：重新抽取 ────────────────────────────────────────
        _print_header(f"Stage 1b｜重新抽取（第 {attempt} 次重試）")
        t0 = time.time()
        triples = await extract_svo_from_text(text, extra_entity_types=extra_et, extra_rel_types=extra_rt)
        print(f"（耗時 {time.time()-t0:.2f}s）")
        _print_triples(triples, "重新抽取結果")
        if not triples:
            print("\n重新抽取結果為空，流程結束。")
            return

    # ── Stage 3：本體擴充 ──────────────────────────────────────────────
    _print_header("Stage 3｜本體擴充模型（重試用盡仍被拒絕）")
    t0 = time.time()
    extension = await propose_ontology_extension(text, triples, verdicts, kg_id=kg_id_str)
    print(f"（耗時 {time.time()-t0:.2f}s）")
    print(f"提議新增實體類型：{extension['entity_types'] or '（無）'}")
    print(f"提議新增關係類型：{extension['rel_types'] or '（無）'}")
    print(f"適用範圍：{extension['scope']}")
    print(f"理由：{extension['rationale']}")

    if not extension["entity_types"] and not extension["rel_types"]:
        _print_header("最終結果（將寫入知識圖譜，本體無擴充）")
        _print_triples(triples, "最終三元組（重試後，未通過但沿用）")
        return

    confirm = input(
        f"\n是否要實際寫入這些新類型到 {'全域' if extension['scope']=='global' else f'KG {kg_id_str}'}？"
        f"（y/N，直接 Enter 視為 N，僅預覽不寫入）："
    ).strip().lower()
    if confirm == "y":
        await ontology_service.add_extension(
            kg_id_str, extension["entity_types"], extension["rel_types"], extension["scope"],
        )
        print("已寫入。")
        extra_et = ontology_service.get_extra_entity_types(kg_id_str)
        extra_rt = ontology_service.get_extra_rel_types(kg_id_str)

        _print_header("Stage 1c｜用擴充後類型再抽取一次")
        t0 = time.time()
        triples = await extract_svo_from_text(text, extra_entity_types=extra_et, extra_rel_types=extra_rt)
        print(f"（耗時 {time.time()-t0:.2f}s）")
        _print_triples(triples, "擴充後抽取結果")
    else:
        print("未寫入，本次僅為預覽（與 extract_svo_verified() 實際運行時的自動行為不同——正式流程不會詢問，會直接寫入）。")

    _print_header("最終結果（將寫入知識圖譜）")
    _print_triples(triples, "最終三元組")


async def main():
    parser = argparse.ArgumentParser(description="SVO 品質驗證機制互動式逐階段檢驗工具")
    parser.add_argument("--text", type=str, default=None, help="要測試的原文，不提供則進入互動模式")
    parser.add_argument("--kg", type=str, default=None, help="KG UUID（不提供則用隨機臨時 ID，不影響真實 KG）")
    parser.add_argument("--no-verify", action="store_true", help="停用審查機制，只看單純抽取結果")
    parser.add_argument("--max-retries", type=int, default=None, help="覆寫 svo_verify_max_retries（預設讀 .env）")
    args = parser.parse_args()

    await connect()
    init_providers()

    from core.config import settings
    max_retries = args.max_retries if args.max_retries is not None else settings.svo_verify_max_retries
    kg_id = args.kg or str(uuid4())
    if not args.kg:
        print(f"[提示] 未指定 --kg，使用臨時 KG ID：{kg_id}（本體擴充若發生只會寫進這個臨時 bucket，不影響任何真實 KG）")

    if args.text:
        await run_pipeline(args.text, kg_id, not args.no_verify, max_retries)
    else:
        print("互動模式：貼上要測試的句子，直接 Enter 結束程式。\n")
        while True:
            try:
                text = input("\n請輸入原文 > ").strip()
            except EOFError:
                break
            if not text:
                break
            await run_pipeline(text, kg_id, not args.no_verify, max_retries)

    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
