#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use a local Qwen2.5-72B-Instruct-AWQ model to correct PHQ-9 predictions.

The script keeps binary/ternary labels unchanged. It only replaces phq9_pred by:

    corrected_phq9 = 0.9 * mean(qwen_phq9_10_runs) + 0.1 * original_phq9_pred

The PHQ range implied by the voted class labels is provided in the prompt only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_BINARY_CSV = PROJECT_ROOT / "predict/binary.csv"
DEFAULT_TERNARY_CSV = PROJECT_ROOT / "predict/ternary.csv"
DEFAULT_DESCRIPTIONS_CSV = PROJECT_ROOT / "Elder-trainval/descriptions.csv"
DEFAULT_TRANSCRIPTS_CSV = PROJECT_ROOT / "Elder-test/sample_transcriptions_qwen_asr.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "qwen_phq9_corrected"


@dataclass
class SubjectInput:
    sid: int
    binary_pred: int
    ternary_pred: int
    original_phq9: float
    allowed_min: float
    allowed_max: float
    description: str
    transcript: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct PHQ-9 predictions with local Qwen2.5-72B-Instruct-AWQ."
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--binary_csv", type=Path, default=DEFAULT_BINARY_CSV)
    parser.add_argument("--ternary_csv", type=Path, default=DEFAULT_TERNARY_CSV)
    parser.add_argument("--descriptions_csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    parser.add_argument("--transcripts_csv", type=Path, default=DEFAULT_TRANSCRIPTS_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_description_chars", type=int, default=900)
    parser.add_argument("--max_transcript_chars", type=int, default=2200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated subject ids to process, e.g. 1,6,10.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip subjects already present in phq9_correction_details.csv.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write prompts_preview.jsonl, without loading the LLM.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer from local files only.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument(
        "--backend",
        type=str,
        default="autoawq",
        choices=["autoawq", "transformers"],
        help="Use AutoAWQ direct loader by default; transformers is kept as a fallback.",
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head - 24
    return text[:keep_head] + " ...[中间截断]... " + text[-keep_tail:]


def sample_sort_key(sample_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(sample_name))
    if match:
        return int(match.group(1)), str(sample_name)
    return 9999, str(sample_name)


def allowed_range(binary_pred: int, ternary_pred: int) -> tuple[float, float]:
    if ternary_pred == 0:
        return 0.0, 4.0
    if ternary_pred == 1:
        return 5.0, 9.0
    if ternary_pred == 2:
        return 10.0, 27.0
    if binary_pred == 0:
        return 0.0, 4.0
    return 5.0, 27.0


def load_descriptions(path: Path, max_chars: int) -> dict[int, str]:
    desc: dict[int, str] = {}
    for row in read_csv_rows(path):
        sid = int(row.get("ID") or row.get("id"))
        desc[sid] = compact_text(row.get("Descriptions", ""), max_chars)
    return desc


def load_transcripts(path: Path, max_chars: int) -> dict[int, str]:
    grouped: dict[int, list[tuple[str, str]]] = {}
    for row in read_csv_rows(path):
        sid = int(row.get("ID") or row.get("id"))
        sample = row.get("Sample", "")
        transcript = row.get("Transcript", "")
        if not transcript:
            continue
        grouped.setdefault(sid, []).append((sample, transcript))

    merged: dict[int, str] = {}
    for sid, items in grouped.items():
        items = sorted(items, key=lambda x: sample_sort_key(x[0]))
        joined = "\n".join(f"{sample}: {text}" for sample, text in items)
        merged[sid] = compact_text(joined, max_chars)
    return merged


def load_subject_inputs(args: argparse.Namespace) -> list[SubjectInput]:
    binary_rows = read_csv_rows(args.binary_csv)
    ternary_rows = read_csv_rows(args.ternary_csv)
    binary_by_id = {int(r["id"]): r for r in binary_rows}
    ternary_by_id = {int(r["id"]): r for r in ternary_rows}
    descriptions = load_descriptions(args.descriptions_csv, args.max_description_chars)
    transcripts = load_transcripts(args.transcripts_csv, args.max_transcript_chars)

    ids = sorted(set(binary_by_id) & set(ternary_by_id))
    if args.ids:
        wanted = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        ids = [sid for sid in ids if sid in wanted]
    if args.limit is not None:
        ids = ids[: args.limit]

    subjects: list[SubjectInput] = []
    for sid in ids:
        b_row = binary_by_id[sid]
        t_row = ternary_by_id[sid]
        binary_pred = int(float(b_row["binary_pred"]))
        ternary_pred = int(float(t_row["ternary_pred"]))
        original_phq9 = float(t_row.get("phq9_pred") or b_row["phq9_pred"])
        low, high = allowed_range(binary_pred, ternary_pred)
        subjects.append(
            SubjectInput(
                sid=sid,
                binary_pred=binary_pred,
                ternary_pred=ternary_pred,
                original_phq9=original_phq9,
                allowed_min=low,
                allowed_max=high,
                description=descriptions.get(sid, ""),
                transcript=transcripts.get(sid, ""),
            )
        )
    return subjects


def build_messages(subject: SubjectInput) -> list[dict[str, str]]:
    system_prompt = (
        "你是一个严谨的PHQ-9抑郁程度评分校准助手。"
        "你只能根据给定分类标签、PHQ-9映射规则、人格描述和ASR文本以及抑郁倾向相关知识做分数校准。"
        "不要改变分类标签对应的PHQ范围。输出必须是JSON。"
    )
    user_prompt = f"""
请为该受试者估计一个PHQ-9总分，范围0到27。

已知模型五投票分类结果：
- 二分类 binary_pred = {subject.binary_pred}
- 三分类 ternary_pred = {subject.ternary_pred}
- 原模型回归 phq9_pred = {subject.original_phq9:.6f}

分类与PHQ-9映射必须遵守：
- PHQ 0-4：三分类=0，二分类=0
- PHQ 5-9：三分类=1，二分类=1
- PHQ >=10：三分类=2，二分类=1

因此本样本最终可选PHQ-9范围是 [{subject.allowed_min:.0f}, {subject.allowed_max:.0f}]。
如果文本信息与分类范围有冲突，必须优先遵守分类范围。

PHQ-9知识和评分参考：
PHQ-9由9个抑郁相关条目组成，每项按过去两周频率计0-3分：
- 0分：完全没有或基本没有。
- 1分：偶尔出现，几天有。
- 2分：较频繁，超过一半时间有。
- 3分：几乎每天都有，持续且影响明显。

9个核心条目包括：
1. 做事兴趣或愉快感下降。
2. 情绪低落、沮丧、无望。
3. 睡眠问题：入睡困难、易醒、睡太多。
4. 疲劳、精力不足。
5. 食欲差或吃太多。
6. 自我评价低、自责、觉得自己失败或拖累他人。
7. 注意力难以集中。
8. 动作或说话变慢，或坐立不安。
9. 伤害自己、轻生、死亡相关想法。

总分严重程度分层：
- 0-4：无/极轻微。一般只有短暂烦恼，能明确否认多数症状，日常功能基本正常。
- 5-9：轻度。可见一些烦恼、睡眠/精力/兴趣下降、孤独或担心，但症状数量少或强度低，功能损害较轻。
- 10-14：中度。多个条目较稳定出现，例如持续低落、明显兴趣下降、睡眠差、乏力、注意力下降、自责，生活功能受到影响。
- 15-19：中重度。症状更广泛、更频繁，情绪低落和功能受损明显，可能伴随强烈无望感、自责或躯体化表达。
- 20-27：重度。多项症状接近每天出现，严重影响生活，或出现明确死亡/自伤想法。

文本线索解释规则：
- 明确否认“没有烦恼、没有担心、睡眠还行、吃饭正常、心情可以”时，应降低分数；但如果同时存在长期疾病、孤独、丧偶、明显担忧或反复抱怨，不要只因一句否认就给极低分。
- 老年样本可能不直接说“抑郁”，而用“没意思、没精神、睡不好、身体难受、没人陪、拖累孩子、想不开、活着没劲”等表达抑郁倾向。
- 单纯讲故事、回答很短、ASR信息少时，不要过度推断；更多依赖分类标签范围、人格背景和原回归分数。
- 神经质高、经济压力、独居/家庭支持少、慢性病或神经系统疾病，可作为轻微上调因素；外向性/宜人性/尽责性较高、家庭支持好，可作为轻微下调因素。
- 如果三分类=0，分数应在0-4内根据文本强弱细分；如果三分类=1，分数应在5-9内细分；如果三分类=2，分数应在10-27内细分。
- 不要为了“平均”总是输出区间中点。请根据证据选择区间低端、中端或高端。

人格和背景描述：
{subject.description or "无可用人格描述。"}

ASR转录文本：
{subject.transcript or "无可用ASR文本。"}

请综合判断PHQ-9分数，该分数为六位小数。只输出一行JSON，格式如下：
{{"phq9": 具体数值, "rationale": "不超过30个汉字的简短依据"}}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_phq9(text: str) -> float | None:
    text = str(text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "phq9" in obj:
            return float(obj["phq9"])
    except Exception:
        pass
    match = re.search(r'"phq9"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    match = re.search(r"PHQ-?9[^0-9-]*(-?\d+(?:\.\d+)?)", text, flags=re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return None


def load_model(args: argparse.Namespace):
    try:
        import torch
        import transformers.activations as transformer_activations
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "当前环境缺少 torch/transformers。请在 mpddavg 环境中运行该脚本。"
        ) from exc

    # AutoAWQ 0.2.x imports the old Transformers activation symbol
    # PytorchGELUTanh, which was renamed in newer Transformers versions.
    if (
        not hasattr(transformer_activations, "PytorchGELUTanh")
        and hasattr(transformer_activations, "PytorchGELUTanhActivation")
    ):
        transformer_activations.PytorchGELUTanh = (
            transformer_activations.PytorchGELUTanhActivation
        )
    if (
        not hasattr(transformer_activations, "PytorchGELUTanh")
        and hasattr(transformer_activations, "GELUTanh")
    ):
        transformer_activations.PytorchGELUTanh = transformer_activations.GELUTanh

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.torch_dtype]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        if args.backend == "autoawq":
            from awq import AutoAWQForCausalLM

            try:
                model = AutoAWQForCausalLM.from_quantized(
                    str(args.model_path),
                    trust_remote_code=args.trust_remote_code,
                    safetensors=True,
                    fuse_layers=True,
                    device_map=args.device_map,
                )
            except TypeError:
                model = AutoAWQForCausalLM.from_quantized(
                    str(args.model_path),
                    trust_remote_code=args.trust_remote_code,
                    safetensors=True,
                    fuse_layers=True,
                )
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map=args.device_map,
                dtype=dtype,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
        model.eval()
        return tokenizer, model, torch
    except Exception as exc:
        raise RuntimeError(
            "Qwen2.5-72B-Instruct-AWQ加载失败。当前脚本默认使用AutoAWQ直接加载；"
            "如果仍失败，请把完整Traceback贴出来，通常需要单独固定transformers/autoawq版本，"
            "或改用支持AWQ的vLLM环境。"
        ) from exc


def generate_once(
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    seed: int,
) -> str:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")
    if args.backend == "autoawq":
        device = "cuda:0" if torch_module.cuda.is_available() else "cpu"
        inputs = inputs.to(device)
    else:
        device = getattr(model, "device", None)
        if device is not None:
            inputs = inputs.to(device)

    with torch_module.no_grad():
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.eos_token_id,
        }
        try:
            output_ids = model.generate(**inputs, **generate_kwargs)
        except TypeError:
            output_ids = model.generate(inputs.input_ids, **generate_kwargs)
    new_tokens = output_ids[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def existing_done_ids(details_path: Path) -> set[int]:
    if not details_path.exists():
        return set()
    done: set[int] = set()
    for row in read_csv_rows(details_path):
        try:
            done.add(int(row["id"]))
        except Exception:
            continue
    return done


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = load_subject_inputs(args)
    if not subjects:
        raise RuntimeError("没有读到可处理的subject，请检查binary/ternary输入路径。")

    prompts_path = args.output_dir / "prompts_preview.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for subject in subjects:
            f.write(
                json.dumps(
                    {"id": subject.sid, "messages": build_messages(subject)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    if args.dry_run:
        print(f"[dry-run] wrote prompts: {prompts_path}")
        return

    raw_path = args.output_dir / "llm_raw_generations.csv"
    details_path = args.output_dir / "phq9_correction_details.csv"
    raw_fields = [
        "id",
        "run_idx",
        "raw_text",
        "parsed_phq9",
    ]
    detail_fields = [
        "id",
        "binary_pred",
        "ternary_pred",
        "original_phq9",
        "llm_mean_phq9",
        "corrected_phq9",
        "allowed_min",
        "allowed_max",
        "n_valid_generations",
    ]

    done_ids = existing_done_ids(details_path) if args.resume else set()
    tokenizer, model, torch_module = load_model(args)

    processed_details: list[dict[str, Any]] = []
    for idx, subject in enumerate(subjects, start=1):
        if subject.sid in done_ids:
            print(f"[{idx}/{len(subjects)}] id={subject.sid} skip (resume)")
            continue

        print(
            f"[{idx}/{len(subjects)}] id={subject.sid} "
            f"bin={subject.binary_pred} ter={subject.ternary_pred} "
            f"prompt_range=[{subject.allowed_min:.0f},{subject.allowed_max:.0f}]"
        )
        messages = build_messages(subject)
        parsed_values: list[float] = []

        for run_idx in range(args.n_samples):
            run_seed = args.seed + subject.sid * 1000 + run_idx
            raw_text = generate_once(tokenizer, model, torch_module, messages, args, run_seed)
            parsed = parse_phq9(raw_text)
            if parsed is not None:
                parsed_values.append(float(parsed))
            append_csv_row(
                raw_path,
                raw_fields,
                {
                    "id": subject.sid,
                    "run_idx": run_idx + 1,
                    "raw_text": raw_text,
                    "parsed_phq9": "" if parsed is None else f"{parsed:.6f}",
                },
            )

        if parsed_values:
            llm_mean = sum(parsed_values) / len(parsed_values)
            corrected = 0.8 * llm_mean + 0.2 * subject.original_phq9
        else:
            llm_mean = subject.original_phq9
            corrected = subject.original_phq9

        detail = {
            "id": subject.sid,
            "binary_pred": subject.binary_pred,
            "ternary_pred": subject.ternary_pred,
            "original_phq9": f"{subject.original_phq9:.6f}",
            "llm_mean_phq9": f"{llm_mean:.6f}",
            "corrected_phq9": f"{corrected:.6f}",
            "allowed_min": f"{subject.allowed_min:.6f}",
            "allowed_max": f"{subject.allowed_max:.6f}",
            "n_valid_generations": len(parsed_values),
        }
        append_csv_row(details_path, detail_fields, detail)
        processed_details.append(detail)

    all_details = read_csv_rows(details_path)
    detail_by_id = {int(row["id"]): row for row in all_details}
    binary_rows: list[dict[str, Any]] = []
    ternary_rows: list[dict[str, Any]] = []
    for subject in subjects:
        row = detail_by_id.get(subject.sid)
        if row is None:
            continue
        binary_rows.append(
            {
                "id": subject.sid,
                "binary_pred": subject.binary_pred,
                "phq9_pred": f"{float(row['corrected_phq9']):.6f}",
            }
        )
        ternary_rows.append(
            {
                "id": subject.sid,
                "ternary_pred": subject.ternary_pred,
                "phq9_pred": f"{float(row['corrected_phq9']):.6f}",
            }
        )

    write_csv_rows(args.output_dir / "binary.csv", ["id", "binary_pred", "phq9_pred"], binary_rows)
    write_csv_rows(args.output_dir / "ternary.csv", ["id", "ternary_pred", "phq9_pred"], ternary_rows)
    meta = {
        "model_path": str(args.model_path),
        "binary_csv": str(args.binary_csv),
        "ternary_csv": str(args.ternary_csv),
        "descriptions_csv": str(args.descriptions_csv),
        "transcripts_csv": str(args.transcripts_csv),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "formula": "corrected_phq9 = 0.9 * mean(qwen_parsed_phq9) + 0.1 * original_phq9_pred; PHQ class range is prompt-only",
        "processed_new_subjects": len(processed_details),
        "total_output_subjects": len(binary_rows),
    }
    (args.output_dir / "prediction_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote: {args.output_dir / 'binary.csv'}")
    print(f"wrote: {args.output_dir / 'ternary.csv'}")
    print(f"raw generations: {raw_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
