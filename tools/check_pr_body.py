#!/usr/bin/env python3
"""Validate readable PR descriptions using only the Python standard library.

Adapted from AIPlatForm's general PR template gate; no project code is imported.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HEADINGS = (
    "# 概述",
    "## 变更类型",
    "## 行为变化",
    "### 修改前",
    "### 修改后",
    "## 影响范围",
    "## 实现说明",
    "## 验证",
    "## 风险与回滚",
    "## Checklist",
)
CHANGE_TYPES = ("功能开发", "Bug 修复", "重构", "性能优化", "文档 / CI / 维护", "其他")
IMPACT_FIELDS = ("评测 / baseline", "Agent / GPU / 面板", "数据 / 兼容性", "依赖 / 部署")
CHECKLIST = (
    "验证结果与实际执行一致，未执行项已说明",
    "已更新相关文档，或说明无需更新的原因",
    "已说明兼容性与升级影响",
    "不包含无关改动或敏感信息",
)


def meaningful(value):
    value = value.strip().strip("`*_ \t")
    value = re.sub(r"^(?:N/?A|不适用)\s*[：:]\s*", "", value, flags=re.I)
    return bool(value) and not re.fullmatch(
        r"(?:\.{3,}|…+|待补充|待填写|TBD|TODO|N/?A|none|无|不适用)[。.!！]?", value, re.I
    )


def parse(body):
    """Ignore comments and fenced headings while preserving code as section content."""
    body = re.sub(r"<!--.*?(?:-->|\Z)", "", body, flags=re.S)
    lines = body.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    headings, fence = [], None
    content = list(lines)
    for index, line in enumerate(lines):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if marker:
            token, suffix = marker.groups()
            if fence is None:
                fence = token
                content[index] = ""
            elif token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip():
                fence = None
                content[index] = ""
        elif fence is None and re.match(r"^#{1,6}\s+\S", line):
            headings.append((index, line.strip()))
            content[index] = ""
    return lines, content, headings


def section(content, headings, target):
    for position, (start, heading) in enumerate(headings):
        if heading != target:
            continue
        end = headings[position + 1][0] if position + 1 < len(headings) else len(content)
        return "\n".join(content[start + 1 : end]).strip()
    return ""


def field(section_body, label):
    matches = re.findall(rf"^-[ \t]+{re.escape(label)}[ \t]*[：:][ \t]*(.*)$", section_body, re.M)
    return matches[0].strip() if len(matches) == 1 else ""


def substantive(section_body):
    lines = [re.sub(r"^\s*[-*+]\s+", "", line) for line in section_body.splitlines()]
    return any(meaningful(line) for line in lines) and not any(
        line.strip() and not meaningful(line) for line in lines
    )


def validate_pr_body(body):
    lines, content, headings = parse(body)
    errors = []
    expected = [heading for _, heading in headings if heading in HEADINGS]
    if expected != list(HEADINGS):
        errors.append("保留模板标题及顺序，每个标题只出现一次：" + " → ".join(HEADINGS))
    if not lines or next((line.strip() for line in lines if line.strip()), "") != HEADINGS[0]:
        errors.append("正文应以 # 概述 开始")
    for heading in ("# 概述", "### 修改前", "### 修改后", "## 实现说明"):
        if not substantive(section(content, headings, heading)):
            errors.append(f"{heading}：填写具体内容，删除占位符")

    types = section(content, headings, "## 变更类型")
    checked = re.findall(r"^- \[[xX]\]\s+(.+)$", types, re.M)
    if len(checked) != 1 or checked[0].split("：", 1)[0].split(":", 1)[0] not in CHANGE_TYPES:
        errors.append("变更类型：只勾选一个模板选项")
    elif checked[0].startswith("其他") and not meaningful(
        re.split("[：:]", checked[0], 1)[-1].replace("其他", "")
    ):
        errors.append("变更类型：选择其他时说明具体类型")

    for heading, labels in (
        ("## 影响范围", IMPACT_FIELDS),
        ("## 验证", ("执行", "结果")),
        ("## 风险与回滚", ("主要风险", "回滚或缓解方式")),
    ):
        text = section(content, headings, heading)
        for label in labels:
            if not meaningful(field(text, label)):
                errors.append(f"{heading} / {label}：填写内容；不适用时解释原因")

    risk = section(content, headings, "## 风险与回滚")
    if field(risk, "风险等级") not in {"低", "中", "高"}:
        errors.append("风险等级：只选择低、中、高之一")
    verification = section(content, headings, "## 验证")
    if not re.search(
        r"`[^`]+`|https?://|\b(?:CI|tests?|checks?)\b|人工检查", field(verification, "执行"), re.I
    ):
        errors.append("验证 / 执行：提供实际命令、CI、人工检查或证据链接")
    checklist = section(content, headings, "## Checklist")
    for label in CHECKLIST:
        if len(re.findall(rf"^- \[[xX]\]\s+{re.escape(label)}(?:[：:].*)?$", checklist, re.M)) != 1:
            errors.append(f"Checklist：确认后勾选“{label}”")
    if re.search(r"\\n(?:\\n)+", body):
        errors.append(r"使用真实换行，不要写入字面的 \n\n")
    if body.count("<details") != body.count("</details>"):
        errors.append("<details> 折叠块未正确闭合")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body-file", type=Path, help="UTF-8 Markdown body for local checks")
    source.add_argument("--pr-file", type=Path, help="GitHub pull-request API JSON")
    args = parser.parse_args(argv)
    try:
        body = (
            args.body_file.read_text(encoding="utf-8")
            if args.body_file
            else json.loads(args.pr_file.read_text(encoding="utf-8")).get("body") or ""
        )
        if not isinstance(body, str):
            raise ValueError("PR body must be text")
        errors = validate_pr_body(body)
    except (OSError, ValueError, AttributeError) as exc:
        errors = [f"无法读取 PR 正文：{exc}"]
    for error in errors:
        escaped = error.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error title=Invalid PR description::{escaped}")
    if errors:
        print("\nPR 正文校验失败，请按 .github/PULL_REQUEST_TEMPLATE.md 补全。", file=sys.stderr)
        return 1
    print("PR 正文符合 kernel_envolve 模板。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
