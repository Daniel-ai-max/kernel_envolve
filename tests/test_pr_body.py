"""Check accepted descriptions and failures that would hide missing review information."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check_pr_body.py"
spec = importlib.util.spec_from_file_location("check_pr_body", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

BODY = """# 概述

后台扫描失败后，面板仍显示旧证据的同步时间；现在展示最近一次有效扫描时间。

## 变更类型

- [x] Bug 修复

## 行为变化

### 修改前

HTTP 成功响应会掩盖证据扫描失败。

### 修改后

扫描失败时保留已有结果，并显示证据更新延迟。

## 影响范围

- 评测 / baseline：N/A：不改变评测协议或 baseline。
- Agent / GPU / 面板：面板读取证据观测时间。
- 数据 / 兼容性：复用已有时间字段，无需迁移。
- 依赖 / 部署：N/A：没有新增依赖，重启 Web 服务生效。

## 实现说明

将证据观测时间与 HTTP 响应时间分开，避免混淆。

## 验证

- 执行：`pytest tests/test_manager.py -q`；人工检查扫描失败后重连。
- 结果：相关测试通过；重连后保留旧结果和对应观测时间。

## 风险与回滚

- 风险等级：低
- 主要风险：扫描较慢时延迟提示可能持续显示。
- 回滚或缓解方式：回退本次 UI 改动并重启 Web 服务；无需修改数据库。

## Checklist

- [x] 验证结果与实际执行一致，未执行项已说明
- [x] 已更新相关文档，或说明无需更新的原因
- [x] 已说明兼容性与升级影响
- [x] 不包含无关改动或敏感信息
"""


def test_complete_description_and_crlf():
    assert checker.validate_pr_body(BODY) == []
    assert checker.validate_pr_body(BODY.replace("\n", "\r\n")) == []


def test_unfilled_repository_template_is_rejected():
    body = (ROOT / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
    errors = checker.validate_pr_body(body)
    for fragment in ("概述", "修改前", "执行", "结果", "风险等级", "Checklist"):
        assert any(fragment in error for error in errors)


@pytest.mark.parametrize("replacement", ["", "<!-- 说明原因 -->", "TODO", "...", "N/A"])
def test_overview_cannot_borrow_content_from_later_sections(replacement):
    body = BODY.replace(
        "后台扫描失败后，面板仍显示旧证据的同步时间；现在展示最近一次有效扫描时间。", replacement
    )
    assert any("# 概述" in error for error in checker.validate_pr_body(body))


@pytest.mark.parametrize("value", ["", "N/A", "N/A：", "N/A：TODO", "待补充", "`...`"])
def test_missing_or_placeholder_field_is_rejected(value):
    body = BODY.replace("N/A：不改变评测协议或 baseline。", value)
    assert any("评测 / baseline" in error for error in checker.validate_pr_body(body))


@pytest.mark.parametrize(
    ("before", "after", "error"),
    [
        ("### 修改后", "### 修改后\n\n### 修改后", "标题"),
        ("### 修改前", "### 之前", "标题"),
        ("- [x] Bug 修复", "- [x] Bug 修复\n- [x] 功能开发", "只勾选"),
        ("- [x] Bug 修复", "- [x] 其他：", "具体类型"),
        ("- [x] Bug 修复", "- [ ] Bug 修复", "只勾选"),
        ("风险等级：低", "风险等级：低 / 中 / 高", "风险等级"),
        ("- [x] 不包含无关改动或敏感信息", "- [ ] 不包含无关改动或敏感信息", "Checklist"),
        ("相关测试通过；重连后保留旧结果和对应观测时间。", "TODO", "结果"),
        ("回退本次 UI 改动并重启 Web 服务；无需修改数据库。", "无", "回滚"),
    ],
)
def test_invalid_descriptions_have_actionable_errors(before, after, error):
    assert any(error in item for item in checker.validate_pr_body(BODY.replace(before, after)))


def test_heading_order_is_enforced():
    body = BODY.replace("### 修改前", "### 临时").replace("### 修改后", "### 修改前")
    assert checker.validate_pr_body(body.replace("### 临时", "### 修改后"))


def test_comments_and_code_cannot_supply_required_headings():
    assert checker.validate_pr_body(f"<!--\n{BODY}\n-->")
    assert checker.validate_pr_body(f"```markdown\n{BODY}\n```")
    assert checker.validate_pr_body(f"~~~markdown\n{BODY}\n~~~")
    body = BODY.replace(
        "将证据观测时间与 HTTP 响应时间分开，避免混淆。", "示例：\n```markdown\n# 概述\n```"
    )
    assert checker.validate_pr_body(body) == []


def test_readability_and_other_change_type():
    assert checker.validate_pr_body(BODY + r"\n\n")
    assert checker.validate_pr_body(BODY + "\n<details>展开说明\n")
    assert (
        checker.validate_pr_body(BODY.replace("- [x] Bug 修复", "- [x] 其他：示例数据更新")) == []
    )


def test_cli_reads_markdown_and_github_api_without_executing_body(tmp_path):
    marker = tmp_path / "should-not-exist"
    body = BODY.replace("将证据观测时间", f"`$(touch {marker})`\n\n将证据观测时间")
    for option, name, value in (
        ("--body-file", "pr.md", body),
        ("--pr-file", "pr.json", json.dumps({"body": body})),
    ):
        path = tmp_path / name
        path.write_text(value)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), option, str(path)], capture_output=True
        )
        assert result.returncode == 0, result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("data", ["{", "[]", '{"body": null}', '{"body": 7}'])
def test_cli_rejects_missing_or_invalid_api_body(tmp_path, data):
    path = tmp_path / "pr.json"
    path.write_text(data)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--pr-file", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "::error title=Invalid PR description::" in result.stdout


def test_target_workflow_only_runs_trusted_code():
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-body.yml").read_text())
    # PyYAML follows YAML 1.1, where the GitHub Actions key 'on' is a boolean.
    events = workflow.get("on", workflow.get(True))
    assert "edited" in events["pull_request_target"]["types"]
    assert "workflow_dispatch" in events
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
    steps = workflow["jobs"]["validate"]["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert checkout["with"]["persist-credentials"] is False
    for step in steps:
        script = step.get("run", "")
        assert "${{" not in script  # Event data only enters quoted environment variables.
        assert "pull_request.head" not in str(step)
