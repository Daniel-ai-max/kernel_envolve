import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
const source = await readFile(new URL("../src/kernel_envolve/manager/static/summary.js", import.meta.url), "utf8");
const { executionLabel, executionLabelFor, benefitLabel, noteSnippet, visibleTask } = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
test("empty verified best has an explicit label, never a parsed report speedup", () => {
  assert.equal(benefitLabel(null), "尚无完整验收收益");
  assert.equal(benefitLabel({outcome: "快20%"}), "尚无完整验收收益");
  assert.equal(benefitLabel({baseline_speedup: 1.2}), "1.200×");
});
test("GPU job status and evaluator result remain separate", () => {
  assert.equal(executionLabel("idle"), "无 GPU 作业");
  assert.equal(executionLabelFor({kind: "gpu_job", status: "failed"}), "GPU 作业 · 失败");
  assert.equal(executionLabelFor({kind: "evaluation", mode: "quick", correctness: "pass"}), "初筛 · 正确性通过");
  assert.equal(executionLabelFor(null), "尚无执行记录");
});
test("rich notes fit the overview without interpreting their numbers", () => {
  const text = "GPU job bf34931af0f349e28f147698360f50a8完成，22/22通过，未提速。";
  assert.match(noteSnippet(text, "空"), /bf34931a…完成，22\/22通过，未提速/);
  assert.ok(noteSnippet("文字".repeat(1000), "空").length <= 180);
  assert.equal(noteSnippet("  ", "尚无结论"), "尚无结论");
});

test("stopped current tasks remain visible; only superseded tasks are hidden", () => {
  for (const desired of ["active", "paused", "budget", "blocked"]) assert.equal(visibleTask({desired}), true);
  assert.equal(visibleTask({desired: "superseded"}), false);
  assert.equal(visibleTask({desired: "superseded"}, true), true);
});
