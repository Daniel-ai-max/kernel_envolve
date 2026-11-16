import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
const source = await readFile(new URL("../src/kernel_envolve/manager/static/summary.js", import.meta.url), "utf8");
const { executionLabel, executionLabelFor, benefitLabel, noteSnippet, visibleTask } = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
test("empty verified best uses a placeholder, never a parsed report speedup", () => {
  assert.equal(benefitLabel(null), "—");
  assert.equal(benefitLabel({outcome: "快20%"}), "—");
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

const extra = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
test("stale research is not relabeled as normal running or a user todo", () => {
  assert.equal(extra.phaseLabel({phase:"awaiting_progress"}), "暂无新实验");
  assert.equal(extra.phaseLabel({phase:"blocked"}), "恢复失败");
  assert.equal(extra.phaseLabel({phase:"observing"}), "记录同步");
  assert.equal(extra.phaseLabel({phase:"gpu_running",stale:true}), "会话失联");
});
test("local signal is distinct from verified best and failure never shows speed", () => {
  const quick={kind:"evaluation",mode:"quick",correctness:"pass",incremental_speedup:1.2};
  assert.match(extra.resultSummary(quick), /局部信号 1.200×/);
  assert.doesNotMatch(extra.resultSummary({...quick,evidence_error:"changed"}), /1.200/);
  assert.doesNotMatch(extra.resultSummary({...quick,execution:"error"}), /1.200/);
  assert.doesNotMatch(extra.resultSummary({...quick,correctness:"fail"}), /1.200/);
  assert.equal(benefitLabel(null), "—");
});
test("evaluator result survives its enclosing GPU wrapper completion", () => {
  const evalRecord={kind:"evaluation",mode:"full",recorded_at:15};
  const task={recent_execution:{kind:"gpu_job",status:"succeeded",started:10,recorded_at:20},recent_evaluation:evalRecord};
  assert.equal(extra.recentResult(task), evalRecord);
  task.recent_execution.task="different-task";
  assert.equal(extra.recentResult(task),task.recent_execution);
  delete task.recent_execution.task;
  task.recent_execution.status="failed";
  assert.equal(extra.recentResult(task),task.recent_execution);
});

const scopedTask = () => ({id:"example", spec:{baseline:{revision:"base"}}, best:null,
  verified_assessment:{schema:"scoped-assessment-v1", status:"verified", promotion:false,
    task:"example", baseline_revision:"base", source_sha256:"a".repeat(64), evidence_sha256:"b".repeat(64), recorded_at:100,
    metrics:[{label:"算子",baseline_speedup:1.4,cases:46,slower_cases:0},{label:"Graph",baseline_speedup:1.05,cases:46,slower_cases:11,worst_slowdown_percent:4.2}],
    checkpoints:[{minutes:60,eager_ratio:1/1.4,graph_ratio:1/1.05}],
    scenarios:[{name:"example",mode:"前反向",group:"forward_backward",baseline_us:10,candidate_us:9}]}});
test("verified scoped evidence is displayed without promoting best", () => {
  const task=scopedTask(), before=structuredClone(task);
  assert.deepEqual(extra.taskBenefit(task),{label:"算子 1.400×",detail:"Graph 1.050×",positive:true});
  assert.equal(extra.recentResult(task).kind,"scoped_assessment");
  assert.equal(extra.resultSummary(extra.recentResult(task)),"46 场景验收 · Graph 11 场景退化");
  assert.equal(extra.scopedAssessment(task).metrics[1].slower_cases,11);
  assert.deepEqual(task,before);
  task.best={baseline_speedup:1.8};
  assert.equal(extra.taskBenefit(task).label,"1.800×");
  task.recent_execution={recorded_at:110,kind:"gpu_job",status:"failed"};
  assert.equal(extra.recentResult(task).kind,"scoped_assessment");
});
test("invalid, mismatched, failed or malformed scoped evidence is never displayed", () => {
  const mutations=[t=>t.id="different",t=>t.spec.baseline.revision="other",t=>t.evidence_error="modified",
    t=>t.verified_assessment.status="failed",t=>t.verified_assessment.promotion=true,
    t=>t.verified_assessment.source_sha256="short",t=>t.verified_assessment.evidence_sha256="",
    t=>t.verified_assessment.recorded_at=NaN,t=>t.verified_assessment.metrics=[],
    t=>t.verified_assessment.metrics=[null],t=>t.verified_assessment.metrics[0].baseline_speedup=Infinity,
    t=>t.verified_assessment.metrics[0].slower_cases=47,t=>t.verified_assessment.metrics[0].worst_slowdown_percent=NaN,
    t=>t.verified_assessment.checkpoints=[null],t=>t.verified_assessment.checkpoints[0].eager_ratio=0,
    t=>t.verified_assessment.scenarios=[null],t=>t.verified_assessment.scenarios[0].baseline_us=0,
    t=>t.verified_assessment.scenarios[0].group="unrecognized"];
  for (const mutate of mutations) {
    const task=scopedTask();mutate(task);
    assert.equal(extra.scopedAssessment(task),null);
    assert.equal(extra.taskBenefit(task).label,"—");
  }
});
test("deadline and explicit stop remain distinct", () => {
  assert.equal(extra.phaseLabel({phase:"paused",reason:"task deadline reached"}),"预算结束");
  assert.equal(extra.phaseLabel({phase:"paused",reason:"user stop"}),"已停止");
});

{
  const research = {schema: "research-result-v1", status: "research", promotion: false,
    task: "research", baseline_revision: "base", source_sha256: "a".repeat(64), evidence_sha256: "b".repeat(64),
    recorded_at: 20, scope: "dq/dw stage", summary: "100 → 80 µs"};
  const task = {id: "research", spec: {baseline: {revision: "base"}}, research_result: research,
    recent_execution: {kind: "gpu_job", status: "succeeded", task: "research", recorded_at: 30},
    assessment_sync: {error: "Independent assessment failed", recorded_at: 40}};
  assert.equal(extra.recentResult(task).kind, "research_result");
  assert.equal(extra.resultSummary(extra.recentResult(task)), "阶段结果 · 100 → 80 µs");
  assert.equal(extra.executionLabelFor(extra.recentResult(task)), "阶段结果");
  assert.deepEqual(extra.taskBenefit(task), {label: "—", detail: "", positive: false});
  assert.equal(task.best, undefined);
  assert.equal(extra.taskBenefit({...task, evidence_error: "changed"}).label, "—");
  assert.equal(extra.taskBenefit({...task, best: {baseline_speedup: 1.2}}).label, "1.200×");
  assert.equal(extra.taskBenefit({...task, research_result: {...research, baseline_revision: "other"}}).label, "—");
  assert.equal(extra.scopedAssessment(task), null);
  const assessed = scopedTask();
  assessed.research_result = {...research, task: assessed.id, recorded_at: 90};
  assessed.recent_execution = {kind: "gpu_job", status: "failed", recorded_at: 200};
  assert.equal(extra.recentResult(assessed).kind, "scoped_assessment");
  assessed.research_result.recorded_at = 110;
  assert.equal(extra.recentResult(assessed).kind, "research_result");
  assert.equal(extra.researchResult({...task, research_result: {...research, status: "verified"}}), null);
  assert.equal(extra.researchResult({...task, research_result: {...research, task: "other"}}), null);
}

test("detail headline acknowledges research without implying full qualification", () => {
  const task = {id: "research", spec: {baseline: {revision: "base"}}, research_result: {
    schema: "research-result-v1", status: "research", promotion: false, task: "research",
    baseline_revision: "base", source_sha256: "a".repeat(64), evidence_sha256: "b".repeat(64),
    recorded_at: 10, scope: "stage", summary: "partial result"}};
  assert.equal(extra.evidenceSummary(task), "阶段结果");
  assert.equal(extra.evidenceSummary({...task, research_result: null}), "当前契约尚无评测记录");
  assert.equal(extra.evidenceSummary({...task, evidence_error: "changed"}), "证据异常");
  assert.equal(extra.evidenceSummary({...task, latest_evaluation: {mode: "full", correctness: "fail"}}), "完整评测 · 正确性失败");
});

test("narration is cited explanation and never a benefit or activity signal", () => {
  const task = {id: "toy", spec: {baseline: {revision: "base"}}, narration: {
    schema: "narration-v1", task: "toy", baseline: "base", updated_at: 20, source_at: 10,
    sources: {"event:1": {text: "observed probe", at: 10}},
    claims: {direction: null, finding: {text: "长序列改善；短序列未改善", sources: ["event:1"]}, failure: null, next: null}
  }};
  assert.equal(extra.narrationText(task), "长序列改善；短序列未改善");
  assert.equal(extra.taskBenefit(task).label, "—");
  assert.equal(extra.narrationFor({...task, spec: {baseline: {revision: "new"}}}), null);
  const failed = structuredClone(task);
  failed.narration.claims.failure = {text: "布局编译失败", sources: ["event:1"]};
  assert.equal(extra.narrationText(failed), "布局编译失败");
  const changed = structuredClone(task);
  changed.narration.claims.finding.sources = ["missing"];
  assert.equal(extra.narrationFor(changed), null);
});

test("external outcomes override stale narration without adding verified gains", () => {
  const task = {id: "toy", desired: "paused", phase: "paused", spec: {baseline: {revision: "base"}}, latest_validation: {
    schema: "validation-result-v1", task: "toy", baseline_revision: "base", promotion: false,
    source_sha256: "a".repeat(64), evidence_sha256: "b".repeat(64), recorded_at: 30,
    status: "no_benefit", scope: "public F+B", summary: "调用链没有收益"},
    narration: {schema: "narration-v1", task: "toy", baseline: "base", updated_at: 40, source_at: 10,
      sources: {old: {at: 10}}, claims: {direction: null, finding: null,
        failure: {text: "旧编译故障", sources: ["old"]}, next: {text: "旧计划", sources: ["old"]}}}};
  assert.match(extra.narrationText(task), /调用链没有收益/);
  assert.doesNotMatch(extra.narrationText(task), /旧编译故障|旧计划/);
  assert.equal(extra.recentResult(task).kind, "validation");
  assert.equal(extra.taskBenefit(task).label, "—");
  assert.equal(extra.validationResult({...task, spec: {baseline: {revision: "other"}}}), null);
  assert.match(extra.narrationText({...task, narration: null}), /调用链没有收益/);
});

test("new findings outrank older cited failures and stopped tasks have no next step", () => {
  const task = {id: "toy", desired: "paused", phase: "paused", spec: {baseline: {revision: "base"}}, narration: {
    schema: "narration-v1", task: "toy", baseline: "base", updated_at: 40, source_at: 30,
    sources: {old: {at: 10}, fresh: {at: 30}}, claims: {direction: null,
      finding: {text: "新结论", sources: ["fresh"]}, failure: {text: "旧故障", sources: ["old"]},
      next: {text: "旧计划", sources: ["old"]}}}};
  assert.equal(extra.narrationText(task), "新结论");
});


test("system fallback keeps its own source timestamp without a narrator", async () => {
  const {progressSource} = await import("../src/kernel_envolve/manager/static/summary.js");
  const t = {id: "toy", spec: {baseline: {revision: "base"}}, latest_validation: {
    schema: "validation-result-v1", task: "toy", promotion: false, baseline_revision: "base",
    status: "no_benefit", recorded_at: 42, summary: "no gain", scope: "full call",
    source_sha256: "a".repeat(64), evidence_sha256: "b".repeat(64)}};
  const source = progressSource(t);
  assert.equal(source.kind, "validation");
  assert.equal(source.at, 42);
  assert.match(source.text, /no gain/);
});
