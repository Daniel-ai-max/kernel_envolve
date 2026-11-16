/* Presentation only: report prose never becomes validated performance evidence. */
export const executionLabel = (state) => ({archived: "历史记录", running: "GPU 执行中", queued: "GPU 排队", idle: "无 GPU 作业"})[state] || "无 GPU 作业";
export const jobLabel = (status) => ({running: "执行中", succeeded: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断"})[status] || status || "状态未记录";
export const benefitLabel = (best) => typeof best?.baseline_speedup === "number" && Number.isFinite(best.baseline_speedup)
  ? `${best.baseline_speedup.toFixed(3)}×` : "尚无完整验收收益";
// Optional, server-verified scoped evidence in a read-only replay. Never a best
// promotion, a parsed agent report, or an external evaluator admission mechanism.
export function scopedAssessment(task) {
  const a = task?.verified_assessment;
  const digest = (s) => typeof s === "string" && /^[a-f0-9]{64}$/.test(s);
  const positive = (v) => typeof v === "number" && Number.isFinite(v) && v > 0;
  if (!a || a.schema !== "scoped-assessment-v1" || a.status !== "verified" ||
      a.promotion !== false || a.task !== task.id || task.evidence_error ||
      (typeof a.baseline_revision !== "string" || !a.baseline_revision.trim() || a.baseline_revision !== task.spec?.baseline?.revision) || !digest(a.source_sha256) ||
      !digest(a.evidence_sha256) || !positive(a.recorded_at) ||
      !Array.isArray(a.metrics) || !a.metrics.length || a.metrics.length > 8 ||
      !a.metrics.every(m => m && typeof m.label === "string" && m.label.trim() &&
        positive(m.baseline_speedup) && Number.isInteger(m.cases) && m.cases > 0 &&
        (m.slower_cases == null || (Number.isInteger(m.slower_cases) && m.slower_cases >= 0 && m.slower_cases <= m.cases)) &&
        (m.worst_slowdown_percent == null || (typeof m.worst_slowdown_percent === "number" && Number.isFinite(m.worst_slowdown_percent))))) return null;
  if (a.checkpoints != null && (!Array.isArray(a.checkpoints) || a.checkpoints.length > 100 ||
      !a.checkpoints.every(p => p && positive(p.minutes) && positive(p.eager_ratio) && positive(p.graph_ratio)))) return null;
  if (a.scenarios != null && (!Array.isArray(a.scenarios) || a.scenarios.length > 2000 ||
      !a.scenarios.every(c => c && typeof c.name === "string" && typeof c.mode === "string" &&
        positive(c.baseline_us) && positive(c.candidate_us) && positive(c.candidate_us / c.baseline_us) &&
        (c.group == null || ["leaf", "forward_backward"].includes(c.group))))) return null;
  return a;
}
export function taskBenefit(task) {
  if (task.best) return {label: benefitLabel(task.best), detail: "", positive: task.best.baseline_speedup > 1};
  const a = scopedAssessment(task);
  if (!a) return {label: benefitLabel(null), detail: "", positive: false};
  return {label: `${a.metrics[0].label} ${a.metrics[0].baseline_speedup.toFixed(3)}×`,
    detail: a.metrics.slice(1).map(m => `${m.label} ${m.baseline_speedup.toFixed(3)}×`).join(" · "),
    positive: a.metrics[0].baseline_speedup > 1};
}
export function noteSnippet(value, fallback) {
  if (!value?.trim()) return fallback;
  const text = value.replace(/\b[a-f0-9]{16,64}\b/gi, (id) => `${id.slice(0, 8)}…`).replace(/\s+/g, " ").trim();
  return text.length > 180 ? `${text.slice(0, 179)}…` : text;
}
export function executionLabelFor(record) {
  if (!record) return "尚无执行记录";
  if (record.kind === "assessment_sync") return "验收同步失败";
  if (record.kind === "scoped_assessment") return "算子验收 · 正确性通过";
  if (record.kind === "cpu_request") return `本地诊断 · ${jobLabel(record.status)}`;
  if (record.kind === "gpu_job") return `GPU 作业 · ${jobLabel(record.status)}`;
  const mode = record.mode === "full" ? "完整评测" : record.mode === "quick" ? "初筛" : "评测";
  const result = record.evidence_error ? "证据异常" : record.execution === "error" ? "执行失败"
    : record.correctness === "pass" ? "正确性通过" : record.correctness === "fail" ? "正确性失败" : "未完成";
  return `${mode} · ${result}`;
}

export const visibleTask = (task, showHistory = false) => showHistory || task.desired !== "superseded";

export function resultSummary(record) {
  const label = executionLabelFor(record);
  if (record?.kind === "scoped_assessment") return record.summary || label;
  if (!record || record.evidence_error || record.execution === "error" || record.correctness !== "pass") return label;
  if (record.mode === "quick" && typeof record.incremental_speedup === "number" && Number.isFinite(record.incremental_speedup))
    return `${label} · 局部信号 ${record.incremental_speedup.toFixed(3)}×（当前对照）`;
  return label;
}
export function phaseLabel(task) {
  if (task.stale) return "会话失联";
  if (task.phase === "paused" && task.reason === "task deadline reached") return "预算结束";
  return ({archived: "历史快照", active: "持续优化", cooldown: "准备下一轮", resource_blocked: "资源异常",
    paused: "已停止", superseded: "历史版本", budget: "预算结束", blocked: "恢复失败",
    gpu_running: "GPU 执行", waiting_gpu: "等待 GPU", cpu_running: "本地诊断",
    agent_running: "会话在线", awaiting_progress: "暂无新实验", observing: "记录同步",
    recovering: "故障诊断", queued: "准备会话", backoff: "故障诊断退避", waiting_session: "等待会话"})[task.phase] || task.phase;
}

export function recentResult(task) {
  if (task.assessment_sync?.error) return {kind: "assessment_sync", task: task.id, recorded_at: task.assessment_sync.recorded_at};
  const recent = task.recent_execution, evaluated = task.recent_evaluation;
  const assessed = scopedAssessment(task);
  if (assessed && assessed.recorded_at >= Math.max(recent?.recorded_at || 0, evaluated?.recorded_at || 0))
    return {kind: "scoped_assessment", task: task.id, recorded_at: assessed.recorded_at,
      summary: [`${assessed.metrics[0].cases} 场景验收`, ...assessed.metrics.filter(m => m.slower_cases > 0).map(m => `${m.label} ${m.slower_cases} 场景退化`)].join(" · ")};
  // A wrapper exits after its evaluator writes the receipt: retain that actual result.
  if (recent?.kind === "gpu_job" && recent.status === "succeeded" && evaluated && recent.task === evaluated.task &&
      evaluated.recorded_at >= recent.started && evaluated.recorded_at <= recent.recorded_at)
    return evaluated;
  return recent;
}
