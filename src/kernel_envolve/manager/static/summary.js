/* Presentation only: report prose never becomes validated performance evidence. */
export const executionLabel = (state) => ({running: "GPU 执行中", queued: "GPU 排队", idle: "无 GPU 作业"})[state] || "无 GPU 作业";
export const jobLabel = (status) => ({running: "执行中", succeeded: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断"})[status] || status || "状态未记录";
export const benefitLabel = (best) => typeof best?.baseline_speedup === "number" && Number.isFinite(best.baseline_speedup)
  ? `${best.baseline_speedup.toFixed(3)}×` : "尚无完整验收收益";
export function noteSnippet(value, fallback) {
  if (!value?.trim()) return fallback;
  const text = value.replace(/\b[a-f0-9]{16,64}\b/gi, (id) => `${id.slice(0, 8)}…`).replace(/\s+/g, " ").trim();
  return text.length > 180 ? `${text.slice(0, 179)}…` : text;
}
export function executionLabelFor(record) {
  if (!record) return "尚无执行记录";
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
  if (!record || record.evidence_error || record.correctness !== "pass") return label;
  if (record.mode === "quick" && typeof record.incremental_speedup === "number" && Number.isFinite(record.incremental_speedup))
    return `${label} · 局部信号 ${record.incremental_speedup.toFixed(3)}×（当前对照）`;
  return label;
}
export function phaseLabel(task) {
  if (task.stale) return "会话失联";
  return ({active: "持续优化", cooldown: "准备下一轮", resource_blocked: "资源异常",
    paused: "已停止", superseded: "历史版本", budget: "预算结束", blocked: "恢复失败",
    gpu_running: "GPU 执行", waiting_gpu: "等待 GPU", cpu_running: "本地诊断",
    agent_running: "会话在线", awaiting_progress: "暂无新实验", observing: "记录同步",
    recovering: "故障诊断", queued: "准备会话", backoff: "故障诊断退避", waiting_session: "等待会话"})[task.phase] || task.phase;
}

export function recentResult(task) {
  const recent = task.recent_execution, evaluated = task.recent_evaluation;
  // A wrapper exits after its evaluator writes the receipt: retain that actual result.
  if (recent?.kind === "gpu_job" && recent.status === "succeeded" && evaluated &&
      evaluated.recorded_at >= recent.started && evaluated.recorded_at <= recent.recorded_at)
    return evaluated;
  return recent;
}
