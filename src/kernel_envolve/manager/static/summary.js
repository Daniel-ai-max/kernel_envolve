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
  if (record.kind === "gpu_job") return `GPU 作业 · ${jobLabel(record.status)}`;
  const mode = record.mode === "full" ? "完整评测" : record.mode === "quick" ? "初筛" : "评测";
  const result = record.evidence_error ? "证据异常" : record.execution === "error" ? "执行失败"
    : record.correctness === "pass" ? "正确性通过" : record.correctness === "fail" ? "正确性失败" : "未完成";
  return `${mode} · ${result}`;
}

export const visibleTask = (task, showHistory = false) => showHistory || task.desired !== "superseded";
