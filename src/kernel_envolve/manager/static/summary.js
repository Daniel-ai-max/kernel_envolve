/* Presentation only: report prose never becomes validated performance evidence. */
export const executionLabel = (state) => ({archived: "历史记录", running: "GPU 执行中", queued: "GPU 排队", idle: "无 GPU 作业"})[state] || "无 GPU 作业";
export const jobLabel = (status) => ({running: "执行中", succeeded: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断"})[status] || status || "状态未记录";
export const benefitLabel = (best) => typeof best?.baseline_speedup === "number" && Number.isFinite(best.baseline_speedup)
  ? `${best.baseline_speedup.toFixed(3)}×` : "—";
// Explanatory text never participates in taskBenefit, phase or progress clocks.
export function narrationFor(task, { findingOnly = false } = {}) {
  const n = task?.narration;
  if (!n || n.schema !== "narration-v1" || n.task !== task.id ||
      n.baseline !== task.spec?.baseline?.revision || !n.claims || !n.sources ||
      !Number.isFinite(n.updated_at) || !Number.isFinite(n.source_at)) return null;
  const fields = ["direction", "finding", "failure", "next"];
  if (Object.keys(n.claims).length !== fields.length || !fields.every(k => Object.hasOwn(n.claims, k))) return null;
  for (const [field, c] of Object.entries(n.claims)) {
    if (c !== null && (!c || typeof c.text !== "string" || !c.text.trim() || c.text.length > (field === "finding" ? 160 : 120) ||
        !Array.isArray(c.sources) || !c.sources.length || c.sources.some(id => !Object.hasOwn(n.sources, id)))) return null;
  }
  if (n.row) {
    const r = n.row, latest = r.latest_result;
    const current = task.row_facts;
    const sameEvidence = typeof n.evidence_revision === "string" && n.evidence_revision &&
      n.evidence_revision === current?.evidence_revision;
    if (!current || (r.revision !== current.revision && !(findingOnly && sameEvidence)) ||
        r.benefit !== "system:benefit" || r.optimization_pr !== "system:optimization_pr" ||
        r.current_progress !== "finding" || !n.claims.finding ||
        latest?.ref !== "system:latest_result" || typeof latest.text !== "string" ||
        !latest.text.trim() || latest.text.length > 120 || !Array.isArray(latest.sources) ||
        !latest.sources.length || latest.sources.some(id => !Object.hasOwn(n.sources, id))) return null;
  } else if (task.row_facts) return null; // Old prose cannot accompany a new atomic snapshot.
  return n;
}
export function narrationText(task) {
  return progressSource(task).text;
}
export function progressSource(task) {
  const n = narrationFor(task, {findingOnly: true});
  if (n?.row) return {text: n.claims.finding.text, kind: "narration",
    at: n.row.revision === task.row_facts.revision ? n.updated_at
      : Math.max(...n.claims.finding.sources.map(ref => n.sources[ref].at || 0))};
  if (task.row_facts && !n) {
    const recent = recentResult(task);
    if (recent) return {text: resultSummary(recent), kind: recent.kind, at: recent.recorded_at};
  }
  const validation = validationResult(task);
  const time = c => c ? Math.max(...c.sources.map(ref => n.sources[ref].at || 0)) : 0;
  const claims = n ? [n.claims.failure, n.claims.finding, n.claims.direction].filter(Boolean)
    .sort((a, b) => time(b) - time(a)) : [];
  // A freshly imported system fact wins over an older failure, including when
  // the model adapter is unavailable or retained its old failure claim.
  if (validation && (!claims.length || validation.recorded_at >= time(claims[0])))
    return {text: resultSummary(validation), kind: "validation", at: validation.recorded_at};
  return {text: n ? [claims[0], canContinue(task) ? n.claims.next : null]
    .filter(Boolean).map(c => c.text).join(" ") : "", kind: "narration", at: n?.updated_at};
}
export function canContinue(task) {
  return !["paused", "superseded"].includes(task.desired) && !task.archived &&
    !["archived", "paused", "budget", "blocked"].includes(task.phase);
}
export function validationResult(task) {
  const v = task?.latest_validation;
  if (!v || v.schema !== "validation-result-v1" || v.task !== task.id || v.promotion !== false ||
      v.baseline_revision !== task.spec?.baseline?.revision ||
      !["passed", "no_benefit", "failed", "interrupted", "inconclusive"].includes(v.status) ||
      !Number.isFinite(v.recorded_at) || v.recorded_at <= 0 ||
      typeof v.summary !== "string" || typeof v.scope !== "string" ||
      !/^[a-f0-9]{64}$/.test(v.source_sha256) || !/^[a-f0-9]{64}$/.test(v.evidence_sha256)) return null;
  return {...v, kind: "validation"};
}
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
// Host-checked provenance does not validate any numerical claim in this summary.
export function researchResult(task) {
  const r = task?.research_result;
  if (task.evidence_error || !r || r.schema !== "research-result-v1" || r.status !== "research" ||
      r.promotion !== false || r.task !== task.id || r.baseline_revision !== task.spec?.baseline?.revision ||
      !Number.isFinite(r.recorded_at) || r.recorded_at <= 0 ||
      typeof r.summary !== "string" || typeof r.scope !== "string" ||
      !/^[a-f0-9]{64}$/.test(r.source_sha256) || !/^[a-f0-9]{64}$/.test(r.evidence_sha256)) return null;
  return {...r, kind: "research_result"};
}
export function evidenceSummary(task) {
  const r = task.latest_evaluation;
  if (task.evidence_error || r?.evidence_error) return "证据异常";
  const validation = validationResult(task);
  if (validation && validation.recorded_at >= (r?.recorded_at || 0)) return executionLabelFor(validation);
  if (!r) return researchResult(task) ? "阶段结果" : "当前契约尚无评测记录";
  const status = r.execution === "error" ? "执行失败"
    : r.correctness === "fail" ? "正确性失败"
    : r.correctness === "pass" ? "正确性通过" : "未完成";
  return `${({quick: "初筛", full: "完整评测"})[r.mode] || "评测"} · ${status}`;
}
export function deliveredBest(task) {
  return task.spec?.kind === "research" && task.delivery?.status === "verified" ? task.delivery.best : null;
}
export function taskBenefit(task) {
  const delivered = deliveredBest(task);
  if (delivered) return {label: benefitLabel(delivered), detail: "已验证 native 交付", positive: delivered.baseline_speedup > 1};
  if (task.best) return {label: benefitLabel(task.best), detail: validationResult(task) ? "已保留最佳方案" : "", positive: task.best.baseline_speedup > 1};
  const a = scopedAssessment(task);
  // Research findings remain in progress/history, never in verified benefits.
  if (!a) return {label: benefitLabel(null), detail: "", positive: false};
  return {label: `${a.metrics[0].label} ${a.metrics[0].baseline_speedup.toFixed(3)}×`,
    detail: [a.scope_label, a.metrics.slice(1).map(m => `${m.label} ${m.baseline_speedup.toFixed(3)}×`).join(" · ")].filter(Boolean).join(" · "),
    positive: a.metrics[0].baseline_speedup > 1};
}
export function noteSnippet(value, fallback) {
  if (!value?.trim()) return fallback;
  const text = value.replace(/\b[a-f0-9]{16,64}\b/gi, (id) => `${id.slice(0, 8)}…`).replace(/\s+/g, " ").trim();
  return text.length > 180 ? `${text.slice(0, 179)}…` : text;
}
export function executionLabelFor(record) {
  if (!record) return "尚无执行记录";
  if (record.kind === "research_result") return "阶段结果";
  if (record.kind === "validation") return `验收 · ${({passed: "检查通过", no_benefit: "无收益", failed: "失败", interrupted: "中断", inconclusive: "无结论"})[record.status]}`;
  if (record.kind === "assessment_sync") return "验收同步失败";
  if (record.kind === "scoped_assessment") return "算子验收 · 正确性通过";
  if (record.kind === "cpu_request") return `本地诊断 · ${jobLabel(record.status)}`;
  if (record.kind === "gpu_job") return `GPU 作业 · ${jobLabel(record.status)}`;
  const mode = record.mode === "full" ? "完整评测" : record.mode === "quick" ? "初筛" : "评测";
  const result = record.evidence_error ? "证据异常" : record.execution === "error" ? "执行失败"
    : record.correctness === "pass" ? "正确性通过" : record.correctness === "fail" ? "正确性失败" : "未完成";
  return `${mode} · ${result}`;
}

export const visibleTask = (task, showHistory = false) => !task.delivery_parent && (showHistory || task.desired !== "superseded");

export function resultSummary(record) {
  const label = executionLabelFor(record);
  if (record?.kind === "research_result") return `阶段结果 · ${record.summary}`;
  if (record?.kind === "validation") return `${label} · ${record.summary}`;
  if (record?.kind === "scoped_assessment") return record.summary || label;
  if (!record || record.evidence_error || record.execution === "error" || record.correctness !== "pass") return label;
  if (record.mode === "quick" && typeof record.incremental_speedup === "number" && Number.isFinite(record.incremental_speedup))
    return `${label} · 局部信号 ${record.incremental_speedup.toFixed(3)}×（当前对照）`;
  return label;
}
export function phaseLabel(task) {
  if (deliveredBest(task) && task.delivery.policy_phase === "succeeded") return "已交付";
  if (task.stale) return "会话失联";
  if (task.phase === "paused" && task.reason === "task deadline reached") return "预算结束";
  return ({archived: "历史快照", active: "持续优化", cooldown: "准备下一轮", resource_blocked: "资源异常",
    paused: "已停止", superseded: "历史版本", budget: "预算结束", blocked: "恢复失败",
    gpu_running: "GPU 执行", waiting_gpu: "等待 GPU", cpu_running: "本地诊断",
    agent_running: "会话在线", awaiting_progress: "暂无新实验", observing: "记录同步",
    recovering: "故障诊断", queued: "准备会话", backoff: "故障诊断退避", waiting_session: "等待会话"})[task.phase] || task.phase;
}

export function recentResult(task) {
  const delivered = deliveredBest(task);
  if (delivered && delivered.recorded_at >= Math.max(task.latest_validation?.recorded_at || 0,
      task.research_result?.recorded_at || 0, task.recent_evaluation?.recorded_at || 0,
      task.verified_assessment?.recorded_at || 0))
    return {...delivered, task: task.delivery.task, kind: "evaluation"};
  const validation = validationResult(task);
  if (validation && validation.recorded_at >= Math.max(task.research_result?.recorded_at || 0,
      task.recent_evaluation?.recorded_at || 0, task.verified_assessment?.recorded_at || 0)) return validation;
  if (task.assessment_sync?.error && !researchResult(task)) return {kind: "assessment_sync", task: task.id, recorded_at: task.assessment_sync.recorded_at};
  const recent = task.recent_execution, evaluated = task.recent_evaluation;
  const assessed = scopedAssessment(task), research = researchResult(task);
  if (assessed && assessed.recorded_at >= Math.max(research?.recorded_at || 0, evaluated?.recorded_at || 0))
    return {kind: "scoped_assessment", task: task.id, recorded_at: assessed.recorded_at,
      summary: [`${assessed.metrics[0].cases} 场景验收`, ...assessed.metrics.filter(m => m.slower_cases > 0).map(m => `${m.label} ${m.slower_cases} 场景退化`)].join(" · ")};
  // Later cleanup/diagnostic jobs must not erase an already source-bound result.
  if (research && research.recorded_at >= (evaluated?.recorded_at || 0)) return research;
  // A wrapper exits after its evaluator writes the receipt: retain that actual result.
  if (recent?.kind === "gpu_job" && recent.status === "succeeded" && evaluated && recent.task === evaluated.task &&
      evaluated.recorded_at >= recent.started && evaluated.recorded_at <= recent.recorded_at)
    return evaluated;
  return recent;
}

// Formal evidence guards every model field. An earlier cited finding can survive
// exploratory completions, but latest_result requires the exact completion revision.
// Numbers, PR identity and execution status always come from this task snapshot.
export function narratedRow(task) {
  const n = narrationFor(task), recent = recentResult(task);
  return {benefit: taskBenefit(task), pr: deliveredBest(task) ? (task.delivery.implementation_pr?.current_head_verified ? task.delivery.implementation_pr : null) : task.implementation_pr,
    progress: progressSource(task), recent,
    outcomeAt: n?.row ? Math.max(...n.row.latest_result.sources.map(ref => n.sources[ref].at || 0)) : recent?.recorded_at,
    outcome: n?.row ? n.row.latest_result.text : resultSummary(recent)};
}
