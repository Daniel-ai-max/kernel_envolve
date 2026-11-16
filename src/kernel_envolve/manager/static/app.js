import { patch, poller } from "/sync.js";
import { executionLabel, jobLabel, noteSnippet, executionLabelFor, visibleTask, resultSummary, phaseLabel, recentResult, scopedAssessment, taskBenefit, researchResult, evidenceSummary } from "/summary.js";
/* All external text passes through textContent or escaping. No HTML from agent reports. */
const $ = (id) => document.getElementById(id);
const escape = (s) =>
  String(s ?? "—").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const stamp = (t) => (t ? new Date(t * 1000).toLocaleString() : "—");
const duration = (s) =>
  s == null
    ? "—"
    : s < 60
      ? `${s.toFixed(1)} s`
      : s < 3600
        ? `${(s / 60).toFixed(1)} min`
        : `${(s / 3600).toFixed(2)} h`;
const cardHours = (usage) => usage ? `${usage.gpu_hours.toFixed(2)} 卡时` : "—";
const speed = (n) => (n == null ? "—" : `${n.toFixed(3)}×`);
const shortTime = (t) => t ? new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "—";
const mode = (value) => ({quick: "初筛", full: "完整评测"})[value] || value || "—";
let state = null,
  selected = null,
  view = "overview",
  connected = false;
const link = (url, label) =>
  typeof url === "string" && /^https:\/\//.test(url)
    ? `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)}</a>`
    : escape(label);
async function request(url, signal) {
  const response = await fetch(url, {signal: AbortSignal.any([
    AbortSignal.timeout(8000), ...(signal ? [signal] : []),
  ])});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}
function render() {
  const tasks = state.tasks;
  patch(
    $("metrics"),
    metric("任务", tasks.filter((t) => t.desired !== "superseded").length) +
      metric(
        "GPU 执行",
        tasks.filter((t) => t.phase === "gpu_running").length,
      ) +
      metric(
        "GPU 排队",
        tasks.filter((t) => t.phase === "waiting_gpu").length,
      ) +
      metric(
        escape(state.allocation_usage?.label || "消耗卡时"),
        cardHours(state.allocation_usage) + (state.allocation_usage?.scope_label ? `<small>${escape(state.allocation_usage.scope_label)}</small>` : ""),
      ),
  );
  const query = $("search").value.toLowerCase();
  const filtered = tasks.filter(
    (t) =>
      visibleTask(t, $("show-history").checked) &&
      JSON.stringify([t.id, t.spec.name, t.spec.baseline, t.spec.resource, phaseLabel(t), t.research_record, executionLabel(t.execution_state)])
        .toLowerCase()
        .includes(query),
  );
  patch(
    $("tasks"),
    filtered
      .map((t) => {
        const record = t.research_record, recent = recentResult(t), baseline = t.spec.baseline;
        const outcome = resultSummary(recent), benefit = taskBenefit(t), assessment = scopedAssessment(t);
        const direction = noteSnippet(t.current_work || record?.hypothesis, assessment ? "验收完成" : phaseLabel(t));
        const next = noteSnippet(record?.next_action, "");
        return `<tr data-key="${escape(t.id)}">
          <td data-label="算子"><button class="name" data-task="${escape(t.id)}">${escape(t.spec.name)}</button>${t.spec.kind === "research" ? `<small>${assessment ? "探索 · 已有范围验收" : "探索 · 未接入验收"}</small>` : ""}</td>
          <td data-label="阶段">${escape(phaseLabel(t))}</td>
          <td data-label="基线"><div title="${escape(baseline.revision)}"><code>${escape(baseline.revision.slice(0, 8))}</code></div><small>${link(baseline.repository, baseline.repository.split("/").filter(Boolean).pop())}</small></td>
          <td class="benefit ${benefit.positive ? "good" : benefit.research ? "stage-benefit" : "muted"}" data-label="收益">${benefit.research ? `<small>${escape(benefit.detail)}</small>` : ""}${escape(benefit.label)}${!benefit.research && benefit.detail ? `<small>${escape(benefit.detail)}</small>` : ""}</td>
          <td data-label="最新实现">${t.implementation_pr ? `${link(t.implementation_pr.url, "PR #" + t.implementation_pr.url.split("/").pop())}${t.implementation_pr.current_head_verified === false ? `<small>${link(t.implementation_pr.commit_url, "已验 " + t.implementation_pr.head.slice(0, 7))}</small>` : ""}<small>${escape(stamp(t.implementation_pr.updated_at))}</small>` : "—"}</td>
          <td data-label="消耗卡时" title="${escape(t.allocation_usage ? `含准备与空闲占卡 · 截至 ${stamp(t.allocation_usage.observed_at)}` : "尚无占卡用量记录")}">${cardHours(t.allocation_usage)}${t.allocation_usage?.scope_label ? `<small>${escape(t.allocation_usage.scope_label)}</small>` : ""}</td>
          <td class="research" data-label="当前工作"><div class="snippet" title="${escape(t.current_work || record?.hypothesis || "")}">${escape(direction)}</div>${next ? `<small class="snippet" title="${escape(record.next_action)}">下一步：${escape(next)}</small>` : ""}</td>
          <td class="research" data-label="最近结果"><div class="snippet" title="${escape(recent ? resultSummary(recent) : "")}">${escape(outcome)}</div>${recent ? `<small><button class="source-link execution-link" data-task="${escape(recent.task)}" title="${escape(stamp(recent.recorded_at))}">${escape(executionLabelFor(recent))} · ${shortTime(recent.recorded_at)}${recent.inherited ? " · 历史" : ""}</button></small>` : ""}</td>
        </tr>`;
      })
      .join(""),
  );
  $("empty").hidden = filtered.length > 0;
  $("empty").textContent = tasks.length ? "没有匹配的任务" : "尚未注册任务";
  patch(
    $("resource-list"),
    state.resources
      .map(
        (r) =>
          `<article data-key="${escape(r.id)}" class="card"><div class="resource-head"><h3>${escape(r.id)}</h3><span>${r.problem ? "资源异常" : r.archived ? "历史记录" : r.released ? "已回收" : r.jobs.some(j => j.kind === "gpu" && j.status === "running") ? "执行中" : "空闲"}</span></div><p>${r.allocation_usage ? cardHours(r.allocation_usage) : `GPU 作业 ${duration(r.gpu_seconds)}`}</p><div class="scroll"><table><thead><tr><th>任务</th><th>类型</th><th>状态</th><th>用时</th></tr></thead><tbody>${r.jobs.map(j => `<tr><td>${escape(j.task)}</td><td>${j.kind === "agent" ? "会话" : "GPU"}</td><td>${j.status === "running" ? "运行" : "排队"}</td><td>${duration(j.seconds)}</td></tr>`).join("") || '<tr><td colspan="4">—</td></tr>'}</tbody></table></div><details><summary>设备详情</summary><p><code>${escape(r.spec.devices ? r.spec.devices.join(", ") : r.spec.device)}</code> · ${r.spec.sessions} 个会话</p>${r.problem ? `<pre>${escape(r.problem)}</pre>` : ""}</details></article>`,
      )
      .join("") || '<p class="empty">尚未配置计算资源。</p>',
  );
  renderStatus();
}
function renderStatus() {
  if (state?.archived) {
    $("updated").textContent = connected ? `归档记录 · ${stamp(state.as_of)}` : `连接中断 · ${shortTime(state.as_of)}`;
    $("updated").classList.toggle("warn", !connected);
    $("error").hidden = true;
    return;
  }
  const live =
    state?.coordinator && state.as_of - state.coordinator.heartbeat < 5;
  const observing = state?.observer && state.as_of - state.observer.heartbeat < 5;
  $("updated").textContent = connected
    ? (live ? `在线 · ${shortTime(state.as_of)}` : observing ? `同步在线 · ${shortTime(state.as_of)}` : `离线 · 最后观测 ${stamp(state?.coordinator?.heartbeat)}`)
    : `连接中断 · ${shortTime(state?.as_of)}`;
  $("updated").classList.toggle("warn", !connected || (!live && !observing));
  const problem = state?.archive_status?.error || (live ? state?.coordinator : state?.observer)?.error || (live ? state?.coordinator : state?.observer)?.evidence_error;
  $("error").hidden = !problem;
  if (problem) { $("error").textContent = state?.archive_status?.error ? "归档保存失败" : "服务异常"; $("error").title = String(problem); }
}

function chart(history) {
  const points = history.filter(
    (h) => h.promoted_at && h.baseline_speedup && !h.evidence_error,
  );
  if (!points.length) return '<p class="empty">暂无验证收益</p>';
  const w = 780,
    h = 210,
    l = 65,
    r = 22,
    t = 20,
    b = 38;
  const min = Math.min(...points.map((p) => p.promoted_at)),
    max = Math.max(min + 60, ...points.map((p) => p.promoted_at));
  const low = Math.min(1, ...points.map((p) => p.baseline_speedup)),
    high =
      Math.max(low + 0.05, ...points.map((p) => p.baseline_speedup)) * 1.03;
  const x = (p) => l + ((p.promoted_at - min) / (max - min)) * (w - l - r),
    y = (p) => t + ((high - p.baseline_speedup) / (high - low)) * (h - t - b);
  let path = `M${x(points[0])},${y(points[0])}`;
  for (let i = 1; i < points.length; i++)
    path += ` H${x(points[i])} V${y(points[i])}`;
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="已验证 best 相对固定 baseline 的加速历史"><path class="axis" d="M${l},${t} V${h - b} H${w - r}"/><text x="0" y="${t + 8}">${high.toFixed(2)}×</text><text x="0" y="${h - b}">${low.toFixed(2)}×</text><text x="${l}" y="${h - 8}">${escape(new Date(min * 1000).toLocaleString())}</text><text text-anchor="end" x="${w - r}" y="${h - 8}">${escape(new Date(max * 1000).toLocaleString())}</text><path class="step" d="${path}"/>${points.map((p) => `<circle cx="${x(p)}" cy="${y(p)}" r="4"><title>${escape(stamp(p.promoted_at))} · ${speed(p.baseline_speedup)}</title></circle>`).join("")}</svg>`;
}
function assessmentDetail(t) {
  const a = scopedAssessment(t);
  if (!a) return "";
  const pct = (ratio) => `${(100 * (1 - ratio)).toFixed(2)}%`;
  const history = Array.isArray(a.checkpoints) ? a.checkpoints : [];
  const cases = Array.isArray(a.scenarios) ? a.scenarios : [];
  const native = cases.filter(c => c.group === "forward_backward");
  return `<article class="card assessment"><h3>算子验收</h3><p>${escape(a.scope_label || "算子")} · ${escape(stamp(a.recorded_at))} · 未晋升全链</p>
    <div class="scroll"><table><thead><tr><th>计时方式</th><th>加速</th><th>场景</th><th>退化场景</th><th>最大退化</th></tr></thead><tbody>${a.metrics.map(m => `<tr><td>${escape(m.label)}</td><td>${speed(m.baseline_speedup)}</td><td>${m.cases}</td><td>${Number.isInteger(m.slower_cases) ? m.slower_cases : "—"}</td><td>${typeof m.worst_slowdown_percent === "number" ? Math.max(0, m.worst_slowdown_percent).toFixed(2) + "%" : "—"}</td></tr>`).join("")}</tbody></table></div>
    ${native.length ? `<h3>原生前反向</h3><div class="scroll"><table><thead><tr><th>场景</th><th>延迟下降</th></tr></thead><tbody>${native.map(c => `<tr><td>${escape(c.name)}</td><td>${pct(c.candidate_us / c.baseline_us)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    ${history.length ? `<h3>研究进展</h3><div class="scroll"><table><thead><tr><th>研究时间</th><th>普通调用延迟下降</th><th>Graph 延迟下降</th></tr></thead><tbody>${history.map(p => `<tr><td>${escape(p.minutes)} 分钟</td><td>${pct(p.eager_ratio)}</td><td>${pct(p.graph_ratio)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    ${cases.length ? `<details><summary>逐场景性能</summary><div class="scroll"><table><thead><tr><th>场景</th><th>方式</th><th>基线 μs</th><th>候选 μs</th><th>延迟下降</th></tr></thead><tbody>${cases.map(c => `<tr><td>${escape(c.name)}</td><td>${escape(c.mode)}</td><td>${c.baseline_us.toFixed(3)}</td><td>${c.candidate_us.toFixed(3)}</td><td>${pct(c.candidate_us / c.baseline_us)}</td></tr>`).join("")}</tbody></table></div></details>` : ""}
    <details><summary>验收来源</summary><pre>${escape(JSON.stringify({source: a.source_sha256, evidence: a.evidence_sha256, qualification: a.qualification}, null, 2))}</pre></details></article>`;
}
function researchRows(t) {
  return (t.research_result_history || []).map(record => {
    const r = researchResult({...t, research_result: record.research_result});
    if (!r) return "";
    return `<tr><td>${escape(stamp(r.recorded_at))}</td><td>阶段结果</td><td>未完整验收</td><td>${escape(r.summary)}</td><td><details><summary>范围与来源</summary><p>${escape(r.scope)}</p><p>${escape(r.limitations.join("；"))}</p><small>实验 ${escape(stamp(r.recorded_at))} · 追记 ${escape(stamp(record.indexed_at))}</small><pre>${escape(JSON.stringify({source: r.source_sha256, evidence: r.evidence_sha256, binding: r.binding}, null, 2))}</pre></details></td></tr>`;
  }).join("");
}
function renderDetail(t) {
  if (!t || t.id !== selected) return;
  const b = t.spec.baseline, r = t.latest_evaluation, record = t.research_record, assessment = scopedAssessment(t), benefit = taskBenefit(t);
  const rows = t.best?.decision?.cases || [];
  const fields = (pairs) => pairs.filter(([, value]) => value).map(([label, value]) => `<dt>${label}</dt><dd>${escape(value)}</dd>`).join("");
  const diagnoses = {session: phaseLabel(t), reason: t.reason, evidence_error: t.evidence_error, latest: r, recovery: t.recovery, current_work_argv: t.current_work_argv, assessment_sync: t.assessment_sync};
  patch($("detail"), `
    <div class="section-head"><h2>${escape(t.spec.name)}</h2><button id="close-detail">返回</button></div>
    <div class="detail-summary"><span>${executionLabel(t.execution_state)}</span><span>${assessment ? "算子验收通过" : evidenceSummary(t)}</span><span>${escape(benefit.label)}</span></div>
    ${t.superseded_by ? `<p>历史契约 · 后续任务：<button class="source-link" data-task="${escape(t.superseded_by)}">${escape(t.superseded_by)}</button></p>` : ""}
    ${assessmentDetail(t)}
    ${t.research_result_sync?.error ? `<p class="warn">阶段结果同步失败 · ${escape(t.research_result_sync.error)}</p>` : ""}
    ${t.assessment_sync?.error ? `<p class="warn">验收同步失败 · ${escape(t.assessment_sync.error)}</p>` : ""}
    ${t.allocation_usage?.breakdown ? `<article class="card"><h3>卡时</h3><p>${escape(t.allocation_usage.breakdown)}</p></article>` : ""}
    <article class="card"><h3>当前工作</h3><p>${escape(t.current_work || (assessment ? "验收完成" : phaseLabel(t)))}</p><p>${escape(resultSummary(recentResult(t)))}</p>${record?.inherited ? `<p>来源：<button class="source-link" data-task="${escape(record.task)}">${escape(record.task)}</button>（历史契约）</p>` : ""}<dl>${fields([["记录时间", record ? stamp(record.recorded_at) : ""], ["方向", record?.hypothesis], ["结论", record?.outcome], ["下一步", record?.next_action]])}${t.report.pr_urls?.length ? `<dt>PR</dt><dd>${t.report.pr_urls.map(u => link(u, u.replace("https://github.com/", "").replace("/pull/", "#"))).join("<br>")}</dd>` : ""}</dl></article>
    ${t.best ? `<article class="card"><h3>加速趋势</h3>${chart(t.history)}</article><details><summary>逐用例性能</summary><div class="scroll"><table><thead><tr><th>用例</th><th>批次</th><th>基线 μs</th><th>上一版 μs</th><th>最优 μs</th><th>加速</th></tr></thead><tbody>${rows.map(c => {
      const base = c.baseline_us ?? c.comparator_us;
      return `<tr><td>${escape(c.case)}${c.performance_role === "guard" ? '<small title="不计入总收益">护栏</small>' : ""}</td><td>${c.batch + 1}</td><td>${base?.toFixed(3) ?? "—"}</td><td>${c.comparator_us?.toFixed(3) ?? "—"}</td><td>${c.candidate_us?.toFixed(3) ?? "—"}</td><td>${base ? speed(base / c.candidate_us) : "—"}</td></tr>`;
    }).join("")}</tbody></table></div></details>` : ""}
    <article class="card"><h3>评测记录</h3><div class="scroll"><table><thead><tr><th>记录</th><th>模式</th><th>正确性</th><th>性能</th><th>详情</th></tr></thead><tbody>${researchRows(t)}${t.runs.map(run => `<tr data-key="${escape(run.path)}"><td>${escape(run.path.split("/").pop().slice(0, 8))}</td><td>${escape(mode(run.data.mode))}</td><td>${escape(({pass: "通过", fail: "失败", partial: "未完成"})[run.data.correctness] || "—")}</td><td>${escape(({faster: "更快", slower: "更慢", inconclusive: "不确定", not_measured: "未测量"})[run.data.decision?.performance] || run.data.decision?.performance || "—")}</td><td><details><summary>查看</summary><pre>${escape(run.error || run.data.decision?.reasons?.join("\n") || "通过")}\n${escape(run.path)}</pre></details></td></tr>`).join("") || (researchRows(t) ? "" : `<tr><td colspan="5">${assessment ? '尚无可晋升的完整契约评测' : '当前契约尚无评测记录，可在历史来源查看此前记录'}</td></tr>`)}</tbody></table></div></article>
    <article class="card"><h3>历史来源</h3>${t.history_complete ? "" : '<p>迁移链异常，已停止追溯无法确认归属的历史记录。</p>'}<div class="lineage">${t.lineage.map(row => `<div><button class="source-link" data-task="${escape(row.task)}">${escape(row.task)}</button><span>${row.inherited ? "历史契约" : "当前契约"} · ${row.run_count} 条评测记录 · ${row.gpu_job_count} 次 GPU 作业</span></div>`).join("")}</div></article>
    <article class="card"><h3>GPU 作业记录</h3><div class="scroll"><table><thead><tr><th>作业</th><th>来源任务</th><th>状态</th><th>开始 / 结束</th><th>用时</th></tr></thead><tbody>${t.gpu_history.map(job => `<tr data-key="${escape(job.id)}"><td><code>${escape(job.id)}</code>${job.error ? `<small>${escape(job.error)}</small>` : ""}</td><td><button class="source-link" data-task="${escape(job.task)}">${escape(job.task)}</button></td><td>${escape(jobLabel(job.status))}</td><td>${stamp(job.started)}<br>${job.ended ? stamp(job.ended) : "尚未结束"}</td><td>${duration(job.seconds)}</td></tr>`).join("") || '<tr><td colspan="5">尚无已启动的 GPU 作业</td></tr>'}</tbody></table></div></article>
    <details><summary>基线与配置</summary><dl>${fields([["仓库", b.repository], ["版本", b.revision], ["入口", b.entrypoint], ["提出日期", b.introduced_on], ["冻结日期", b.frozen_on], ["硬件", b.hardware], ["契约", b.contract], ["已集成", t.report.integrated_revision]])}<dt>会话</dt><dd>${link(t.report.session_url, t.report.session_url || "—")}</dd></dl></details>
    <details><summary>用量与预算</summary><p>会话 ${duration(t.wall_seconds)} / ${duration(t.spec.wall_budget_seconds)} · GPU 作业 ${duration(t.gpu_seconds)} / ${duration(t.spec.gpu_budget_seconds)}</p></details>
    <details><summary>诊断信息</summary><pre>${escape(JSON.stringify(diagnoses, null, 2))}</pre></details>
    <details><summary>版本信息</summary><pre>${escape(JSON.stringify({candidate: t.runs[0]?.data.source ?? null, best: t.best?.source ?? null}, null, 2))}</pre></details>
    <details><summary>研究笔记</summary><pre>${escape(t.notes || "—")}</pre></details>
    <details><summary>事件记录</summary>${t.events.map(e => `<div data-key="${e.seq}" class="event"><time>${stamp(e.at)}</time><div>${escape(e.kind)}</div><div>${escape(JSON.stringify(e.data))}</div></div>`).join("")}</details>`);
}
function switchView(next) {
  view = next;
  selected = null;
  $("detail").hidden = true;
  $("overview").hidden = next !== "overview";
  $("resources").hidden = next !== "resources";
  document
    .querySelectorAll("[data-view]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.view === next)),
    );
}
const updates = poller({
  read: (signal) =>
    request(
      `/api/state${selected ? `?task=${encodeURIComponent(selected)}` : ""}`,
      signal,
    ),
  apply: (data) => {
    state = data;
    connected = true;
    render();
    if (selected) renderDetail(data.detail);
  },
  failed: () => {
    connected = false;
    renderStatus();
  },
  visible: () => !document.hidden,
});
document.addEventListener("click", async (event) => {
  const b = event.target.closest("button");
  if (!b) return;
  if (b.dataset.view) {
    switchView(b.dataset.view);
    updates.refresh();
  } else if (b.id === "refresh") updates.refresh();
  else if (b.dataset.task) {
    selected = b.dataset.task;
      $("overview").hidden = true;
    $("resources").hidden = true;
    $("detail").hidden = false;
    $("detail").textContent = "读取中…";
    updates.refresh();
  } else if (b.id === "close-detail") {
    switchView(view);
    updates.refresh();
  }
});
document.addEventListener("change", async (event) => {
  if (event.target.id === "show-history") render();

});
$("search").addEventListener("input", () => state && render());
document.addEventListener("visibilitychange", () =>
  document.hidden ? updates.stop() : updates.refresh(),
);
window.addEventListener("online", () => updates.refresh());
updates.refresh();
