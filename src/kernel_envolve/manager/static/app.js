import { patch, poller } from "/sync.js";
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
const speed = (n) => (n == null ? "—" : `${n.toFixed(3)}×`);
const shortTime = (t) => t ? new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "—";
const mode = (value) => ({quick: "初筛", full: "完整评测"})[value] || value || "—";
const phases = {
  active: "持续优化",
  cooldown: "准备下一轮",
  resource_blocked: "资源异常",
  paused: "已停止",
  superseded: "历史版本",
  budget: "预算耗尽",
  blocked: "执行失败",
  gpu_running: "GPU 执行",
  waiting_gpu: "等待 GPU",
  agent_running: "会话在线",
  awaiting_progress: "会话在线",
  queued: "准备会话",
  backoff: "失败退避",
  waiting_session: "排队中",
};
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
function evidenceSummary(t) {
  const r = t.latest_evaluation;
  if (t.evidence_error || r?.evidence_error) return "证据异常";
  if (!r) return "—";
  const status = r.execution === "error" ? "执行失败"
    : r.correctness === "fail" ? "正确性失败"
    : r.correctness === "pass" ? "正确性通过" : "未完成";
  return `${escape(mode(r.mode))} · ${status}`;
}
function phaseLabel(t) {
  return t.stale ? "会话失联" : phases[t.phase] || t.phase;
}
function render() {
  const tasks = state.tasks;
  patch(
    $("metrics"),
    metric("任务", tasks.filter((t) => !["paused", "superseded", "budget"].includes(t.desired)).length) +
      metric(
        "GPU 执行",
        tasks.filter((t) => t.phase === "gpu_running").length,
      ) +
      metric(
        "GPU 排队",
        tasks.filter((t) => t.phase === "waiting_gpu").length,
      ) +
      metric(
        "GPU 作业用时",
        duration(tasks.reduce((s, t) => s + t.gpu_seconds, 0)),
      ),
  );
  const query = $("search").value.toLowerCase();
  const filtered = tasks.filter(
    (t) =>
      ($("show-history").checked || !["paused", "superseded", "budget"].includes(t.desired)) &&
      JSON.stringify([t.id, t.spec.name, t.spec.baseline, phases[t.phase]])
        .toLowerCase()
        .includes(query),
  );
  patch(
    $("tasks"),
    filtered
      .map((t) => {
        const updated = Math.max(t.report_as_of || 0, t.latest_evaluation?.recorded_at || 0, t.last_promotion || 0);
        return `<tr data-key="${escape(t.id)}"><td><button class="name" data-task="${escape(t.id)}">${escape(t.spec.name)}</button></td><td>${escape(phaseLabel(t))}</td><td>${evidenceSummary(t)}</td><td class="${t.best ? "good" : ""}">${speed(t.best?.baseline_speedup)}</td><td title="${escape(stamp(updated))}">${shortTime(updated)}</td></tr>`;
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
          `<article data-key="${escape(r.id)}" class="card"><div class="resource-head"><h3>${escape(r.id)}</h3><span>${r.problem ? "资源异常" : r.jobs.some(j => j.kind === "gpu" && j.status === "running") ? "执行中" : "空闲"}</span></div><p>GPU 作业 ${duration(r.gpu_seconds)}</p><div class="scroll"><table><thead><tr><th>任务</th><th>类型</th><th>状态</th><th>用时</th></tr></thead><tbody>${r.jobs.map(j => `<tr><td>${escape(j.task)}</td><td>${j.kind === "agent" ? "会话" : "GPU"}</td><td>${j.status === "running" ? "运行" : "排队"}</td><td>${duration(j.seconds)}</td></tr>`).join("") || '<tr><td colspan="4">—</td></tr>'}</tbody></table></div><details><summary>设备详情</summary><p><code>${escape(r.spec.device)}</code> · ${r.spec.sessions} 个会话</p>${r.problem ? `<pre>${escape(r.problem)}</pre>` : ""}</details></article>`,
      )
      .join("") || '<p class="empty">尚未配置计算资源。</p>',
  );
  renderStatus();
}
function renderStatus() {
  const live =
    state?.coordinator && state.as_of - state.coordinator.heartbeat < 5;
  $("updated").textContent = connected
    ? `${live ? "在线" : "离线"} · ${shortTime(state.as_of)}`
    : `连接中断 · ${shortTime(state?.as_of)}`;
  $("updated").classList.toggle("warn", !connected || !live);
  const problem = state?.coordinator?.error || state?.coordinator?.evidence_error;
  $("error").hidden = !problem;
  if (problem) { $("error").textContent = "服务异常"; $("error").title = String(problem); }
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
function renderDetail(t) {
  if (!t || t.id !== selected) return;
  const b = t.spec.baseline, r = t.latest_evaluation;
  const rows = t.best?.decision?.cases || [];
  const fields = (pairs) => pairs.filter(([, value]) => value).map(([label, value]) => `<dt>${label}</dt><dd>${escape(value)}</dd>`).join("");
  const diagnoses = {reason: t.reason, evidence_error: t.evidence_error, latest: r, recovery: t.recovery};
  patch($("detail"), `
    <div class="section-head"><h2>${escape(t.spec.name)}</h2><button id="close-detail">返回</button></div>
    <div class="detail-summary"><span>${escape(phaseLabel(t))}</span><span>${evidenceSummary(t)}</span><span>已验证加速 ${speed(t.best?.baseline_speedup)}</span></div>
    ${t.superseded_by ? `<p>后续任务：${escape(t.superseded_by)}</p>` : ""}
    <article class="card"><h3>研究记录</h3><dl>${fields([["方向", t.report.hypothesis], ["结论", t.report.outcome], ["下一步", t.report.next_action]])}${t.report.pr_urls?.length ? `<dt>PR</dt><dd>${t.report.pr_urls.map(u => link(u, u.replace("https://github.com/", "").replace("/pull/", "#"))).join("<br>")}</dd>` : ""}</dl></article>
    ${t.best ? `<article class="card"><h3>加速趋势</h3>${chart(t.history)}</article><details><summary>逐用例性能</summary><div class="scroll"><table><thead><tr><th>用例</th><th>批次</th><th>基线 μs</th><th>上一版 μs</th><th>最优 μs</th><th>加速</th></tr></thead><tbody>${rows.map(c => {
      const base = c.baseline_us ?? c.comparator_us;
      return `<tr><td>${escape(c.case)}${c.performance_role === "guard" ? '<small title="不计入总收益">护栏</small>' : ""}</td><td>${c.batch + 1}</td><td>${base?.toFixed(3) ?? "—"}</td><td>${c.comparator_us?.toFixed(3) ?? "—"}</td><td>${c.candidate_us?.toFixed(3) ?? "—"}</td><td>${base ? speed(base / c.candidate_us) : "—"}</td></tr>`;
    }).join("")}</tbody></table></div></details>` : ""}
    <article class="card"><h3>评测记录</h3><div class="scroll"><table><thead><tr><th>记录</th><th>模式</th><th>正确性</th><th>性能</th><th>详情</th></tr></thead><tbody>${t.runs.map(run => `<tr data-key="${escape(run.path)}"><td>${escape(run.path.split("/").pop().slice(0, 8))}</td><td>${escape(mode(run.data.mode))}</td><td>${escape(({pass: "通过", fail: "失败", partial: "未完成"})[run.data.correctness] || "—")}</td><td>${escape(({faster: "更快", slower: "更慢", inconclusive: "不确定", not_measured: "未测量"})[run.data.decision?.performance] || run.data.decision?.performance || "—")}</td><td><details><summary>查看</summary><pre>${escape(run.error || run.data.decision?.reasons?.join("\n") || "通过")}\n${escape(run.path)}</pre></details></td></tr>`).join("") || '<tr><td colspan="5">—</td></tr>'}</tbody></table></div></article>
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
