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
const speed = (n) => (n == null ? "尚无晋升" : `${n.toFixed(3)}×`);
const phases = {
  active: "持续优化",
  cooldown: "准备下一轮",
  resource_blocked: "资源待检查",
  paused: "用户暂停",
  budget: "预算耗尽",
  blocked: "等待处理",
  gpu_running: "GPU 作业中",
  waiting_gpu: "等待 GPU",
  agent_running: "Agent 工作中",
  queued: "准备会话",
  backoff: "失败退避",
  waiting_session: "等待会话名额",
};
let state = null,
  selected = null,
  view = "overview",
  busy = false;
const link = (url, label) =>
  typeof url === "string" && /^https:\/\//.test(url)
    ? `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)}</a>`
    : escape(label);
async function request(url, body) {
  const response = await fetch(
    url,
    body
      ? {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Manager-Token": state.token,
          },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function error(e) {
  $("error").hidden = false;
  $("error").textContent = e.message;
}
function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}
function render() {
  const tasks = state.tasks;
  $("metrics").innerHTML =
    metric("持续任务", tasks.length) +
    metric(
      "运行中",
      tasks.filter((t) =>
        t.jobs.some((j) => j.status === "running" && j.kind === "agent"),
      ).length,
    ) +
    metric("等待 GPU", tasks.filter((t) => t.phase === "waiting_gpu").length) +
    metric(
      "需要处理",
      tasks.filter(
        (t) => t.desired === "blocked" || t.evidence_error || t.stale,
      ).length,
    ) +
    metric(
      "GPU 作业累计",
      duration(tasks.reduce((s, t) => s + t.gpu_seconds, 0)),
    );
  const query = $("search").value.toLowerCase();
  const filtered = tasks.filter((t) =>
    JSON.stringify([t.id, t.spec.name, t.spec.baseline, phases[t.phase]])
      .toLowerCase()
      .includes(query),
  );
  $("tasks").innerHTML = filtered
    .map((t) => {
      const b = t.spec.baseline,
        d = t.best?.decision,
        coverage = d?.coverage?.correctness_cases?.length;
      return `<tr><td><button class="name" data-task="${escape(t.id)}">${escape(t.spec.name)}</button><small>${escape(b.hardware)}</small></td><td class="${t.stale || t.evidence_error ? "warn" : ""}">${escape(phases[t.phase])}${t.report.phase ? `<small>${escape(t.report.phase)}</small>` : ""}${t.stale ? "<small>心跳待核实</small>" : ""}${t.evidence_error ? "<small>评测证据需检查</small>" : ""}</td><td><code>${escape(b.revision.length > 14 ? b.revision.slice(0, 10) : b.revision)}</code><small>${escape(b.repository.replace(/^https:\/\/github.com\//, ""))}</small></td><td>${link(b.introduction_url, b.introduced_on || "待确认")}</td><td class="${t.best ? "good" : ""}">${speed(t.best?.baseline_speedup)}<small>${t.best ? `本次 ${speed(t.best.incremental_speedup)}` : ""}</small></td><td>${stamp(t.last_promotion)}</td><td>${coverage == null ? "—" : `${coverage} 个正确性用例`}<small>${escape(t.best?.native_checks?.map((c) => `${c.scope}: ${c.status}`).join(" / ") || "")}</small></td><td>${escape(t.spec.resource)}</td><td>${escape(t.report.hypothesis || "—")}</td></tr>`;
    })
    .join("");
  $("empty").hidden = filtered.length > 0;
  $("empty").textContent = tasks.length ? "没有匹配的任务" : "尚未注册任务";
  $("resource-list").innerHTML =
    state.resources
      .map(
        (r) =>
          `<article class="card"><div class="resource-head"><h3>${escape(r.id)} <span class="pill">${r.spec.sessions} sessions</span></h3><select aria-label="${escape(r.id)} 会话数" data-sessions="${escape(r.id)}"><option value="1" ${r.spec.sessions === 1 ? "selected" : ""}>1 个会话</option><option value="2" ${r.spec.sessions === 2 ? "selected" : ""}>2 个会话</option></select></div><p><code>${escape(r.spec.device)}</code> · GPU 作业 ${duration(r.gpu_seconds)}</p>${r.problem ? `<p class="bad">${escape(r.problem)}</p><button data-clear="${escape(r.id)}">检查设备并解除隔离</button>` : ""}<div class="scroll"><table><thead><tr><th>任务</th><th>工作类型</th><th>状态</th><th>占用时间</th><th>等待原因</th></tr></thead><tbody>${r.jobs.map((j) => `<tr><td>${escape(j.task)}</td><td>${j.kind === "agent" ? "Agent 会话" : "GPU 作业"}</td><td>${escape(j.status)}</td><td>${duration(j.seconds)}</td><td>${escape(j.error || "—")}</td></tr>`).join("") || '<tr><td colspan="5">空闲</td></tr>'}</tbody></table></div></article>`,
      )
      .join("") || '<p class="empty">尚未配置计算资源。</p>';
  const live =
    state.coordinator && state.as_of - state.coordinator.heartbeat < 5;
  $("updated").textContent =
    `${live ? "协调器在线" : "协调器离线 / 未启动"} · 数据更新 ${stamp(state.as_of)}`;
  if (state.coordinator?.error || state.coordinator?.evidence_error)
    error(
      new Error(state.coordinator.error || state.coordinator.evidence_error),
    );
}
function chart(history) {
  const points = history.filter(
    (h) => h.promoted_at && h.baseline_speedup && !h.evidence_error,
  );
  if (!points.length) return '<p class="empty">暂无晋升历史</p>';
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
async function renderDetail() {
  const id = selected;
  if (!id) return;
  const t = await request(`/api/tasks/${encodeURIComponent(id)}`);
  if (id !== selected) return;
  const b = t.spec.baseline;
  const rows = t.best?.decision?.cases || [];
  $("detail").innerHTML =
    `<div class="section-head"><h2>${escape(t.spec.name)}</h2><button id="close-detail">返回总览</button></div><div class="controls"><button data-action="pause" ${t.desired === "paused" ? "disabled" : ""}>暂停</button><button data-action="resume" ${t.desired === "active" ? "disabled" : ""}>恢复</button><span>${escape(phases[t.phase])} ${escape(t.reason || "")}</span></div>${t.evidence_error ? `<p class="warn">证据问题：${escape(t.evidence_error)}</p>` : ""}<div class="grid"><article class="card"><h3>固定 baseline</h3><dl><dt>仓库 / 版本</dt><dd>${escape(b.repository)} · <code>${escape(b.revision)}</code></dd><dt>算子入口</dt><dd><code>${escape(b.entrypoint)}</code></dd><dt>实现提出</dt><dd>${link(b.introduction_url, b.introduced_on || "待确认")}</dd><dt>冻结日期</dt><dd>${escape(b.frozen_on)}</dd><dt>目标硬件</dt><dd>${escape(b.hardware)}</dd><dt>评测契约</dt><dd><code>${escape(b.contract)}</code></dd></dl></article><article class="card"><h3>研究进展</h3><dl><dt>当前方向</dt><dd>${escape(t.report.hypothesis || "—")}</dd><dt>最近结论</dt><dd>${escape(t.report.outcome || "—")}</dd><dt>下一步</dt><dd>${escape(t.report.next_action || "—")}</dd><dt>会话</dt><dd>${link(t.report.session_url, t.report.session_url || "—")}</dd><dt>PR</dt><dd>${(t.report.pr_urls || []).map((u) => link(u, u)).join("<br>") || "—"}</dd><dt>已集成版本</dt><dd>${escape(t.report.integrated_revision || "—")}</dd></dl></article></div><article class="card"><h3>已验证 best · 相对 baseline</h3>${chart(t.history)}</article><article class="card"><h3>当前 best · 逐用例原始延迟</h3><div class="scroll"><table><thead><tr><th>用例</th><th>测量批次</th><th>baseline μs</th><th>上一版对照 μs</th><th>best μs</th><th>相对 baseline</th></tr></thead><tbody>${
      rows
        .map((c) => {
          const base = c.baseline_us ?? c.comparator_us;
          return `<tr><td>${escape(c.case)}</td><td>${c.batch + 1}</td><td>${base?.toFixed(3) ?? "—"}</td><td>${c.comparator_us?.toFixed(3) ?? "—"}</td><td>${c.candidate_us?.toFixed(3) ?? "—"}</td><td class="${c.candidate_us > base ? "warn" : "good"}">${base ? (base / c.candidate_us).toFixed(3) + "×" : "—"}</td></tr>`;
        })
        .join("") || '<tr><td colspan="6">尚无已晋升的完整测量</td></tr>'
    }</tbody></table></div></article><article class="card"><h3>资源与累计额度</h3><p>会话 ${duration(t.wall_seconds)} / ${t.spec.wall_budget_seconds == null ? "未设上限" : duration(t.spec.wall_budget_seconds)} · GPU 作业 ${duration(t.gpu_seconds)} / ${t.spec.gpu_budget_seconds == null ? "未设上限" : duration(t.spec.gpu_budget_seconds)}</p><form id="configure" class="controls"><label>资源 <select name="resource">${state.resources.map((r) => `<option value="${escape(r.id)}" ${r.id === t.spec.resource ? "selected" : ""}>${escape(r.id)}</option>`).join("")}</select></label><label>会话上限（小时） <input name="wall" type="number" min="0.001" step="any" value="${t.spec.wall_budget_seconds == null ? "" : t.spec.wall_budget_seconds / 3600}" placeholder="不设上限"></label><label>GPU 上限（小时） <input name="gpu" type="number" min="0.001" step="any" value="${t.spec.gpu_budget_seconds == null ? "" : t.spec.gpu_budget_seconds / 3600}" placeholder="不设上限"></label><button title="暂停任务并等待作业退出后可修改" ${t.desired !== "paused" || t.jobs.length ? "disabled" : ""}>保存设置</button></form></article><article class="card"><h3>最近实验</h3><div class="scroll"><table><thead><tr><th>记录</th><th>模式</th><th>正确性</th><th>性能结论</th><th>证据 / 原因</th></tr></thead><tbody>${t.runs.map((r) => `<tr><td>${escape(r.path.split("/").pop())}</td><td>${escape(r.data.mode)}</td><td>${escape(r.data.correctness)}</td><td>${escape(r.data.decision?.performance)}</td><td>${escape(r.error || r.data.decision?.reasons?.join("; ") || "有效")}<small>${escape(r.path)}</small></td></tr>`).join("") || '<tr><td colspan="5">暂无实验记录</td></tr>'}</tbody></table></div></article><details><summary>候选与 best 版本</summary><dl><dt>最近评测候选</dt><dd><code>${escape(t.runs[0]?.path.split("/").pop() || "—")}</code></dd><dt>已验证 best</dt><dd><code>${escape(t.best?.run.split("/").pop() || "—")}</code></dd></dl><pre>${escape(JSON.stringify({ candidate: t.runs[0]?.data.source ?? null, best: t.best?.source ?? null }, null, 2))}</pre></details><details><summary>研究笔记 notes.md</summary><pre>${escape(t.notes || "尚未记录")}</pre></details><details><summary>运行记录与可追溯事件</summary>${t.events.map((e) => `<div class="event"><time>${stamp(e.at)}</time><div>${escape(e.kind)}</div><div>${escape(JSON.stringify(e.data))}</div></div>`).join("")}</details>`;
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
async function refresh() {
  if (busy) return;
  busy = true;
  try {
    state = await request("/api/state");
    $("error").hidden = true;
    render();
    if (selected && !$("detail").contains(document.activeElement))
      await renderDetail();
  } catch (e) {
    error(e);
    $("updated").textContent = "连接中断 · 显示上次数据";
  } finally {
    busy = false;
  }
}
document.addEventListener("click", async (event) => {
  const b = event.target.closest("button");
  if (!b) return;
  try {
    if (b.dataset.view) switchView(b.dataset.view);
    else if (b.id === "refresh") await refresh();
    else if (b.dataset.task) {
      selected = b.dataset.task;
      $("overview").hidden = true;
      $("resources").hidden = true;
      $("detail").hidden = false;
      await renderDetail();
    } else if (b.id === "close-detail") switchView(view);
    else if (b.dataset.action) {
      await request("/api/control", {
        task: selected,
        action: b.dataset.action,
      });
      await refresh();
      await renderDetail();
    } else if (b.dataset.clear) {
      await request("/api/resource", { id: b.dataset.clear, clear: true });
      await refresh();
    }
  } catch (e) {
    error(e);
  }
});
document.addEventListener("change", async (event) => {
  if (event.target.dataset.sessions) {
    try {
      await request("/api/resource", {
        id: event.target.dataset.sessions,
        sessions: Number(event.target.value),
      });
      await refresh();
    } catch (e) {
      error(e);
    }
  }
});
document.addEventListener("submit", async (event) => {
  if (event.target.id !== "configure") return;
  event.preventDefault();
  const data = new FormData(event.target);
  try {
    await request("/api/control", {
      task: selected,
      action: "configure",
      changes: {
        resource: data.get("resource"),
        wall_budget_seconds: data.get("wall")
          ? Number(data.get("wall")) * 3600
          : null,
        gpu_budget_seconds: data.get("gpu")
          ? Number(data.get("gpu")) * 3600
          : null,
      },
    });
    await refresh();
    await renderDetail();
  } catch (e) {
    error(e);
  }
});
$("search").addEventListener("input", () => state && render());
refresh();
setInterval(refresh, 3000);
