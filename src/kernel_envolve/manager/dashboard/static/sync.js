/* Poll one snapshot at a time. A navigation/control invalidates every older response. */
export function poller({
  read,
  apply,
  failed,
  visible,
  schedule = setTimeout,
  cancel = clearTimeout,
}) {
  let generation = 0,
    timer,
    controller,
    failures = 0;
  function stop() {
    generation++;
    cancel(timer);
    controller?.abort();
  }
  async function tick(epoch) {
    if (!visible() || epoch !== generation) return;
    controller = new AbortController();
    try {
      const data = await read(controller.signal);
      if (epoch !== generation) return;
      apply(data);
      failures = 0;
    } catch (error) {
      if (epoch !== generation) return;
      failures++;
      failed(error);
    } finally {
      if (epoch === generation && visible())
        timer = schedule(
          () => tick(epoch),
          Math.min(30000, 3000 * 2 ** Math.min(failures, 4)),
        );
    }
  }
  function refresh() {
    stop();
    failures = 0;
    return tick(generation);
  }
  return { refresh, stop };
}

function key(node) {
  if (node.nodeType !== 1) return null;
  return (
    node.id ||
    node.dataset.key ||
    (node.tagName === "ARTICLE"
      ? node.querySelector("h3")?.textContent
      : null) ||
    (node.tagName === "DETAILS"
      ? node.querySelector("summary")?.textContent
      : null)
  );
}

/* Reuse DOM nodes: focus, scroll, text selection and expanded details survive polling. */
export function patch(parent, html, dirty = false) {
  const template = document.createElement("template");
  template.innerHTML = html;
  function node(current, fresh) {
    if (
      current.nodeType !== fresh.nodeType ||
      current.nodeName !== fresh.nodeName
    ) {
      const replacement = fresh.cloneNode(true);
      current.replaceWith(replacement);
      return replacement;
    }
    if (current.nodeType !== 1) {
      if (current.nodeValue !== fresh.nodeValue)
        current.nodeValue = fresh.nodeValue;
      return current;
    }
    const field = current.matches("input,select,textarea");
    const keepValue =
      field &&
      (current === document.activeElement ||
        (dirty && current.closest("#configure")));
    const keep = (name) =>
      (name === "open" && current.tagName === "DETAILS") ||
      (name === "value" && keepValue);
    for (const attr of [...current.attributes])
      if (!keep(attr.name) && !fresh.hasAttribute(attr.name))
        current.removeAttribute(attr.name);
    for (const attr of fresh.attributes)
      if (!keep(attr.name) && current.getAttribute(attr.name) !== attr.value)
        current.setAttribute(attr.name, attr.value);
    if (!keepValue) {
      children(current, fresh);
      if (field && current.value !== fresh.value) current.value = fresh.value;
    }
    return current;
  }
  function children(current, fresh) {
    const old = [...current.childNodes],
      used = new Set();
    const keyed = new Map(old.filter((n) => key(n)).map((n) => [key(n), n]));
    [...fresh.childNodes].forEach((next, i) => {
      let match = key(next) ? keyed.get(key(next)) : old[i];
      if (!key(next) && key(match || {})) match = null;
      if (!match || used.has(match)) {
        match = next.cloneNode(true);
        current.insertBefore(match, current.childNodes[i] || null);
      } else {
        if (current.childNodes[i] !== match)
          current.insertBefore(match, current.childNodes[i] || null);
        match = node(match, next);
      }
      used.add(match);
    });
    for (const child of [...current.childNodes])
      if (!used.has(child)) child.remove();
  }
  children(parent, template.content);
}
