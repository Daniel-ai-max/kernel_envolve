import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
const source = await readFile(
  new URL("../src/kernel_envolve/manager/static/sync.js", import.meta.url),
  "utf8",
);
const { poller } = await import(
  "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);
const flush = () => new Promise((resolve) => setImmediate(resolve));
function fixture() {
  let visible = true;
  const reads = [],
    applied = [],
    errors = [],
    timers = new Map();
  let id = 0;
  const sync = poller({
    read: (signal) =>
      new Promise((resolve, reject) => reads.push({ signal, resolve, reject })),
    apply: (data) => applied.push(data),
    failed: (error) => errors.push(error),
    visible: () => visible,
    schedule: (fn, delay) => {
      timers.set(++id, { fn, delay });
      return id;
    },
    cancel: (id) => timers.delete(id),
  });
  function fire() {
    assert.equal(timers.size, 1);
    const [id, timer] = [...timers][0];
    timers.delete(id);
    timer.fn();
    return timer.delay;
  }
  return {
    sync,
    reads,
    applied,
    errors,
    timers,
    fire,
    hide: () => {
      visible = false;
      sync.stop();
    },
    show: () => {
      visible = true;
      return sync.refresh();
    },
  };
}
test("only a settled request schedules another read; navigation rejects old responses", async () => {
  const f = fixture();
  const old = f.sync.refresh();
  assert.equal(f.reads.length, 1);
  assert.equal(f.timers.size, 0);
  const latest = f.sync.refresh();
  assert.equal(f.reads[0].signal.aborted, true);
  f.reads[1].resolve("operator B");
  await latest;
  f.reads[0].resolve("operator A");
  await old;
  assert.deepEqual(f.applied, ["operator B"]);
  assert.equal(f.fire(), 3000);
  assert.equal(f.reads.length, 3);
  assert.equal(f.timers.size, 0);
  f.sync.stop();
  f.reads[2].resolve("late");
  await flush();
  assert.deepEqual(f.applied, ["operator B"]);
});
test("errors back off to 30 seconds and success restores normal polling", async () => {
  const f = fixture();
  f.sync.refresh();
  for (const delay of [6000, 12000, 24000, 30000, 30000]) {
    f.reads.at(-1).reject(new Error("offline"));
    await flush();
    assert.equal(f.fire(), delay);
  }
  f.reads.at(-1).resolve("reconnected");
  await flush();
  assert.equal(f.fire(), 3000);
  assert.deepEqual(f.applied, ["reconnected"]);
  f.sync.stop();
});
test("hidden pages cancel work; returning and manual refresh bypass backoff", async () => {
  const f = fixture();
  const first = f.sync.refresh();
  f.hide();
  assert.equal(f.reads[0].signal.aborted, true);
  f.reads[0].reject(new Error("abort"));
  await first;
  assert.equal(f.errors.length, 0);
  assert.equal(f.timers.size, 0);
  await f.sync.refresh();
  assert.equal(f.reads.length, 1);
  const next = f.show();
  assert.equal(f.reads.length, 2);
  f.reads[1].reject(new Error("offline"));
  await next;
  assert.equal(f.timers.size, 1);
  const manual = f.sync.refresh();
  assert.equal(f.timers.size, 0);
  f.reads[2].resolve("fresh");
  await manual;
  assert.equal(f.fire(), 3000);
  f.sync.stop();
});
