const assert = require("node:assert/strict");
const test = require("node:test");
const selectTestLinker = require("./select-test-linker.cjs");

function harness(files = [{ filename: "docs/usage.rst", status: "modified" }]) {
  const context = {
    eventName: "pull_request",
    repo: { owner: "example", repo: "inversions" },
    payload: { pull_request: { number: 17, changed_files: files.length,
      head: { sha: "event-head" }, base: { sha: "event-base" } } },
  };
  const calls = [];
  const messages = [];
  const warnings = [];
  const github = {
    rest: { pulls: {
      listFiles() {},
      async get(params) {
        calls.push("get");
        assert.deepEqual(params, { owner: "example", repo: "inversions", pull_number: 17 });
        return { data: { head: { sha: "event-head" }, base: { sha: "event-base" } } };
      },
    } },
    async paginate(endpoint, params) {
      calls.push("list");
      assert.equal(endpoint, github.rest.pulls.listFiles);
      assert.deepEqual(params, {
        owner: "example", repo: "inversions", pull_number: 17, per_page: 100,
      });
      return files;
    },
  };
  return { github, context, calls, messages, warnings,
    core: { info: (message) => messages.push(message), warning: (message) => warnings.push(message) } };
}

test("non-PR events retain compiled coverage and manual inputs accept booleans or strings", { concurrency: true }, async () => {
  for (const eventName of ["push", "schedule", "pull_request_target"]) {
    const h = harness();
    h.context.eventName = eventName;
    assert.equal(await selectTestLinker(h), "auto", eventName);
    assert.deepEqual(h.calls, []);
  }
  for (const [compiled_backend, runtime_comparison, expected] of [
    [false, false, "py"], ["false", "false", "py"], [false, "false", "py"],
    [true, false, "auto"], ["true", "false", "auto"],
    [false, true, "auto"], ["false", "true", "auto"],
    [undefined, undefined, "auto"], ["unknown", false, "auto"],
  ]) {
    const h = harness();
    h.context.eventName = "workflow_dispatch";
    h.context.payload.inputs = { compiled_backend, runtime_comparison };
    assert.equal(await selectTestLinker(h), expected);
    assert.deepEqual(h.calls, []);
  }
});

test("protected paths select compiled coverage while ordinary changes use Python", { concurrency: true }, async () => {
  const protectedPaths = [
    "openghg_inversions/model.py", "examples/run.ini", "tests/models/test_density.py",
    "tests/experimental/test_model.py", "tests/integration/test_api.py",
    ".github/workflows/workflow.yaml", ".github/scripts/selector.cjs",
    "pyproject.toml", "uv.lock", "tox.ini", "pixi.toml", "pixi.lock",
    "tests/conftest.py", "tests/helpers.py",
    "tests/standard_model_helpers.py", "tests/multisector_model_helpers.py",
    "tests/postprocessing/test_linked_paris_outputs.py",
    ...["rhime", "co2", "sigma", "sampl", "pymc", "nested", "handoff", "linked",
      "hbmcmc", "covariance", "likelihood", "boundary"].map((name) => `tests/test_${name}_rule.py`),
  ];
  for (const filename of protectedPaths) {
    const h = harness([{ filename, status: "modified" }]);
    assert.equal(await selectTestLinker(h), "auto", filename);
    assert.deepEqual(h.calls, ["list", "get"]);
    assert.match(h.messages[0], /compiled coverage required/);
  }
  for (const filename of ["README.md", "docs/model.rst", "tests/test_array_ops.py",
    "tests/postprocessing/test_countries.py", "openghg_inversions_extra/helper.py"]) {
    assert.equal(await selectTestLinker(harness([{ filename, status: "modified" }])), "py", filename);
  }
  assert.equal(await selectTestLinker(harness([
    { filename: "docs/usage.rst" }, { filename: "uv.lock" },
  ])), "auto");
});

test("missing metadata, API limits, and incomplete file lists fail closed", { concurrency: true }, async () => {
  const invalidMetadata = [
    (h) => { delete h.context.payload; },
    (h) => { delete h.context.payload.pull_request; },
    (h) => { delete h.context.payload.pull_request.number; },
    (h) => { delete h.context.payload.pull_request.head; },
    (h) => { delete h.context.payload.pull_request.base.sha; },
    (h) => { delete h.context.repo.owner; },
    ...[undefined, 0, -1, 1.5, 3001].map((count) =>
      (h) => { h.context.payload.pull_request.changed_files = count; }),
  ];
  for (const invalidate of invalidMetadata) {
    const h = harness();
    invalidate(h);
    assert.equal(await selectTestLinker(h), "auto");
    assert.deepEqual(h.calls, []);
  }
  const incomplete = harness();
  incomplete.context.payload.pull_request.changed_files = 2;
  assert.equal(await selectTestLinker(incomplete), "auto");
  for (const file of [{}, { filename: "" }, { filename: "   " }, null,
    { filename: "docs/new.rst", status: "renamed" }]) {
    assert.equal(await selectTestLinker(harness([file])), "auto");
  }
  const capped = harness(Array.from({ length: 3000 }, (_, i) => ({ filename: `docs/${i}.rst` })));
  assert.equal(await selectTestLinker(capped), "py");
  assert.deepEqual(capped.calls, ["list", "get"]);
});

test("renames check both old and new paths", { concurrency: true }, async () => {
  for (const [filename, previous_filename, expected] of [
    ["docs/model.rst", "openghg_inversions/model.py", "auto"],
    ["tests/test_co2_new.py", "docs/model.rst", "auto"],
    ["docs/new.rst", "docs/old.rst", "py"],
  ]) {
    assert.equal(await selectTestLinker(harness([
      { filename, previous_filename, status: "renamed" },
    ])), expected);
  }
});

test("stale PR events and API failures retain compiled coverage", { concurrency: true }, async () => {
  for (const data of [
    { head: { sha: "new-head" }, base: { sha: "event-base" } },
    { head: { sha: "event-head" }, base: { sha: "new-base" } },
    {},
  ]) {
    const h = harness();
    h.github.rest.pulls.get = async () => ({ data });
    assert.equal(await selectTestLinker(h), "auto");
    assert.deepEqual(h.warnings, []);
  }
  for (const fail of [
    (h) => { h.github.paginate = async () => { throw new Error("list unavailable"); }; },
    (h) => { h.github.rest.pulls.get = async () => { throw new Error("get unavailable"); }; },
    (h) => { h.github.paginate = async () => { throw null; }; },
  ]) {
    const h = harness();
    fail(h);
    assert.equal(await selectTestLinker(h), "auto");
    assert.match(h.warnings[0], /Unable to verify PR files: ((list|get) unavailable|null)/);
    assert.equal(h.messages.length, 1);
  }
});
