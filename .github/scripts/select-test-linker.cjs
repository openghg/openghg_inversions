module.exports = async ({ github, context, core }) => {
  const select = (linker, reason) => {
    core.info(`Test linker: ${linker} (${reason})`);
    return linker;
  };
  if (context.eventName === "workflow_dispatch") {
    const inputs = context.payload?.inputs || {};
    const disabled = (value) => value === false || value === "false";
    return disabled(inputs.compiled_backend) && disabled(inputs.runtime_comparison)
      ? select("py", "manual run without compiled coverage or runtime comparison")
      : select("auto", "manual compiled coverage, runtime comparison, or missing inputs");
  }
  if (context.eventName !== "pull_request") return select("auto", "non-PR event");

  const pr = context.payload?.pull_request;
  const nonempty = (value) => typeof value === "string" && value.trim().length > 0;
  if (!pr || !Number.isInteger(pr.number) || pr.number < 1 ||
      !nonempty(pr.head?.sha) || !nonempty(pr.base?.sha) ||
      !nonempty(context.repo?.owner) || !nonempty(context.repo?.repo) ||
      !Number.isInteger(pr.changed_files) || pr.changed_files < 1 || pr.changed_files > 3000) {
    return select("auto", "missing PR metadata or file count outside API coverage");
  }
  try {
    const params = { ...context.repo, pull_number: pr.number };
    const files = await github.paginate(github.rest.pulls.listFiles, { ...params, per_page: 100 });
    if (!Array.isArray(files) || files.length !== pr.changed_files) {
      return select("auto", "PR file list is incomplete");
    }
    const paths = [];
    for (const file of files) {
      if (!nonempty(file?.filename) ||
          (file.status === "renamed" && !nonempty(file.previous_filename))) {
        return select("auto", "missing current or renamed file path");
      }
      paths.push(file.filename);
      if (file.status === "renamed") paths.push(file.previous_filename);
    }
    const { data: current } = await github.rest.pulls.get(params);
    if (current?.head?.sha !== pr.head.sha || current?.base?.sha !== pr.base.sha) {
      return select("auto", "PR head or base changed since this event");
    }
    const prefixes = [
      "openghg_inversions/", "examples/", "tests/models/", "tests/experimental/",
      "tests/integration/", ".github/workflows/", ".github/scripts/",
    ];
    const exact = new Set([
      "pyproject.toml", "uv.lock", "tox.ini", "pixi.toml", "pixi.lock",
      "tests/conftest.py", "tests/helpers.py",
      "tests/standard_model_helpers.py", "tests/multisector_model_helpers.py",
      "tests/postprocessing/test_linked_paris_outputs.py",
    ]);
    const scientificTest = /^tests\/[^/]*(rhime|co2|sigma|sampl|pymc|nested|handoff|linked|hbmcmc|covariance|likelihood|boundary)[^/]*\.py$/;
    const protectedPath = paths.find((path) => exact.has(path) ||
      prefixes.some((prefix) => path.startsWith(prefix)) || scientificTest.test(path));
    return protectedPath
      ? select("auto", `compiled coverage required by ${protectedPath}`)
      : select("py", "complete PR file list contains no compiled-coverage paths");
  } catch (error) {
    core.warning(`Unable to verify PR files: ${error?.message || error}`);
    return select("auto", "PR API verification failed");
  }
};
