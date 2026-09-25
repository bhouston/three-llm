# Releasing

`three-llm` publishes to npm through a manually-dispatched release workflow. See [CONTRIBUTING.md](CONTRIBUTING.md) for the general release mechanics (semantic-release, `gh workflow run release.yml --ref main`, no `NPM_TOKEN`). This document covers the repo-specific setup and gates around that process.

## Quality gates enforced before release

The release workflow reuses `ci.yml`, so a release only proceeds if these all pass:

- **Unit tests / coverage.** CPU unit coverage floors are 59% lines, 58% statements, 58% branches, and 52% functions, based on the initial measured baseline. Raise these as coverage improves; do not lower them to pass a change. Coverage measures all library source files, including untested files, excluding tests, declarations, test helpers, and barrel exports.
- **GPU browser and checkpoint coverage** are not part of the CPU coverage gate above. Checkpoint tests in particular remain advisory (`continue-on-error` in CI) because they need external model assets that aren't available to every contributor or CI run.
- **Bundle size.** `pnpm size` (Size Limit) measures the total Brotli-compressed compiled JavaScript, excluding external dependencies, against a 70 kB budget (initial baseline 57.92 kB). It's a distribution-size gate, not an application tree-shaking benchmark. CI logs the size as part of the Unit check.
- **Dependency audit.** High and critical `pnpm audit` findings fail CI; moderate findings remain visible but don't block.

Required branch-protection checks on `main`: Unit, Browser Unit, E2E, and Contribution policy. Checkpoints are advisory only and are not required.

## Codecov

CI uploads coverage artifacts and sends LCOV to Codecov for the README coverage badge. To keep this working:

- Enable the public `bhouston/three-llm` repository in Codecov.
- If Codecov requires it for this repo, set the `CODECOV_TOKEN` repository secret.

Coverage enforcement in CI does not depend on Codecov being reachable — the coverage floors above are checked locally from the `vitest --coverage` run regardless.

## Initial maintainer setup

One-time setup required before the first release can publish:

### npm trusted publishing

In npm → `three-llm` → Settings → Trusted publishing, add GitHub Actions:

- Organization/user: `bhouston`
- Repository: `three-llm`
- Workflow filename: `release.yml` (not the path)
- Environment name: leave empty (the workflow does not use a GitHub environment)
- Allowed action: publish

See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/). Configure this before dispatching the first release.

### Baseline release tag

The initial `v0.5.0` tag must point to npm's published `gitHead`, `d6274946f26ab601d5d728fe7041da79a681c8fc`. Without this baseline, semantic-release would treat the next dispatch as a first release. Existing nonconventional history is retained; new commits are enforced from PR base to head via commitlint.
