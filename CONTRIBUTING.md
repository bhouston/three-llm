# Contributing

This is the shared workflow for humans, Claude, Codex, and other agents. Follow it for every feature, fix, and maintenance change.

## Issue → branch → pull request

1. Before implementation, open a GitHub issue (or use an existing issue) with a description, motivation, constraints, and acceptance criteria. Use the change-request template. Agents may use `gh issue create --body-file` with these same sections.
2. Fetch `origin` and branch from `origin/main`. Branch names are not enforced; pick anything descriptive. Never commit directly to `main`.
3. Implement the acceptance criteria and run the relevant checks below. Keep changes scoped to the issue.
4. Every commit must follow Conventional Commits: `<type>(<optional-scope>): <description>`. Allowed types are `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `perf`, `build`, `ci`, and `revert`. Reference the issue in the commit body when useful. Husky runs commitlint at commit time; do not bypass hooks.
5. Push the branch and open a PR **against `main`**. Use a Conventional Commit PR title and include `Closes #<issue>` in the body. Describe the resulting behavior and verification. GitHub closes linked issues when merged into the default branch, which is `main`.
6. Wait for required checks and review. Squash feature PRs using the validated PR title; preserve any `BREAKING CHANGE:` footer in the squash message. Do not merge without maintainer authorization. Merging never publishes; releases are a separate, manually dispatched step (below).

`feat` causes a minor release; `fix` and `perf` cause a patch release. `feat!:` or a `BREAKING CHANGE:` footer causes a major release. Other types do not release by themselves. These rules also apply before 1.0.0. A plain merge commit is ignored by commitlint and the release analyzer.

## Development and quality gates

Use the Node version in `.nvmrc` and the pnpm version in `package.json`.

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm lint
pnpm test --coverage
pnpm size
pnpm audit --audit-level high
pnpm exec playwright install chromium
pnpm test:browser
pnpm test:e2e
```

`pnpm test` includes TypeScript checking. Coverage measures all library source files, including untested files, excluding tests, declarations, test helpers, and barrel exports. CPU unit coverage floors are 59% lines, 58% statements, 58% branches, and 52% functions, based on the initial measured baseline. Raise these as coverage improves; do not lower them to pass a change. GPU browser and checkpoint coverage are not part of this CPU gate.

Size Limit measures the total Brotli-compressed compiled JavaScript, excluding external dependencies, with a 70 kB budget (initial baseline 57.92 kB). It is a distribution-size gate, not an application tree-shaking benchmark. CI logs the size as part of the Unit check. High and critical audit findings fail CI; moderate findings remain visible. Checkpoint tests remain advisory because they need external model assets.

CI uploads coverage artifacts and sends LCOV to Codecov for the README percentage badge. Enable the public repository in Codecov; if required, set `CODECOV_TOKEN`. Coverage enforcement does not depend on Codecov availability.

## Controlled releases

Merging a feature PR into `main` never publishes; it only runs CI. When ready to release, a maintainer manually dispatches the release workflow on `main`:

```sh
gh workflow run release.yml --ref main
```

The workflow rejects any dispatch not targeting `refs/heads/main`, then runs the full quality-check suite before releasing. Semantic-release computes the version, generates release notes and a changelog, updates the package version in the CI workspace, publishes `three-llm` with npm OIDC, tags the release, and attaches the npm tarball and `CHANGELOG.md` to the GitHub release. The changelog is also included in the npm package. Generated version/changelog changes are intentionally not committed back: tags and GitHub releases are the release record, avoiding protected-branch bot commits and synchronization merges. The checked-in package version is not the published version authority. If there are no release-worthy commits since the last release, the run is a clean no-op.

Pass `-f dry_run=true` (or use the Actions "Run workflow" UI) to run semantic-release in dry-run mode against `main` without publishing — useful for verifying changelog output without releasing.

There are no `NPM_TOKEN` or `NODE_AUTH_TOKEN` secrets. Keep `id-token: write` on the publishing job and use GitHub-hosted runners. Do not run a local publishing command. `pnpm make-release` performs a semantic-release dry run; full OIDC validation and publication happen only in Actions.

### Initial maintainer setup

In npm → `three-llm` → Settings → Trusted publishing, add GitHub Actions:

- Organization/user: `bhouston`
- Repository: `three-llm`
- Workflow filename: `release.yml` (not the path)
- Environment name: leave empty (the workflow does not use a GitHub environment)
- Allowed action: publish

See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/). Configure this before merging the first release PR to `main`.

The initial `v0.5.0` tag must point to npm's published `gitHead`, `d6274946f26ab601d5d728fe7041da79a681c8fc`. Without this baseline, semantic-release would treat this as a first release. Existing nonconventional history is retained; new commits are enforced from PR base to head.

Use `main` as GitHub's default branch. Protect it with PRs, required CI checks, no force pushes, and no deletion. Required checks are Unit, Browser Unit, E2E, and Contribution policy. Checkpoints are advisory. Squash feature PRs. GitHub rules must require the checks for failures to actually block merges. The release workflow is manually dispatched from `main` and is available in Actions only once merged into the default branch.

## Reuse in other repositories

After this pilot passes remote CI and a real release, extract `CONTRIBUTING.md`, agent pointers, templates, commitlint/Husky config, release config, and workflows into a separate GitHub template repository. Keep repo-specific package paths, coverage baselines, release tags, security contact, and size budgets explicit. Do not copy this package's baseline tag or publish identity. A template repository is a separate rollout step, not required to release this package.
