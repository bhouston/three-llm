const { PR_BASE, PR_HEAD, PR_HEAD_REPO, GITHUB_REPOSITORY, PR_BODY = '' } = process.env;
if (PR_BASE === 'main') {
  if (PR_HEAD !== 'dev' || PR_HEAD_REPO !== GITHUB_REPOSITORY) {
    throw new Error('Release PRs to main must come from this repository’s dev branch.');
  }
} else if (PR_BASE === 'dev') {
  const match = /^(?:feat|fix|docs|chore|refactor|test|style|perf|build|ci|revert)\/(\d+)-[a-z0-9-]+$/.exec(
    PR_HEAD ?? '',
  );
  if (!match) throw new Error('Use a branch such as feat/42-batch-export.');
  const closes = new RegExp(`\\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\s+#${match[1]}\\b`, 'i');
  if (!closes.test(PR_BODY)) throw new Error(`PR body must include Closes #${match[1]}.`);
} else {
  throw new Error('Feature PRs target dev; release PRs target main.');
}
