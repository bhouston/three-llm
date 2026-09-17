const { PR_BASE, PR_HEAD, PR_BODY = '' } = process.env;
if (PR_BASE !== 'main') {
  throw new Error('PRs target main.');
}
const match = /^(?:feat|fix|docs|chore|refactor|test|style|perf|build|ci|revert)\/(\d+)-[a-z0-9-]+$/.exec(
  PR_HEAD ?? '',
);
if (!match) throw new Error('Use a branch such as feat/42-batch-export.');
const closes = new RegExp(`\\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\s+#${match[1]}\\b`, 'i');
if (!closes.test(PR_BODY)) throw new Error(`PR body must include Closes #${match[1]}.`);
