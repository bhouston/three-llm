const { PR_BASE, PR_BODY = '' } = process.env;
if (PR_BASE !== 'main') {
  throw new Error('PRs target main.');
}
const closes = /\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#\d+\b/i;
if (!closes.test(PR_BODY)) throw new Error('PR body must include Closes #<issue>.');
