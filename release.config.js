export default {
  branches: ['main'],
  tagFormat: 'v${version}',
  plugins: [
    ['@semantic-release/commit-analyzer', { preset: 'conventionalcommits' }],
    ['@semantic-release/release-notes-generator', { preset: 'conventionalcommits' }],
    ['@semantic-release/changelog', { changelogFile: 'packages/three-llm/CHANGELOG.md' }],
    ['@anolilab/semantic-release-pnpm', { pkgRoot: 'packages/three-llm', tarballDir: 'release-artifacts' }],
    [
      '@semantic-release/github',
      {
        successComment: false,
        failComment: false,
        assets: ['release-artifacts/*.tgz', 'packages/three-llm/CHANGELOG.md'],
      },
    ],
  ],
};
