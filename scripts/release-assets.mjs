import { copyFile } from 'node:fs/promises';

export async function prepare() {
  await Promise.all([
    copyFile('LICENSE', 'packages/three-llm/LICENSE'),
    copyFile('README.md', 'packages/three-llm/README.md'),
  ]);
}
