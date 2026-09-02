#!/usr/bin/env node

import { execSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function getWorkspaceVersions(rootPath) {
  const map = new Map();
  const workspaceDirs = ['packages', 'examples'];

  for (const dir of workspaceDirs) {
    const fullDir = join(rootPath, dir);
    if (!existsSync(fullDir) || !statSync(fullDir).isDirectory()) continue;

    for (const name of readdirSync(fullDir)) {
      const pkgDir = join(fullDir, name);
      const pkgPath = join(pkgDir, 'package.json');
      if (!statSync(pkgDir).isDirectory() || !existsSync(pkgPath)) continue;

      const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'));
      if (pkg.name && pkg.version) map.set(pkg.name, pkg.version);
    }
  }

  return map;
}

function resolveWorkspaceDeps(packageJson, workspaceVersions) {
  const depKeys = ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies'];

  for (const key of depKeys) {
    const deps = packageJson[key];
    if (!deps || typeof deps !== 'object') continue;

    for (const [pkgName, spec] of Object.entries(deps)) {
      if (typeof spec !== 'string' || !spec.startsWith('workspace:')) continue;

      const version = workspaceVersions.get(pkgName);
      if (!version) {
        throw new Error(
          `Package "${pkgName}" is referenced as workspace:* but no package named "${pkgName}" was found in the monorepo.`,
        );
      }
      deps[pkgName] = `^${version}`;
    }
  }
}

function main() {
  const packagePath = process.argv[2];
  if (!packagePath) {
    throw new Error('Error: Package path is required');
  }

  const resolvedPackagePath = resolve(packagePath);
  const publishPath = join(resolvedPackagePath, 'publish');
  const rootPath = resolve(__dirname, '..');

  if (!existsSync(resolvedPackagePath)) {
    throw new Error(`Error: Package directory does not exist: ${resolvedPackagePath}`);
  }

  const packageJsonPath = join(resolvedPackagePath, 'package.json');
  if (!existsSync(packageJsonPath)) {
    throw new Error(`Error: package.json not found in: ${resolvedPackagePath}`);
  }

  console.log('Cleaning publish dir');
  if (existsSync(publishPath)) {
    rmSync(publishPath, { recursive: true, force: true });
  }
  mkdirSync(publishPath, { recursive: true });

  console.log('Building package');
  execSync('pnpm -s build', { cwd: resolvedPackagePath, stdio: 'inherit' });

  console.log('Copying files to publish directory...');

  const distPath = join(resolvedPackagePath, 'dist');
  if (!existsSync(distPath)) {
    throw new Error(`Error: dist directory not found at ${distPath}`);
  }
  cpSync(distPath, join(publishPath, 'dist'), { recursive: true });

  console.log('Copying package.json (removing files field)');
  const packageJsonContent = JSON.parse(readFileSync(packageJsonPath, 'utf-8'));
  const { files: _files, ...packageJsonWithoutFiles } = packageJsonContent;

  const workspaceVersions = getWorkspaceVersions(rootPath);
  if (workspaceVersions.size > 0) {
    console.log(`Resolving workspace dependencies from monorepo (${workspaceVersions.size} package(s))`);
    resolveWorkspaceDeps(packageJsonWithoutFiles, workspaceVersions);
  }

  const publishPackageJsonPath = join(publishPath, 'package.json');
  writeFileSync(publishPackageJsonPath, `${JSON.stringify(packageJsonWithoutFiles, null, 2)}\n`);

  const npmignorePath = join(resolvedPackagePath, '.npmignore');
  if (existsSync(npmignorePath)) {
    console.log('Copying .npmignore');
    cpSync(npmignorePath, join(publishPath, '.npmignore'));
  }

  const licensePath = join(rootPath, 'LICENSE');
  if (existsSync(licensePath)) {
    console.log('Copying LICENSE from root');
    cpSync(licensePath, join(publishPath, 'LICENSE'));
  } else {
    throw new Error('Error: LICENSE not found at repo root');
  }

  const packageReadmePath = join(resolvedPackagePath, 'README.md');
  const rootReadmePath = join(rootPath, 'README.md');
  const readmePath = existsSync(packageReadmePath) ? packageReadmePath : rootReadmePath;
  if (!existsSync(readmePath)) {
    throw new Error('Error: README.md not found in package or root');
  }
  console.log(`Copying README from ${existsSync(packageReadmePath) ? 'package' : 'root'}`);
  cpSync(readmePath, join(publishPath, 'README.md'));

  console.log('Publishing package');
  execSync('npm publish ./publish/ --access public', {
    cwd: resolvedPackagePath,
    stdio: 'inherit',
  });

  console.log('Release completed successfully!');
}

try {
  main();
} catch (error) {
  console.error(`Error: Release failed: ${error}`);
  process.exit(1);
}
