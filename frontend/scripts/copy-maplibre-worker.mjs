import { copyFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const packageJson = require.resolve('maplibre-gl/package.json');
const sourceDirectory = path.join(path.dirname(packageJson), 'dist');
const outputDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../public/maplibre');

mkdirSync(outputDirectory, { recursive: true });
for (const file of ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs']) {
  copyFileSync(path.join(sourceDirectory, file), path.join(outputDirectory, file));
}
