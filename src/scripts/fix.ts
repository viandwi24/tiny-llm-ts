import { readFileSync, writeFileSync } from 'fs'
import { resolve } from 'path'



// This script applies a patch to the wikipedia package to add a 'User-Agent' header to API requests
// https://github.com/dopecodez/Wikipedia/pull/79/files/5202a78dc3f38c21189da4ee7e23074fa9192ebd#diff-922206e08f40680cda1319f0ca4adc020479d6fd48d1731121b934064c0b1aaf
const filePath = resolve(import.meta.dir, '../../node_modules/wikipedia/dist/request.js')
const original = `            "Api-User-Agent": USER_AGENT,`
const patched = `            "Api-User-Agent": USER_AGENT,\n            'User-Agent': USER_AGENT,`

const content = readFileSync(filePath, 'utf-8')

if (content.includes("'User-Agent': USER_AGENT,")) {
  console.log('[fix] already patched, skipping.')
  process.exit(0)
}

if (!content.includes(original)) {
  console.error('[fix] patch target not found, file may have changed.')
  process.exit(1)
}

writeFileSync(filePath, content.replace(original, patched), 'utf-8')
console.log('[fix] patched wikipedia/dist/request.js — added User-Agent header.')
