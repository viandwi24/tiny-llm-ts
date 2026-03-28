import { readFileSync, writeFileSync } from 'fs'
import { resolve } from 'path'

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
