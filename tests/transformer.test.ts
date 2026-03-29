import { test, expect } from 'bun:test'
import { Transformer } from '../src/transformer/index'

test('Transformer init and forward', () => {
  const model = new Transformer({ vocabSize: 50, embedSize: 8, numHeads: 2, numLayers: 2, ffnDim: 16, maxSeqLen: 16 })
  const out = model.forward([0, 1, 2, 3])
  expect(out.length).toBe(4)
  expect(out[0]!.length).toBe(50)
})

test('transformer', () => {
  const model = new Transformer({
    vocabSize: 100,
    embedSize: 16,
    numHeads: 2,
    numLayers: 2,
    ffnDim: 32,
    maxSeqLen: 10
  })

  // input dummy: 5 token IDs
  const tokenIds = [1, 5, 23, 47, 80]

  console.log('[test] forward pass...')
  const logits = model.forward(tokenIds)
  if (logits.length === 0 || !logits[0]) throw new Error('Output logits kosong, ada yang salah di forward pass')

  // validasi shape
  console.log(`[test] input length : ${tokenIds.length}`)
  console.log(`[test] output shape : [${logits.length}, ${logits[0].length}]`)
  console.log(`[test] expected     : [${tokenIds.length}, 100]`)

  // validasi tidak ada NaN
  const hasNaN = logits.some(row => row.some(v => isNaN(v)))
  const hasInf = logits.some(row => row.some(v => !isFinite(v)))
  console.log(`[test] has NaN      : ${hasNaN}`)
  console.log(`[test] has Inf      : ${hasInf}`)

  // cek output wajar
  console.log(`[test] logits sample (token 0, pertama 5 nilai):`)
  console.log(logits[0].slice(0, 5))

  if (!hasNaN && !hasInf && logits.length === tokenIds.length && logits[0].length === 100) {
    console.log('[test] ✅ semua OK!')
  } else {
    console.log('[test] ❌ ada yang salah, cek implementasi')
  }

  expect(hasNaN).toBe(false)
  expect(hasInf).toBe(false)
  expect(logits.length).toBe(tokenIds.length)
  expect(logits[0]!.length).toBe(100)
})