import { existsSync, readFileSync } from 'fs'
import { resolve } from 'path'
import type { WikipediaProviderConfig } from './pretrain/providers/wikipedia'

// ---- Types ----

export interface PretrainProvidersConfig {
  wikipedia?: Partial<WikipediaProviderConfig>
  // provider baru cukup tambah optional key di sini
}

export interface TinyLLMConfig {
  pretrain?: {
    providers?: PretrainProvidersConfig
  }
}

// ---- Loader ----

const CONFIG_PATH = resolve(process.cwd(), 'data/config.json')

export function loadConfig(): TinyLLMConfig {
  if (!existsSync(CONFIG_PATH)) {
    console.warn(`[config] data/config.json not found, using defaults.`)
    return {}
  }

  const raw = readFileSync(CONFIG_PATH, 'utf-8')
  return JSON.parse(raw) as TinyLLMConfig
}
