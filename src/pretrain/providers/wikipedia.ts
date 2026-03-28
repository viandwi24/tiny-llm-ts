import type { Command } from 'commander'
import wiki from 'wikipedia'
import { mkdirSync, writeFileSync } from 'fs'
import { resolve, join } from 'path'
import React from 'react'
import type { PretrainProvider } from '../provider'
import type { WikipediaScraperCallbacks } from '../../tui/components/WikipediaScraper'
import { WikipediaScraper } from '../../tui/components/WikipediaScraper'
import { renderTUI } from '../../tui'

export interface WikipediaProviderConfig {
  topics: string[]
  outputDir: string
  language: string
  searchLimit: number
}

const STOP_SECTIONS = [
  'lihat pula', 'referensi', 'bacaan lanjutan', 'pranala luar',
  'catatan kaki', 'daftar pustaka', 'notes', 'references',
  'see also', 'external links', 'further reading',
]

function cleanText(text: string): string {
  const lines = text.split('\n')
  const result: string[] = []
  let inStopSection = false

  for (let line of lines) {
    const headingMatch = line.match(/^={2,4}\s*(.+?)\s*={2,4}$/)
    if (headingMatch) {
      const title = headingMatch[1]!.toLowerCase()
      inStopSection = STOP_SECTIONS.some(s => title.includes(s))
    }
    if (inStopSection) continue

    // Hapus heading markup, sisakan teks
    line = line.replace(/^={2,4}\s*(.+?)\s*={2,4}$/, '$1')
    // Hapus artifact kode bahasa Wikipedia
    line = line.replace(/code:\s*\w+\s+is deprecated/g, '')
    // Hapus karakter Yunani, Arab, Kirillik yang terselip
    line = line.replace(/[\u0370-\u03FF\u0600-\u06FF\u0400-\u04FF]/g, '')
    // Hapus baris ISBN / OCLC
    if (/ISBN\s[\d\-X]+/.test(line) || /OCLC\s[\d\s]+/.test(line)) continue
    // Hapus baris Wayback Machine
    if (/Diarsipkan.+Wayback Machine/.test(line)) continue
    // Hapus baris sangat pendek tanpa kata
    const stripped = line.trim()
    if (stripped.length > 0 && stripped.length < 4 && !/\w/.test(stripped)) continue
    // Bersihkan whitespace
    line = line.replace(/\s{2,}/g, ' ').trimEnd()

    result.push(line)
  }

  // Collapse baris kosong berlebih
  const collapsed: string[] = []
  let prevEmpty = false
  for (const line of result) {
    const isEmpty = line.trim() === ''
    if (isEmpty && prevEmpty) continue
    collapsed.push(line)
    prevEmpty = isEmpty
  }

  while (collapsed[0]?.trim() === '') collapsed.shift()
  while (collapsed[collapsed.length - 1]?.trim() === '') collapsed.pop()

  return collapsed.join('\n')
}

export async function runWikipediaFetch(
  config: WikipediaProviderConfig,
  cb: WikipediaScraperCallbacks
) {
  const { topics, outputDir, language: lang, searchLimit } = config

  wiki.setLang(lang)
  wiki.setUserAgent('tiny-llm/1.0')
  mkdirSync(outputDir, { recursive: true })

  let totalWritten = 0

  for (let i = 0; i < topics.length; i++) {
    const topic = topics[i]!
    cb.onTopicStart(topic, i, topics.length)

    let results: string[]
    try {
      const searchRes = await wiki.search(topic, { limit: searchLimit })
      results = searchRes.results.map((r: any) => r.title ?? r)
    } catch (err) {
      cb.onFileSkipped(topic, (err as Error).message)
      continue
    }

    cb.onSearchDone(topic, results.length)

    const topicSlug = topic.toLowerCase().replace(/[^a-z0-9]+/gi, '_')

    for (const title of results) {
      try {
        const raw = await wiki.content(title)
        if (!raw) {
          cb.onFileSkipped(title, 'empty content')
          continue
        }
        const content = cleanText(raw)
        if (!content) {
          cb.onFileSkipped(title, 'empty after cleaning')
          continue
        }
        const titleSlug = title.toLowerCase().replace(/[^a-z0-9]+/gi, '_')
        const filePath = join(outputDir, `${topicSlug}-${titleSlug}.txt`)
        writeFileSync(filePath, content, 'utf-8')
        totalWritten++
        cb.onFileWritten(title, filePath, topic)
      } catch (err) {
        cb.onFileSkipped(title, (err as Error).message)
      }
    }

    cb.onTopicDone(topic)
  }

  cb.onDone(totalWritten)
}

export const wikipediaProvider: PretrainProvider<WikipediaProviderConfig> = {
  name: 'wikipedia',
  description: 'Fetch articles from Wikipedia as pre-training data',

  defaultConfig: {
    topics: [
      'Indonesia', 'Sejarah Indonesia', 'Geografi',
      'Ilmu pengetahuan', 'Matematika', 'Fisika',
      'Biologi', 'Kimia', 'Teknologi', 'Seni',
    ],
    outputDir: 'data/pretrain/raw',
    language: 'id',
    searchLimit: 100,
  },

  registerCommand(cmd: Command, config: WikipediaProviderConfig) {
    cmd
      .option('--output <path>', 'Output directory', config.outputDir)
      .option('--lang <lang>', 'Wikipedia language code', config.language)
      .option('--limit <number>', 'Search results limit per topic', String(config.searchLimit))
  },

  async fetch(opts, config) {
    const resolvedConfig: WikipediaProviderConfig = {
      ...config,
      outputDir: resolve(process.cwd(), opts.output ?? config.outputDir),
      language: opts.lang ?? config.language,
      searchLimit: Number(opts.limit ?? config.searchLimit),
    }

    const waitUntilExit = renderTUI(
      React.createElement(WikipediaScraper, {
        ...resolvedConfig,
        onRun: (cb) => runWikipediaFetch(resolvedConfig, cb),
      })
    )

    await waitUntilExit()
  },
}
