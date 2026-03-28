import type { Command } from 'commander'

export interface PretrainProvider<TConfig = unknown> {
  /** Unique name, dipakai sebagai subcommand: `pretrain fetch <name>` */
  name: string
  description: string

  /** Default config untuk provider ini */
  defaultConfig: TConfig

  /** Register options/arguments ke Commander subcommand */
  registerCommand(cmd: Command, config: TConfig): void

  /** Jalankan fetch */
  fetch(opts: any, config: TConfig): Promise<void>
}
