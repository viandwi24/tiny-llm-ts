import { LinearLayer } from '../neural-network/layer'
import { gelu } from '../neural-network/activation'

export class FeedForward {
  linear1: LinearLayer
  linear2: LinearLayer

  constructor(embedDim: number, ffnDim: number) {
    this.linear1 = new LinearLayer(embedDim, ffnDim)
    this.linear2 = new LinearLayer(ffnDim, embedDim)
  }

  forward(x: number[][]): number[][] {
    let out = this.linear1.forward(x)
    out = out.map(row => row.map(v => gelu(v)))
    return this.linear2.forward(out)
  }
}