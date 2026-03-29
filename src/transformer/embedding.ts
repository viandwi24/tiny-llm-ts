export class Embedding {
  tokenWeights: number[][]
  positionWeights: number[][]
  embedSize: number

  constructor(vocabSize: number, embedSize: number, maxSeqLen: number) {
    this.embedSize = embedSize

    // init token weights, random
    this.tokenWeights = Array.from({ length: vocabSize }, () =>
      Array.from({ length: embedSize }, () =>
        (Math.random() * 2 - 1) * 0.01
      )
    )

    // init position weights, random
    this.positionWeights = Array.from({ length: maxSeqLen }, () =>
      Array.from({ length: embedSize }, () =>
        (Math.random() * 2 - 1) * 0.01
      )
    )
  }

  forward(inputIds: number[]): number[][] {
    return inputIds.map((id, pos) => this.tokenWeights[id]!.map((w, i) => w + this.positionWeights[pos]![i]!))
  }
}