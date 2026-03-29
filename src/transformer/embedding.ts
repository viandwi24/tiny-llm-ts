export class Embedding {
  tokenWeights: number[][]
  positionWeights: number[][]
  embedSize: number

  // gradient — diisi saat backward(), dibaca oleh optimizer
  gradTokenWeights: number[][] = []
  gradPositionWeights: number[][] = []

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

  backward(inputIds: number[], dOutput: number[][]): { dTokenWeights: number[][]; dPositionWeights: number[][] } {
    const dTokenWeights = Array.from({ length: this.tokenWeights.length }, () =>
      Array(this.embedSize).fill(0)
    )
    const dPositionWeights = Array.from({ length: this.positionWeights.length }, () =>
      Array(this.embedSize).fill(0)
    )

    inputIds.forEach((id, pos) => {
      for (let i = 0; i < this.embedSize; i++) {
        dTokenWeights[id]![i]! += dOutput[pos]![i]!
        dPositionWeights[pos]![i]! += dOutput[pos]![i]!
      }
    })

    this.gradTokenWeights = dTokenWeights
    this.gradPositionWeights = dPositionWeights
    return { dTokenWeights, dPositionWeights }
  }
}