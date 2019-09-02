struct MLP: Layer {
    var layer1 = Dense<Float>(inputSize: 2, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 1, activation: relu)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let h1 = layer1(input)
        return layer2(h1)
    }
}

var classifier = MLP()
let optimizer = SGD(for: classifier, learningRate: 0.02)

let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y: Tensor<Float> = [0, 1, 1, 0]

let (loss, 𝛁model) = valueWithGradient(at: classifier) { m -> Float in
    let ŷ = m(x)
    print("ŷ =\n\(ŷ)")
    let 𝚫y = ŷ - y
    print("𝚫y =\n\(𝚫y)")
    return 𝚫y.squared().mean().scalarized()
}

print("Layer 2 weight: \(𝛁model.layer2.weight)")
