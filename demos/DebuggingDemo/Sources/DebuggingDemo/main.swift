import TensorFlow

struct MLP: Layer {
    var layer1 = Dense<Float>(inputSize: 2, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 30, activation: relu)
    var layer3 = Dense<Float>(inputSize: 30, outputSize: 10, activation: relu)
    var layer4 = Dense<Float>(inputSize: 10, outputSize: 1, activation: relu)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let h1 = layer1(input)
        let h2 = layer2(h1)
        let h3 = layer3(h2)
        return layer4(h3)
    }
}

var classifier = MLP()
let optimizer = SGD(for: classifier, learningRate: 0.02)

let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y: Tensor<Float> = [0, 1, 1, 0]

let (loss, 𝛁model) = valueWithGradient(at: classifier) { m -> Float in
   let ŷ = classifier(x)
   print("ŷ =\n\(ŷ)")
   let 𝚫y = ŷ - y
   print("𝚫y =\n\(𝚫y)")
   return 𝚫y.squared().mean().scalarized()
}
optimizer.update(&classifier, along: 𝛁model)

print("Layer 4 weight:\n\(𝛁model.layer4.weight)")
