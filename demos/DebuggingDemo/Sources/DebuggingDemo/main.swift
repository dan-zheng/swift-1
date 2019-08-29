import TensorFlow

struct Model: Layer {
  var conv = Conv2D<Float>(filterShape: (5, 5, 3, 6))
  var maxpool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  var flatten = Flatten<Float>()
  var dense = Dense<Float>(inputSize: 36 * 6, outputSize: 10)

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let h0 = conv(input)
    let h1 = maxpool(h0)
    let h2 = flatten(h1).withDerivative { print($0) }
    return dense(h2)
  }
}

let x = Tensor<Float>(randomNormal: [10, 16, 16, 3])
let y = Tensor<Int32>(rangeFrom: 0, to: 10, stride: 1)

var model = Model()
let optimizer = SGD(for: model)
Context.local.learningPhase = .training

for i in 1...10 {
  let (loss, grads) = valueWithGradient(at: model) { m -> Tensor<Float> in
    let logits = model(x)
    return softmaxCrossEntropy(logits: logits, labels: y)
  }
  print("Step \(i), loss is: \(loss)")
  optimizer.update(&model, along: grads)
}
