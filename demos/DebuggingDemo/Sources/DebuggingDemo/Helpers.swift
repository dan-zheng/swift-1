import TensorFlow

extension Conv2D {
  init(filterShape: (Int, Int, Int, Int)) {
    self.init(filterShape: filterShape, filterInitializer: glorotUniform())
  }
}
