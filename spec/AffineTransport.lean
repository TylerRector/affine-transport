/-!
# affine-transport encoder specification

An executable statement of what `src/codec.py` transmits, in dependency order,
with every constant and rounding rule pinned down. Nothing here is imported by
the Python; the two are kept in step by the worked example at the bottom, which
`tools/check_spec.py` reproduces from the Python side.

Run it with `lean --run spec/AffineTransport.lean`. No dependencies.

Layout convention: an image is a flat `Array Float` of `height * width * 3`
entries, row-major, channel fastest. A tile is 4x4 pixels; tile `t` of a
`width`-wide image starts at pixel `(4 * (t / (width / 4)), 4 * (t % (width / 4)))`
and its 16 samples are visited row-major.
-/

namespace AffineTransport

def tileSide : Nat := 4
def tileSamples : Nat := tileSide * tileSide
def quantLevels : Float := 15.0
def smapeEpsilon : Float := 1e-8
def toneGamma : Float := 2.2
def layerRecordBytes : Nat := 8
def atomPayloadBytes : Nat := 12
def colourAxisBytes : Nat := 6
def axisSampleBudget : Nat := 200000
def fillIterations : Nat := 4096

/-- Round half to even, matching `numpy.rint`. Lean's `Float.round` rounds half
away from zero, which is a different function. -/
def roundHalfEven (x : Float) : Float :=
  let low := Float.floor x
  if x - low == 0.5 then
    if low / 2.0 == Float.floor (low / 2.0) then low else low + 1.0
  else
    Float.round x

/-- Round to IEEE 754 binary16, the format of every transmitted scalar. -/
def toFp16 (x : Float) : Float :=
  if x == 0.0 || x.isNaN || x.isInf then x
  else
    let magnitude := x.abs
    if magnitude >= 65520.0 then (if x < 0.0 then -(1.0 / 0.0) else 1.0 / 0.0)
    else
      let exponent := max (Float.floor (Float.log2 magnitude)) (-14.0)
      let step := Float.exp2 (exponent - 10.0)
      roundHalfEven (x / step) * step

def clamp (low high x : Float) : Float := max low (min high x)

/-- Display transfer, `(x / (1 + x)) ^ (1 / 2.2)`. All error is measured here. -/
def tone (x : Float) : Float :=
  let x := max x 0.0
  Float.pow (x / (1.0 + x)) (1.0 / toneGamma)

def untone (t : Float) : Float :=
  let t := clamp 0.0 (1.0 - 1e-9) t
  let p := Float.pow t toneGamma
  p / (1.0 - p)

def smapeTerm (prediction reference : Float) : Float :=
  2.0 * (prediction - reference).abs
    / (prediction.abs + reference.abs + smapeEpsilon)

structure Image where
  width : Nat
  height : Nat
  data : Array Float
deriving Inhabited

def Image.pixels (image : Image) : Nat := image.width * image.height

def Image.channel (image : Image) (pixel channel : Nat) : Float :=
  image.data[3 * pixel + channel]!

def Image.map (image : Image) (f : Float → Float) : Image :=
  { image with data := image.data.map f }

def Image.tiles (image : Image) : Nat :=
  (image.height / tileSide) * (image.width / tileSide)

/-- Pixel index of sample `sample` of tile `tile`. -/
def Image.tileSample (image : Image) (tile sample : Nat) : Nat :=
  let tilesAcross := image.width / tileSide
  let originY := tileSide * (tile / tilesAcross)
  let originX := tileSide * (tile % tilesAcross)
  (originY + sample / tileSide) * image.width + originX + sample % tileSide

/-- A pixel is a hole when the base shading pass wrote nothing to it. -/
def holes (shader : Image) : Array Bool :=
  Array.ofFn (n := shader.pixels) fun pixel =>
    max (max (shader.channel pixel 0) (shader.channel pixel 1))
      (shader.channel pixel 2) <= 0.0

/-- Background behind the holes: the discrete Laplace equation with the
surrounding shaded pixels as boundary data, off-image neighbours replicated from
the edge. Jacobi iteration here; any exact solver agrees. -/
def fillHoles (shader : Image) (hole : Array Bool) : Image := Id.run do
  let width := shader.width
  let height := shader.height
  let mut data := shader.data
  let neighbour := fun (y x : Nat) (dy dx : Int) =>
    let ny := clampInt (Int.ofNat y + dy) 0 (Int.ofNat height - 1)
    let nx := clampInt (Int.ofNat x + dx) 0 (Int.ofNat width - 1)
    (ny.toNat * width + nx.toNat)
  for _ in [0:fillIterations] do
    let previous := data
    for y in [0:height] do
      for x in [0:width] do
        let pixel := y * width + x
        if hole[pixel]! then
          for channel in [0:3] do
            let gather :=
              previous[3 * neighbour y x (-1) 0 + channel]!
                + previous[3 * neighbour y x 1 0 + channel]!
                + previous[3 * neighbour y x 0 (-1) + channel]!
                + previous[3 * neighbour y x 0 1 + channel]!
            data := data.set! (3 * pixel + channel) (gather / 4.0)
  return { shader with data := data }
where
  clampInt (v low high : Int) : Int := max low (min high v)

/-- One transmitted layer: an fp16 transmittance and an fp16 RGB emission,
8 bytes, covering the hole samples of one tile. -/
structure Layer where
  tile : Nat
  transmittance : Float
  emission : Float × Float × Float
deriving Inhabited

/-- Fit `oracle ~ emission + transmittance * background` over the hole samples
of one tile. The transmittance is the least-squares slope over all samples and
channels jointly; the emission absorbs the residual mean. -/
def fitTile (background oracle : Image) (hole : Array Bool) (tile : Nat) :
    Option Layer := Id.run do
  let mut samples : Array Nat := #[]
  for sample in [0:tileSamples] do
    let pixel := background.tileSample tile sample
    if hole[pixel]! then samples := samples.push pixel
  if samples.isEmpty then return none
  let count := samples.size.toFloat
  let mean := fun (image : Image) (channel : Nat) =>
    (samples.foldl (fun acc pixel => acc + image.channel pixel channel) 0.0) / count
  let backgroundMean := (mean background 0, mean background 1, mean background 2)
  let oracleMean := (mean oracle 0, mean oracle 1, mean oracle 2)
  let centred := fun (image : Image) (pixel channel : Nat) (m : Float × Float × Float) =>
    image.channel pixel channel - (match channel with
      | 0 => m.1 | 1 => m.2.1 | _ => m.2.2)
  let mut covariance := 0.0
  let mut variance := 0.0
  for pixel in samples do
    for channel in [0:3] do
      let b := centred background pixel channel backgroundMean
      let o := centred oracle pixel channel oracleMean
      covariance := covariance + b * o
      variance := variance + b * b
  let slope := if variance > 1e-12 then covariance / variance else 0.0
  let transmittance := toFp16 (clamp 0.0 1.0 slope)
  let emit := fun (o b : Float) => toFp16 (o - transmittance * b)
  return some {
    tile := tile
    transmittance := transmittance
    emission :=
      (emit oracleMean.1 backgroundMean.1,
       emit oracleMean.2.1 backgroundMean.2.1,
       emit oracleMean.2.2 backgroundMean.2.2) }

/-- Replace the hole samples with what the layers reconstruct. -/
def applyLayers (background : Image) (hole : Array Bool) (layers : Array Layer) :
    Image := Id.run do
  let mut data := background.data
  for layer in layers do
    for sample in [0:tileSamples] do
      let pixel := background.tileSample layer.tile sample
      if hole[pixel]! then
        for channel in [0:3] do
          let emission := match channel with
            | 0 => layer.emission.1
            | 1 => layer.emission.2.1
            | _ => layer.emission.2.2
          let value :=
            emission + layer.transmittance * background.channel pixel channel
          data := data.set! (3 * pixel + channel) (max value 0.0)
  return { background with data := data }

def fitLayers (background oracle : Image) (hole : Array Bool) :
    Array Layer := Id.run do
  let mut layers : Array Layer := #[]
  for tile in [0:background.tiles] do
    match fitTile background oracle hole tile with
    | some layer => layers := layers.push layer
    | none => pure ()
  return layers

/-- The shared colour direction of the remaining residual: the leading right
singular vector of the residual samples, which is the leading eigenvector of
their 3x3 Gram matrix, found by power iteration. The sign is fixed by making the
components sum positive; the vector is transmitted as three fp16 values and
renormalised in full precision after rounding. -/
def colourAxis (coreTone oracleTone : Image) : Float × Float × Float := Id.run do
  let stride := max 1 (coreTone.pixels / axisSampleBudget)
  let mut gram : Array Float := Array.replicate 9 0.0
  let mut pixel := 0
  while pixel < coreTone.pixels do
    let residual := Array.ofFn (n := 3) fun channel =>
      oracleTone.channel pixel channel - coreTone.channel pixel channel
    for row in [0:3] do
      for column in [0:3] do
        gram := gram.set! (3 * row + column)
          (gram[3 * row + column]! + residual[row]! * residual[column]!)
    pixel := pixel + stride
  let mut vector : Array Float := #[1.0, 1.0, 1.0]
  for _ in [0:64] do
    let next := Array.ofFn (n := 3) fun row =>
      let r : Nat := row.val
      gram[3 * r]! * vector[0]! + gram[3 * r + 1]! * vector[1]!
        + gram[3 * r + 2]! * vector[2]!
    let norm := Float.sqrt (next.foldl (fun acc v => acc + v * v) 0.0)
    vector := if norm > 0.0 then next.map (· / norm) else vector
  let signed :=
    if vector[0]! + vector[1]! + vector[2]! < 0.0 then vector.map (-·) else vector
  let rounded := signed.map toFp16
  let norm := Float.sqrt (rounded.foldl (fun acc v => acc + v * v) 0.0)
  return (rounded[0]! / norm, rounded[1]! / norm, rounded[2]! / norm)

/-- One cleanup atom: 16 signed 5-bit amplitudes along the shared colour axis
plus an fp16 scale, a flat 12 bytes whatever the tile holds. -/
structure Atom where
  tile : Nat
  scale : Float
  amplitudes : Array Float
  gain : Float
deriving Inhabited

def axisComponent (axis : Float × Float × Float) (channel : Nat) : Float :=
  match channel with
  | 0 => axis.1
  | 1 => axis.2.1
  | _ => axis.2.2

/-- Quantise one tile's residual and score it. The gain is the exact sMAPE the
tile gives back, summed over its 16 samples and 3 channels, so atoms are
comparable across the frame and the encoder can simply sort by it. -/
def buildAtom (coreTone oracleTone : Image) (axis : Float × Float × Float)
    (tile : Nat) : Atom := Id.run do
  let mut projected : Array Float := #[]
  for sample in [0:tileSamples] do
    let pixel := coreTone.tileSample tile sample
    let mut amplitude := 0.0
    for channel in [0:3] do
      amplitude := amplitude
        + (oracleTone.channel pixel channel - coreTone.channel pixel channel)
          * axisComponent axis channel
    projected := projected.push amplitude
  let peak := projected.foldl (fun acc v => max acc v.abs) 0.0
  let scale := toFp16 (peak / quantLevels)
  let amplitudes := projected.map fun amplitude =>
    if scale > 0.0 then
      clamp (-quantLevels) quantLevels (roundHalfEven (amplitude / scale))
    else 0.0
  let mut gain := 0.0
  for sample in [0:tileSamples] do
    let pixel := coreTone.tileSample tile sample
    for channel in [0:3] do
      let core := coreTone.channel pixel channel
      let reference := oracleTone.channel pixel channel
      let corrected :=
        clamp 0.0 1.0
          (core + amplitudes[sample]! * scale * axisComponent axis channel)
      gain := gain + smapeTerm core reference - smapeTerm corrected reference
  return { tile := tile, scale := scale, amplitudes := amplitudes, gain := gain }

/-- Atoms in spend order: by gain, ties by tile index. -/
def orderAtoms (atoms : Array Atom) : Array Atom :=
  atoms.qsort fun a b =>
    if a.gain == b.gain then a.tile < b.tile else a.gain > b.gain

def applyAtoms (coreTone : Image) (axis : Float × Float × Float)
    (atoms : Array Atom) (count : Nat) : Image := Id.run do
  let mut data := coreTone.data
  for atom in atoms.extract 0 count do
    for sample in [0:tileSamples] do
      let pixel := coreTone.tileSample atom.tile sample
      for channel in [0:3] do
        let correction :=
          atom.amplitudes[sample]! * atom.scale * axisComponent axis channel
        data := data.set! (3 * pixel + channel)
          (clamp 0.0 1.0 (coreTone.channel pixel channel + correction))
  return { coreTone with data := data }

/-- Elias-Fano size of the sorted tile indices, in bytes. -/
def eliasFanoBytes (tileCount count : Nat) : Nat :=
  if count == 0 then 0
  else
    let lowBits := Nat.log2 (tileCount / count)
    let bits := count * lowBits + count + (tileCount + 2 ^ lowBits - 1) / 2 ^ lowBits
    (bits + 7) / 8

def payloadBytes (layers atoms tileCount : Nat) : Nat :=
  layerRecordBytes * layers + colourAxisBytes + atomPayloadBytes * atoms
    + eliasFanoBytes tileCount atoms

structure Encoded where
  layers : Array Layer
  axis : Float × Float × Float
  atoms : Array Atom
  coreTone : Image
  outputTone : Image
  bytes : Nat

/-- The whole encoder. `count` is the atom budget. -/
def encode (shader oracle : Image) (count : Nat) : Encoded :=
  let hole := holes shader
  let background := fillHoles shader hole
  let layers := fitLayers background oracle hole
  let core := applyLayers background hole layers
  let coreTone := core.map tone
  let oracleTone := oracle.map tone
  let axis := colourAxis coreTone oracleTone
  let atoms := orderAtoms <|
    Array.ofFn (n := coreTone.tiles) fun tile =>
      buildAtom coreTone oracleTone axis tile
  { layers := layers
    axis := axis
    atoms := atoms
    coreTone := coreTone
    outputTone := applyAtoms coreTone axis atoms count
    bytes := payloadBytes layers.size count coreTone.tiles }

def smape (prediction reference : Image) : Float :=
  let total := Array.zip prediction.data reference.data
    |>.foldl (fun acc (p, r) => acc + smapeTerm p r) 0.0
  total / prediction.data.size.toFloat

/-- Worked example: an 8x8 frame whose right half the shader failed to write.
`tools/check_spec.py` prints the same eight numbers from `src/codec.py`. -/
def exampleShader : Image :=
  { width := 8, height := 8
    data := Array.ofFn (n := 8 * 8 * 3) fun index =>
      let pixel := index.val / 3
      let channel := index.val % 3
      let x := pixel % 8
      let y := pixel / 8
      if x >= 4 && y >= 2 && y < 6 then 0.0
      else 0.05 + 0.02 * x.toFloat + 0.01 * y.toFloat + 0.03 * channel.toFloat }

def exampleOracle : Image :=
  { width := 8, height := 8
    data := Array.ofFn (n := 8 * 8 * 3) fun index =>
      let pixel := index.val / 3
      let channel := index.val % 3
      let x := pixel % 8
      let y := pixel / 8
      0.07 + 0.021 * x.toFloat + 0.011 * y.toFloat + 0.028 * channel.toFloat
        + (if x >= 4 && y >= 2 && y < 6 then 0.12 else 0.0) }

def main : IO Unit := do
  let budget := 2
  let result := encode exampleShader exampleOracle budget
  let oracleTone := exampleOracle.map tone
  let axis := result.axis
  let first := result.atoms[0]!
  let report := fun (label value : String) =>
    IO.println s!"{label.pushn ' ' (17 - label.length)}  {value}"
  report "layers" s!"{result.layers.size}"
  report "tiles" s!"{result.coreTone.tiles}"
  report "axis" s!"{axis.1}, {axis.2.1}, {axis.2.2}"
  report "first atom tile" s!"{first.tile}"
  report "first atom gain" s!"{first.gain}"
  report "first atom scale" s!"{first.scale}"
  report "sMAPE core" s!"{smape result.coreTone oracleTone}"
  report s!"sMAPE k={budget}" s!"{smape result.outputTone oracleTone}"
  report "bytes" s!"{result.bytes}"

end AffineTransport

def main : IO Unit := AffineTransport.main
