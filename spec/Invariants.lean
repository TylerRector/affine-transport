import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Why the encoder in `AffineTransport.lean` is allowed to work that way

Two facts the encoder leans on, stated over `ℝ` where they are provable. This
file is the only thing here that needs Mathlib, and nothing in the pipeline
depends on it.

1. A transparent layer acts on a background radiance field by the affine map

      eval L B x = emit x + trans x * B (warp x)

   Front-to-back composition makes `AffineLayer X` a monoid, so flattening a
   depth-ordered stack agrees with evaluating its samples in order, and a
   layer with transmittance in `[0,1]` never amplifies a background change.

2. Cleanup atoms all carry the same payload, so the selection is separable:
   taking exactly the atoms whose distortion gain covers `lambda * rate`
   minimises the Lagrangian cost of the whole frame. This is why `orderAtoms`
   can sort by gain once and cut the list at any budget.

Colour is `Fin 3 → ℝ`; no floating-point semantics are assumed, so none of this
transfers to the `Float` arithmetic of the encoder without an error argument.
-/

abbrev RGB := Fin 3 → ℝ

@[ext]
structure AffineLayer (X : Type*) where
  emit : X → RGB
  trans : X → ℝ
  warp : X → X

namespace AffineLayer

variable {X : Type*}

/-- Apply one transparent layer to a background radiance field. -/
def eval (layer : AffineLayer X) (background : X → RGB) : X → RGB :=
  fun x channel =>
    layer.emit x channel + layer.trans x * background (layer.warp x) channel

/-- Front-to-back composition: `front` is encountered first. -/
def stack (front back : AffineLayer X) : AffineLayer X where
  emit := fun x channel =>
    front.emit x channel + front.trans x * back.emit (front.warp x) channel
  trans := fun x => front.trans x * back.trans (front.warp x)
  warp := fun x => back.warp (front.warp x)

/-- The fully transparent layer. -/
def identity : AffineLayer X where
  emit := fun _ _ => 0
  trans := fun _ => 1
  warp := id

theorem stack_assoc (a b c : AffineLayer X) :
    stack (stack a b) c = stack a (stack b c) := by
  apply AffineLayer.ext
  · funext x channel
    simp only [stack]
    ring
  · funext x
    simp only [stack]
    ring
  · rfl

theorem identity_stack (layer : AffineLayer X) : stack identity layer = layer := by
  apply AffineLayer.ext <;> funext x <;> simp [identity, stack]

theorem stack_identity (layer : AffineLayer X) : stack layer identity = layer := by
  apply AffineLayer.ext <;> funext x <;> simp [identity, stack]

instance : Monoid (AffineLayer X) where
  mul := stack
  one := identity
  mul_assoc := stack_assoc
  one_mul := identity_stack
  mul_one := stack_identity

theorem mul_def (a b : AffineLayer X) : a * b = stack a b := rfl

/-- Composing then evaluating equals nested evaluation. -/
theorem eval_stack (front back : AffineLayer X) (background : X → RGB) :
    eval (stack front back) background = eval front (eval back background) := by
  funext x channel
  simp only [eval, stack]
  ring

/-- Flatten a depth-ordered stack into one equivalent layer. -/
def flatten : List (AffineLayer X) → AffineLayer X
  | [] => identity
  | layer :: rest => stack layer (flatten rest)

theorem flatten_eq_prod (layers : List (AffineLayer X)) : flatten layers = layers.prod := by
  induction layers with
  | nil => simp only [flatten, List.prod_nil]; rfl
  | cons layer rest ih => simp [flatten, List.prod_cons, mul_def, ih]

/-- Evaluating a flattened deep pixel equals evaluating the samples in order. -/
theorem eval_flatten (layers : List (AffineLayer X)) (background : X → RGB) :
    eval (flatten layers) background
      = layers.foldr (fun layer accumulated => eval layer accumulated) background := by
  induction layers with
  | nil =>
      funext x channel
      simp [flatten, identity, eval]
  | cons layer rest ih => simp [flatten, eval_stack, ih]

/-- A background perturbation is scaled by the transmittance. -/
theorem background_delta (layer : AffineLayer X) (background₀ background₁ : X → RGB)
    (x : X) (channel : Fin 3) :
    eval layer background₁ x channel - eval layer background₀ x channel
      = layer.trans x
        * (background₁ (layer.warp x) channel - background₀ (layer.warp x) channel) := by
  simp only [eval]
  ring

/-- Transmittance in `[0,1]` cannot amplify a change in the sampled background. -/
theorem background_nonexpansive (layer : AffineLayer X)
    (h₀ : ∀ x, 0 ≤ layer.trans x) (h₁ : ∀ x, layer.trans x ≤ 1)
    (background₀ background₁ : X → RGB) (x : X) (channel : Fin 3) :
    |eval layer background₁ x channel - eval layer background₀ x channel|
      ≤ |background₁ (layer.warp x) channel - background₀ (layer.warp x) channel| := by
  rw [background_delta, abs_mul, abs_of_nonneg (h₀ x)]
  nlinarith [abs_nonneg (background₁ (layer.warp x) channel
    - background₀ (layer.warp x) channel), h₀ x, h₁ x]

end AffineLayer

namespace RateDistortion

variable {ι : Type*}

/-- Lagrangian cost of taking or skipping one equal-cost cleanup atom. -/
def atomCost (lambda gain rate : ℝ) (take : Bool) : ℝ :=
  if take then lambda * rate - gain else 0

theorem take_le (lambda gain rate : ℝ) (h : lambda * rate ≤ gain) (choice : Bool) :
    atomCost lambda gain rate true ≤ atomCost lambda gain rate choice := by
  cases choice with
  | false => simp [atomCost]; linarith
  | true => simp [atomCost]

theorem skip_le (lambda gain rate : ℝ) (h : gain ≤ lambda * rate) (choice : Bool) :
    atomCost lambda gain rate false ≤ atomCost lambda gain rate choice := by
  cases choice with
  | false => simp [atomCost]
  | true => simp [atomCost]; linarith

/-- The threshold rule the encoder uses when it sorts atoms by gain. -/
noncomputable def greedy (lambda : ℝ) (gain rate : ι → ℝ) (i : ι) : Bool :=
  decide (lambda * rate i ≤ gain i)

theorem greedy_le (lambda : ℝ) (gain rate : ι → ℝ) (i : ι) (choice : Bool) :
    atomCost lambda (gain i) (rate i) (greedy lambda gain rate i)
      ≤ atomCost lambda (gain i) (rate i) choice := by
  by_cases h : lambda * rate i ≤ gain i
  · simpa [greedy, h] using take_le lambda (gain i) (rate i) h choice
  · have h' : gain i ≤ lambda * rate i := le_of_lt (lt_of_not_ge h)
    simpa [greedy, h] using skip_le lambda (gain i) (rate i) h' choice

/-- Frame cost of a selection. -/
def frameCost (s : Finset ι) (lambda : ℝ) (gain rate : ι → ℝ) (take : ι → Bool) : ℝ :=
  ∑ i ∈ s, atomCost lambda (gain i) (rate i) (take i)

/-- Equal payloads make the frame decision separable: the pointwise threshold
rule is globally optimal. -/
theorem greedy_frameCost_le (s : Finset ι) (lambda : ℝ) (gain rate : ι → ℝ)
    (take : ι → Bool) :
    frameCost s lambda gain rate (greedy lambda gain rate)
      ≤ frameCost s lambda gain rate take :=
  Finset.sum_le_sum fun i _ => greedy_le lambda gain rate i (take i)

/-- Transmitted size: layer records, the shared colour axis, the atom index,
and one fixed-size payload per atom. -/
def payloadBytes (layers atoms indexBytes : ℕ) : ℕ :=
  8 * layers + 6 + indexBytes + 12 * atoms

theorem payloadBytes_mono (layers indexBytes : ℕ) {a b : ℕ} (h : a ≤ b) :
    payloadBytes layers a indexBytes ≤ payloadBytes layers b indexBytes := by
  unfold payloadBytes
  omega

end RateDistortion
