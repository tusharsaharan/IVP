# Quantum model: circuit, kernels and controls

Owner: Vivek G. This section describes the code in `src/asd_eeg/models/` and
`src/asd_eeg/training/model.py` as implemented. It is the model subsection for the
report; numerical results are in `METHOD.md` and are not repeated here.

## Input

Each subject is one 28-value row in the fixed order of `CONTRACTS.md`: 16 node
band powers (node-major delta, theta, alpha, beta), 6 alpha-envelope connectivity
medians and 6 connectivity IQRs, in edge order (0,1), (0,2), (0,3), (1,2), (1,3),
(2,3). Inside each training fold a `StandardScaler` is fitted on training subjects
only, and every model receives the same bounded angles

$$\phi = \tfrac{\pi}{2}\tanh(z/2) \in (-\tfrac{\pi}{2}, \tfrac{\pi}{2}).$$

The circuit multiplies $\phi$ by a fixed bandwidth (1.0 in all shipped configurations).

## Connectivity-guided feature map

Four qubits, one per configured scalp node (SFARI: F3, F4, P3, P4; TUOS: F7, Fz,
P1, O2). Starting from $|0000\rangle$, a Hadamard is applied to every qubit, then
the following block is repeated for two passes with the same inputs (data
re-uploading):

1. node $n$: $R_Y(\phi_{n,\delta})\,R_Z(\phi_{n,\theta})$ on qubit $n$;
2. edge $(i,j)$: $\mathrm{IsingZZ}(\phi^{\text{median}}_{ij}) = e^{-i\phi Z_iZ_j/2}$ on qubits $i,j$;
3. node $n$: $R_Y(\phi_{n,\alpha})\,R_Z(\phi_{n,\beta})$ on qubit $n$;
4. edge $(i,j)$: $\mathrm{IsingZZ}(\phi^{\text{IQR}}_{ij})$ on qubits $i,j$.

One pass has 16 single-qubit rotations and 12 IsingZZ gates, one gate per input;
the full circuit uses all 28 inputs twice (56 data-dependent gates plus 4 Hadamards).
Every gate angle is a subject feature: there is no trainable or shared fixed
entangling layer, so all entanglement differs between subjects. The coupling
between node power and connectivity comes from the non-commuting order of the
rotations and the ZZ layers. The ZZ gates within one layer commute, so their
order inside a layer has no effect, but **which wire pair** receives which edge
value does. The circuit is deterministic. PennyLane `default.qubit` returns the
exact 16-amplitude statevector (`shots=None`).

Figure: `results/figures/vivek_model/circuit_one_pass.png` (one pass, structure
only). Regenerate the figures locally with `qml.draw_mpl(make_qnode(config))`.
The application renders the same circuit with the subject's numerical angles.

## Kernels

**Graph quantum kernel.** With $|\psi(x)\rangle = U(x)|0000\rangle$,

$$K_Q(x, x') = |\langle\psi(x)\mid\psi(x')\rangle|^2 = \mathrm{Tr}[\rho(x)\rho(x')].$$

This is an inner product of density matrices, so $K_Q$ is positive semidefinite,
symmetric, has unit diagonal, and lies in $[0, 1]$. It does not depend on global phase.
Training states are encoded once per fold and cached in the fitted model. A new
subject's kernel row is computed against all training states **in their original
training order**, which the precomputed-kernel SVM requires.

**Amplitude-fidelity baseline.** Inspired by the reference paper's amplitude-encoded
kernel, not a reproduction of its pipeline. The 28 angles are extended with a
constant 1 (so the zero vector is still a valid state), zero-padded to 32 entries,
normalized, and loaded as a 5-qubit amplitude state. Its fidelity has the exact
classical form

$$K_A(x, x') = \frac{(x\cdot x' + 1)^2}{(\lVert x\rVert^2 + 1)(\lVert x'\rVert^2 + 1)},$$

which is a normalized degree-2 polynomial kernel. The code evaluates this formula
directly. Tests check it against PennyLane `AmplitudeEmbedding` state overlaps to
1e-12. It has no computational advantage: preparing the state costs at least as
much as evaluating the formula.

**RBF-SVM.** Classical RBF kernel on the same 28 angles (`gamma="scale"`).

All four models use an SVC with C = 1.0 and balanced class weights. There is no
hyperparameter search.

## Reassigned-connection control

The control uses the same circuit, but before encoding, the edge inputs are rearranged
with one fixed permutation drawn from `edge_seed` (1729). The same permutation is
applied to every subject and to both the median and the IQR layers. Node features,
gate count, circuit depth and each subject's set of edge values are unchanged; only
which electrode pair receives which connection value changes. This tests whether the
kernel benefits from the anatomically correct assignment of connections to qubit
pairs, or would do just as well with the same numbers on the wrong pairs.

**Limitation found during review.** The seed-1729 permutation is `[0, 1, 2, 4, 5, 3]`.
The three edges touching node 0 keep their own values; only the three edges among
nodes 1, 2 and 3 are cyclically rotated. For SFARI that means F4–P3 receives F4–P4's
values, F4–P4 receives P3–P4's, and P3–P4 receives F4–P3's. The control therefore
perturbs half of the graph, not all of it. This is now pinned by a regression test.
It was not changed, because every frozen bundle and the recorded run depend on it.
A full derangement, or several random permutations, would be a stronger control in
a new experiment and must be decided before results are viewed.

## Relationship to the reference paper

Saranya and Menaka (2025, IEEE Access, DOI 10.1109/ACCESS.2025.3531979) classify ASD
EEG with wavelet/filter-bank CSP and engineered features followed by a quantum
kernel classifier, on a private 20-child cohort. This project borrows only the idea
of a state-fidelity kernel: our amplitude baseline is inspired by their kernel, but
it is applied to our 28 graph features without their CSP/wavelet pipeline. The
connectivity-guided circuit (one qubit per scalp node, ZZ couplings driven by
connectivity on the matching qubit pair) and the reassignment control are our
adaptation. The paper's accuracy must not be attributed to this code, cohort or
features.

## Verification

`tests/test_quantum_math.py` and `tests/test_quantum.py` check:

- unit-norm states;
- a symmetric, unit-diagonal, PSD kernel;
- that every node and edge input changes the state;
- that the amplitude formula matches the quantum state fidelity, including the zero vector;
- training-only scaling;
- identical predictions after bundle save and reload for all four models;
- that the reassigned circuit equals the original circuit applied to permuted edge columns
  (node features unchanged, edge values only permuted);
- the pinned permutation.

A separate review script reimplemented the circuit in plain NumPy matrices. Its
states agreed with PennyLane to a fidelity of $1 - 5\times10^{-15}$ for 40 random
inputs, with and without reassignment. Each of the 28 inputs individually changed the
state.

## Computational limits

Four qubits give 16 amplitudes; exact simulation is trivial on a laptop, so this
work makes **no quantum speedup or hardware claim**. Results are noiseless and
shot-free. Real hardware would need fidelity estimation from finite shots (for
example, compute-uncompute), and noise would bias the kernel. Training costs
$O(n)$ circuit simulations plus an $O(n^2)$ kernel matrix. Each prediction costs one
simulation and $n$ overlaps against the stored reference states. With bounded
angles and random inputs, off-diagonal kernel values average near
$1/16$. Such concentration can reduce SVM margins when features are noisy. This was
not tuned (bandwidth fixed at 1.0) to avoid score-driven changes. On the recorded
SFARI cohort none of the four models beat the majority-class baseline (`METHOD.md`),
so no benefit of the connectivity-guided encoding is established.
