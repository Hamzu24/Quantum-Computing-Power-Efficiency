Here's the full dependency-ordered list, bottom-up from IBM calibration data to final temperature-dependent predictions.

Global constants (not per-qubit)
These remain global because they're material or process constants that IBM doesn't expose per-qubit:
Parameter
Value
Justification
Δ (superconducting gap, Al)
~2.88×10⁻²³ J (180 μeV)
Material constant; same Al film across chip
ζ⁻¹ (subgap transparency factor)
~10³–10⁵
Fabrication-process constant; same junction recipe
γ_MXC/γ₁⁰
~0.85–0.95 (needs sensitivity analysis)
Property of cryostat shielding/filtering, not per-qubit
T_op (operational MXC temperature)
13 mK
Cryostat setpoint


IBM calibration data needed per qubit
From backend properties (e.g. backend.properties() or BackendProperties):
Field
Symbol
Units
frequency
ω_ge/2π
Hz
anharmonicity
α/2π
Hz (negative for transmon)
T1
T1_meas
s
T2
T2_meas
s
prob_meas1_prep0
P(1|0)_raw
dimensionless
prob_meas0_prep1
P(0|1)
dimensionless

The last two together form the readout confusion matrix. You also need per-gate gate_error and gate_length for fidelity calculations later.

Per-qubit parameter derivation (in dependency order)
Step 1: E_c (charging energy)
Equation:
$$E_c = -\hbar \alpha$$
where α is the anharmonicity (negative for a transmon).
Reference: Koch et al., PRA 76, 042319 (2007), Eq. (2.5): for a transmon in the regime E_J/E_c ≫ 1, the anharmonicity approaches α → −E_c/ℏ. This is the leading-order result; corrections are exponentially small in √(E_J/E_c).
Justification: IBM provides anharmonicity per-qubit because it varies with junction parameters. This directly gives you E_c without any fitting.
IBM data used: anharmonicity

Step 2: E_J (Josephson energy)
Equation:
$$E_J = \frac{(\hbar\omega_{ge} + E_c)^2}{8,E_c}$$
This comes from inverting the transmon frequency formula ℏω_ge ≈ √(8E_JE_c) − E_c.
Reference: Koch et al. (2007), Eq. (2.5). The transition frequency of the 0→1 transition is ω₀₁ ≈ (√(8E_JE_c) − E_c)/ℏ. Solving for E_J gives the equation above.
Justification: Different qubits on the same chip are deliberately fabricated with different E_J (and hence different frequencies) to avoid frequency collisions. This is the primary source of per-qubit variation in all downstream quantities.
IBM data used: frequency, plus E_c from Step 1
Sanity check: Verify E_J/E_c ≫ 1 (should be ~30–80 for IBM transmons). If not, the approximation breaks down.

Step 3: ω_p (plasma frequency)
Equation:
$$\omega_p = \frac{\sqrt{8,E_J,E_c}}{\hbar}$$
Reference: Catelani, Koch et al., PRL 106, 077002 (2011), and Lvov et al. (2025) text above Eq. (26). The plasma frequency sets the scale for QP-induced transition rates.
Justification: Since E_J varies per-qubit (Step 2), ω_p varies per-qubit. In the current code, ω_p is global — this is the source of issue #5. The ratio ω_p²/ω_ge that appears in γ_QP and γ_φ_QP is sensitive to per-qubit E_J variation.
Note on screening: At T ≲ 300 mK, the QP screening correction (ω_p → ω_p√(1−2x_QP)) is negligible. You can safely ignore it and document why.

Step 4: R_n (normal-state junction resistance)
Equation:
$$R_n = \frac{\pi,\hbar,\Delta}{4,e^2,E_J}$$
This is the Ambegaokar-Baratoff relation inverted: I_c = πΔ/(2eR_n) and E_J = ℏI_c/(2e), giving E_J = πℏΔ/(4e²R_n).
Reference: Ambegaokar & Baratoff, PRL 10, 486 (1963); Koch et al. (2007), Sec. II.
Justification: Different E_J per qubit implies different junction areas and hence different R_n. The current code uses a global R_n = 5 kΩ, which is inconsistent with per-qubit E_J. For a typical IBM transmon with E_J/h ~ 15 GHz: R_n ≈ πℏ·(180 μeV)/(4e²·h·15 GHz) ≈ 5–10 kΩ — the right ballpark.
IBM data used: E_J from Step 2, Δ (global)

Step 5: N_e (effective number of junction channels)
Equation:
$$N_e = \frac{1}{\zeta} \cdot \frac{g_T}{2,g_K} = \frac{1}{\zeta} \cdot \frac{1}{2,R_n,g_K}$$
where g_K = e²/h is the conductance quantum and ζ is the subgap transparency.
Reference: Catelani et al., PRB 86, 184514 (2012), discussion around Eq. (15); Lvov et al. (2025), Eq. (29) and surrounding text. They write N_e = ζ⁻¹g_T/(2g_K).
Justification: Since R_n varies per-qubit (Step 4), N_e varies per-qubit. This enters γ_φ_QP as 1/√N_e, so a factor-of-2 variation in R_n produces a ~40% change in γ_φ_QP.
IBM data used: R_n from Step 4, ζ (global)

Step 6: Corrected excited-state population p_e
Equation:
$$p_e = \frac{P(1|0){\text{raw}} - \epsilon{01}}{1 - \epsilon_{01} - \epsilon_{10}}$$
where ε₀₁ = P(1|0) due to pure readout misassignment (not thermal) and ε₁₀ = P(0|1).
Practical approach: IBM provides both prob_meas1_prep0 and prob_meas0_prep1. These satisfy:
$$P(1|0){\text{raw}} = (1 - p_e),\epsilon{01}^{\text{ro}} + p_e,(1 - \epsilon_{10}^{\text{ro}})$$
Disentangling readout error from thermal population is non-trivial without additional calibration. A simpler conservative approach: if the backend has readout error mitigation data, use it. Otherwise, flag that your T_env estimate has an upward bias proportional to readout assignment error.
Reference: No single paper; this is standard practice in IBM's Qiskit readout error mitigation framework. See Bravyi et al., PRA 103, 042605 (2021) for readout error mitigation.
Justification: This is issue #12. At 13 mK, true thermal p_e ~ 10⁻³–10⁻⁴ for ~5 GHz transmons, while typical readout assignment error is ~1–3%. Without correction, you'd massively overestimate p_e and hence T_env.
IBM data used: prob_meas1_prep0, prob_meas0_prep1

Step 7: T_env (effective environment temperature)
Equation:
$$n_{\text{eff}} = \frac{p_e}{1 - 2,p_e}$$
$$T_{\text{env}} = \frac{\hbar\omega_{ge}}{k_B \ln!\left(1 + \frac{\gamma_{\text{env}}/\gamma_1^0}{n_{\text{eff}}}\right)}$$
where γ_env/γ₁⁰ = 1 − γ_MXC/γ₁⁰ (global ratio). This assumes n_MXC(T_op) ≈ 0 at 13 mK.
Reference: Lvov et al. (2025), Eq. (14) specialized to the low-T_MXC limit; your paper's Sec. on T_env estimation.
Justification: Per-qubit because ω_ge varies per qubit and p_e varies per qubit. The original code computes per-qubit T_env estimates and takes the median — you should instead keep the per-qubit values. Different qubits may couple to the environment differently (edge qubits vs. interior, proximity to control lines, etc.).
Decision: You can either use per-qubit T_env values directly, or take the median if you believe the environment is spatially uniform and want noise robustness. Document which choice you make and why.
IBM data used: Corrected p_e from Step 6, frequency, γ_MXC/γ₁⁰ (global)

Step 8: γ₁⁰ (zero-temperature relaxation rate)
Equation (closed-form, not optimization):
At T_op = 13 mK, γ_QP(T_op) ≈ 0 and n_MXC(T_op) ≈ 0, so from Lvov Eq. (15) and (27):
$$\frac{1}{T_{1,\text{meas}}} = \gamma_1^0,(2,n_{\text{eff}} + 1)$$
where n_eff = (1 − γ_MXC/γ₁⁰)·n(ω_ge, T_env). Solving:
$$\gamma_1^0 = \frac{1}{T_{1,\text{meas}},(2,n_{\text{eff}} + 1)}$$
Reference: Lvov et al. (2025), Eq. (15) and (27).
Justification: Per-qubit because T1_meas, ω_ge, and T_env all vary per-qubit. When fitting per-qubit, this becomes a direct analytic inversion — no optimizer needed. The optimizer was only necessary when fitting a single global γ₁⁰ across all qubits.
Important: n_eff here uses the per-qubit T_env from Step 7. If using median T_env, use that instead — just be consistent.
IBM data used: T1, n_eff from Step 7

Step 9: γ_φ_base (base pure dephasing rate)
Equation (closed-form):
At T_op = 13 mK, γ_φ_QP(T_op) ≈ 0, so:
$$\frac{1}{T_{2,\text{meas}}} = \frac{1}{2,T_{1,\text{meas}}} + \gamma_{\varphi,\text{base}}$$
$$\gamma_{\varphi,\text{base}} = \frac{1}{T_{2,\text{meas}}} - \frac{1}{2,T_{1,\text{meas}}}$$
Reference: Standard decoherence decomposition; see Krantz et al., Appl. Phys. Rev. 6, 021318 (2019), Sec. IV. Also implicit in Lvov et al. discussion around Eq. (29).
Justification: Per-qubit because T1 and T2 vary per-qubit. This captures qubit-specific dephasing from 1/f flux noise, charge noise, TLS, etc. Again, closed-form — no optimizer.
Caveat (issue #13): Verify whether IBM reports T2 from Ramsey (T2*) or echo (T2_echo). If echo:
$$\gamma_{\varphi,\text{base}}^{\text{echo}} = \frac{1}{T_{2,\text{echo}}} - \frac{1}{2,T_1}$$
This will give a smaller γ_φ_base than Ramsey would. The choice affects temperature extrapolation since γ_φ_QP adds equally to both, but the base value differs. IBM's recent backends report T2 from echo. Use consistently.
Physical check: γ_φ_base must be ≥ 0. If T2 > 2T1 (which means pure dephasing is negligible), set γ_φ_base = 0.
IBM data used: T1, T2

Temperature extrapolation (using per-qubit parameters)
With all per-qubit parameters from Steps 1–9, the temperature-dependent quantities at arbitrary T are:
x_QP(T) — quasiparticle density:
$$x_{\text{QP}}(T) = \sqrt{\frac{2\pi k_B T}{\Delta}},\exp!\left(-\frac{\Delta}{k_B T}\right)$$
γ_QP(T) — QP relaxation rate (Lvov Eq. 26):
$$\gamma_{\text{QP}}(T) = \frac{\omega_p^2}{\pi,\omega_{ge}}\left[x_{\text{QP}}\sqrt{\frac{2\Delta}{\hbar\omega_{ge}}} + 4,e^{-\Delta/k_BT}\cosh!\left(\frac{\hbar\omega_{ge}}{2k_BT}\right)K_0!\left(\frac{\hbar\omega_{ge}}{2k_BT}\right)\right]$$
All of ω_p, ω_ge are now per-qubit.
γ_φ_QP(T) — QP dephasing rate (Lvov Eq. 29):
$$\gamma_{\varphi,\text{QP}}(T) \sim 4\pi,\frac{\omega_p^2}{\omega_{ge}}\sqrt{\frac{x_{\text{QP}}^A}{N_e}}$$
where x_QP^A = exp(−Δ/k_BT). Both ω_p and N_e are now per-qubit.
T1(T):
$$T_1(T) = \frac{1}{\gamma_{\text{QP}}(T) + \gamma_1^0,(2,n_{\text{eff}}(T) + 1)}$$
T2(T):
$$T_2(T) = \frac{1}{\frac{1}{2,T_1(T)} + \gamma_{\varphi,\text{base}} + \gamma_{\varphi,\text{QP}}(T)}$$
Gate fidelity (per gate, not per qubit):
For an N-qubit gate, compute T1 and T_φ per constituent qubit, then:
$$F_N = 1 - \frac{N,t_{\text{gate}}}{2(d+1)}\sum_{i=1}^{N}\left(\frac{1}{T_{1,i}} + \frac{1}{T_{\varphi,i}}\right)$$
with d = 2^N. This replaces the average-frequency approximation (issue #16).

Summary dependency graph
IBM: ω_ge, α ──→ [1] E_c ──→ [2] E_J ──→ [3] ω_p
                                    │          │
                                    ├──→ [4] R_n ──→ [5] N_e
                                    │
IBM: P(1|0), P(0|1) ──→ [6] p_e ──→ [7] T_env
                                              │
IBM: T1 ─────────────────────────────→ [8] γ₁⁰
IBM: T1, T2 ─────────────────────────→ [9] γ_φ_base

All feed into: T1(T), T2(T), F_N(T)
The key insight: making this per-qubit eliminates the need for numerical optimization entirely — every parameter has a closed-form expression. The optimizer was only an artifact of fitting a single global value across heterogeneous qubits.

