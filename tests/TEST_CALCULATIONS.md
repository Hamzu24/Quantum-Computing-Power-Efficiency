# Manual Calculations for DefaultBuilder Unit Tests

This document provides detailed manual calculations for verifying the DefaultBuilder physics calculations.

## Physical Constants Used

```python
k = 1.380649e-23 J/K      # Boltzmann constant
hbar = 1.054571817e-34 J·s # Reduced Planck constant
e = 1.602176634e-19 C      # Elementary charge
h = 6.62607015e-34 J·s     # Planck constant
```

## Test Parameters

Standard test configuration:
```python
delta = 1.764e-23 J         # Superconducting gap (Al at ~1.2 K)
E_J = 2.1e-22 J             # Josephson energy
E_c = 3.5e-24 J             # Charging energy
y0 = 1e5 s^-1               # Base relaxation rate
gamma_psi_base = 1e3 s^-1   # Base dephasing rate
T_env = 0.050 K             # Environment temperature (50 mK)
ymxc_y0 = 0.8               # MXC bath coupling ratio
R_n = 5000 Ω                # Normal state resistance
subgap_transparency = 0.01  # Subgap transparency
```

## 1. Quasiparticle Density (x_qp)

**Formula:**
```
x_qp = sqrt(2*π*k*T/Δ) * exp(-Δ/(k*T))
```

**Physical meaning:** Fraction of electrons that are unpaired (quasiparticles) rather than Cooper pairs.

**Test case: T = 0.020 K (20 mK)**

Step-by-step calculation:
1. Calculate k*T:
   ```
   k*T = 1.380649e-23 * 0.020 = 2.761298e-25 J
   ```

2. Calculate exponent argument:
   ```
   Δ/(k*T) = 1.764e-23 / 2.761298e-25 = 63.883
   ```

3. Calculate square root argument:
   ```
   2*π*k*T/Δ = 2 * 3.14159 * 2.761298e-25 / 1.764e-23
            = 1.735e-24 / 1.764e-23
            = 0.098428
   ```

4. Calculate components:
   ```
   sqrt(0.098428) = 0.31373
   exp(-63.883) = 1.467e-28
   ```

5. Final result:
   ```
   x_qp = 0.31373 * 1.467e-28 = 4.603e-29
   ```

**Physical interpretation:** At 20 mK, only ~4.6e-29 of electrons are quasiparticles. This is exponentially suppressed due to the large gap-to-temperature ratio.

## 2. Plasma Frequency (w_p)

**Formula:**
```
w_p0 = sqrt(8*E_J*E_c) / hbar
screening_factor = 1 - 2*x_qp
w_p = w_p0 * sqrt(screening_factor)
```

**Physical meaning:** Natural oscillation frequency of charge in the Josephson junction.

**Test case: T = 0.015 K (15 mK)**

Step-by-step calculation:
1. Calculate base plasma frequency:
   ```
   8*E_J*E_c = 8 * 2.1e-22 * 3.5e-24 = 5.88e-45 J²
   sqrt(5.88e-45) = 7.668e-23 J
   w_p0 = 7.668e-23 / 1.054571817e-34 = 7.272e11 rad/s
   ```

2. Calculate screening at this temperature:
   ```
   x_qp(0.015 K) ≈ 2e-30 (very small)
   screening_factor = 1 - 2*(2e-30) ≈ 1.0
   ```

3. Final result:
   ```
   w_p ≈ 7.272e11 * sqrt(1.0) = 7.272e11 rad/s
   ```

**Physical interpretation:** At low temperatures, screening from quasiparticles is negligible, so plasma frequency equals its base value.

## 3. Bose-Einstein Distribution

**Formula:**
```
n(ω, T) = 1 / (exp(ℏω/(k*T)) - 1)
```

**Physical meaning:** Average number of thermal photons at frequency ω and temperature T.

**Test case: ω = 5e10 rad/s, T = 0.025 K (25 mK)**

Step-by-step calculation:
1. Calculate photon energy:
   ```
   ℏω = 1.054571817e-34 * 5e10 = 5.273e-24 J
   ```

2. Calculate thermal energy:
   ```
   k*T = 1.380649e-23 * 0.025 = 3.452e-25 J
   ```

3. Calculate ratio:
   ```
   x = ℏω/(k*T) = 5.273e-24 / 3.452e-25 = 15.28
   ```

4. Calculate exponential:
   ```
   exp(15.28) = 4,316,000
   ```

5. Final result:
   ```
   n = 1 / (4,316,000 - 1) ≈ 2.317e-7
   ```

**Physical interpretation:** At 25 mK with 8 GHz qubit frequency, thermal occupation is ~2.3e-7 photons - essentially zero. The system is deep in the quantum regime.

## 4. Effective Photon Number (n_eff)

**Formula:**
```
n_eff = γ_MXC_ratio * n_MXC + (1 - γ_MXC_ratio) * n_env
```

**Physical meaning:** Weighted average of thermal photons from the MXC (mixing chamber) and environment baths.

**Test case: T_MXC = 0.020 K, T_env = 0.050 K, γ_MXC_ratio = 0.8, ω = 5e10 rad/s**

Step-by-step calculation:
1. Calculate n_MXC at 20 mK:
   ```
   x_MXC = ℏω/(k*T_MXC) = 5.273e-24 / (1.38e-23 * 0.020) = 19.10
   n_MXC = 1 / (exp(19.10) - 1) ≈ 4.95e-9
   ```

2. Calculate n_env at 50 mK:
   ```
   x_env = ℏω/(k*T_env) = 5.273e-24 / (1.38e-23 * 0.050) = 7.64
   n_env = 1 / (exp(7.64) - 1) ≈ 4.88e-4
   ```

3. Calculate weighted average:
   ```
   n_eff = 0.8 * 4.95e-9 + 0.2 * 4.88e-4
        = 3.96e-9 + 9.76e-5
        ≈ 9.76e-5
   ```

**Physical interpretation:** Even though MXC is colder, the environment contribution dominates because it's weighted at 20% but has ~10^5 times more thermal photons.

## 5. Quasiparticle Relaxation Rate (gamma_qp)

**Formula:**
```
γ_qp = (ω_p²)/(π*ω_ge) * [x_qp*sqrt(2Δ/(ℏω_ge)) + 4*exp(-Δ/(k*T))*cosh(ℏω_ge/(2k*T))*K_0(ℏω_ge/(2k*T))]
```

where K_0 is the modified Bessel function of the second kind.

**Physical meaning:** Rate at which quasiparticles cause energy relaxation (T1 processes).

**Test case: T = 0.020 K, ω = 5e10 rad/s**

This is a complex calculation involving Bessel functions. Key steps:

1. Calculate ω_p ≈ 7.27e11 rad/s (from above)
2. Calculate x_qp ≈ 4.6e-29 (from above)
3. Term 1: x_qp * sqrt(2Δ/(ℏω)) ≈ 4.6e-29 * sqrt(2*1.764e-23/(1.05e-34*5e10))
                                  ≈ 4.6e-29 * 2.59 ≈ 1.2e-28
4. Term 2 involves: exp(-Δ/(k*T)) ≈ 1.47e-28 and Bessel function K_0
5. γ_qp ≈ (7.27e11)²/(π*5e10) * (small terms) ~ O(1e3 - 1e5) s^-1

**Physical interpretation:** Quasiparticle-induced relaxation is typically subdominant at low temperatures compared to other loss mechanisms.

## 6. T1 (Energy Relaxation Time)

**Formula:**
```
T1 = 1 / [γ_qp(T) + γ_0*(2*n_eff + 1)]
```

**Physical meaning:** Time for excited state population to decay to 1/e of initial value.

**Test case: T = 0.020 K, ω = 5e10 rad/s, γ_0 = 1e5 s^-1**

Step-by-step calculation:
1. From above: γ_qp ≈ O(1e3 - 1e5) s^-1
2. From above: n_eff ≈ 9.76e-5
3. Calculate total rate:
   ```
   γ_total = γ_qp + γ_0*(2*n_eff + 1)
          ≈ 1e4 + 1e5*(2*9.76e-5 + 1)
          ≈ 1e4 + 1e5*1.0002
          ≈ 1.1e5 s^-1
   ```

4. Calculate T1:
   ```
   T1 = 1 / 1.1e5 ≈ 9.1 μs
   ```

**Physical interpretation:** T1 ~ 10 μs is typical for superconducting qubits. The γ_0 term dominates, with small corrections from thermal photons and quasiparticles.

## 7. Pure Dephasing Time (T_psi / T_φ)

**Formula:**
```
T_psi = 1 / (γ_φ_base + γ_φ_qp)

where:
γ_φ_qp = 4π*(ω_p²/ω_ge) * sqrt(x_A_qp/N_e)
x_A_qp = exp(-Δ/(k*T))
N_e = (1/subgap_transparency) * (g_t/(2*g_k))
g_t = 1/R_n
g_k = e²/h
```

**Physical meaning:** Pure dephasing rate (no energy exchange, just phase randomization).

**Test case: T = 0.020 K, ω = 5e10 rad/s**

Step-by-step calculation:
1. Calculate conductances:
   ```
   g_t = 1/R_n = 1/5000 = 2e-4 S
   g_k = e²/h = (1.602e-19)² / 6.626e-34 = 3.874e-5 S
   ```

2. Calculate number of channels:
   ```
   N_e = (1/0.01) * (2e-4 / (2*3.874e-5))
      = 100 * 2.58 = 258
   ```

3. Calculate Andreev quasiparticle occupation:
   ```
   x_A_qp = exp(-Δ/(k*T)) = exp(-63.88) ≈ 1.47e-28
   ```

4. Calculate γ_φ_qp:
   ```
   γ_φ_qp = 4π * (7.27e11)² / 5e10 * sqrt(1.47e-28 / 258)
         ≈ 4π * 1.06e13 * sqrt(5.7e-31)
         ≈ 1.33e14 * 2.39e-16
         ≈ 3.18e-2 s^-1
   ```

5. Calculate T_psi:
   ```
   T_psi = 1 / (1e3 + 3.18e-2) ≈ 1 / 1000 = 1 ms
   ```

**Physical interpretation:** Pure dephasing time ~ 1 ms, dominated by the base dephasing rate. This is much longer than T1.

## 8. T2 (Total Dephasing Time)

**Formula:**
```
T2 = 1 / (1/(2*T1) + 1/T_psi)
```

**Physical meaning:** Time for phase coherence to decay. Limited by both energy relaxation and pure dephasing.

**Test case: Using T1 = 9.1 μs and T_psi = 1 ms from above**

Step-by-step calculation:
1. Calculate rates:
   ```
   1/(2*T1) = 1/(2*9.1e-6) = 5.49e4 s^-1
   1/T_psi = 1/1e-3 = 1e3 s^-1
   ```

2. Calculate total dephasing rate:
   ```
   γ_2 = 5.49e4 + 1e3 = 5.59e4 s^-1
   ```

3. Calculate T2:
   ```
   T2 = 1 / 5.59e4 = 17.9 μs
   ```

**Verification of physical constraint:**
```
T2 = 17.9 μs < 2*T1 = 18.2 μs ✓
```

**Physical interpretation:** T2 ~ 18 μs ≈ 2*T1, indicating pure dephasing is relatively weak. The T1 process is the dominant decoherence mechanism.

## 9. Gate Fidelity (F_N)

**Formula:**
```
F_N = 1 - (d*N*gate_length)/(2*(d+1)) * (1/T1 + 1/T_psi)

where d = 2^N (Hilbert space dimension)
```

**Physical meaning:** Probability that gate succeeds without decoherence errors.

**Test case: Two-qubit gate (N=2), gate_length = 40 ns, T1 = 9.1 μs, T_psi = 1 ms**

Step-by-step calculation:
1. Calculate Hilbert space dimension:
   ```
   d = 2^2 = 4
   ```

2. Calculate geometric factor:
   ```
   d*N / (2*(d+1)) = 4*2 / (2*5) = 8/10 = 0.8
   ```

3. Calculate decoherence rate:
   ```
   1/T1 + 1/T_psi = 1/(9.1e-6) + 1/(1e-3)
                  = 1.099e5 + 1e3
                  = 1.109e5 s^-1
   ```

4. Calculate error:
   ```
   error = 0.8 * 40e-9 * 1.109e5
        = 0.8 * 4.436e-3
        = 3.549e-3
   ```

5. Calculate fidelity:
   ```
   F_N = 1 - 0.003549 = 0.9965 = 99.65%
   ```

**Physical interpretation:** Two-qubit gate with 40 ns duration achieves 99.65% fidelity, corresponding to 0.35% error rate. This is near the fault-tolerance threshold for surface codes (~1%).

**Comparison with single-qubit gate:**
```
For N=1: geometric factor = 2*1/(2*3) = 1/3 ≈ 0.333
Error (N=1) = 0.333 * 40e-9 * 1.109e5 = 1.48e-3
F_1 = 99.85%
```

Single-qubit gates have higher fidelity as expected.

## Key Physical Insights

1. **Exponential suppression:** Quasiparticle effects are exponentially suppressed at low T
2. **T2 ≤ 2*T1 constraint:** Always satisfied by construction
3. **Thermal occupation:** At typical dilution fridge temperatures (~10-50 mK) and GHz frequencies, thermal photon occupation is negligible
4. **Gate fidelity scaling:** Longer gates and multi-qubit operations accumulate more decoherence
5. **Dominant mechanisms:** For typical parameters, the base relaxation rate γ_0 dominates over quasiparticle effects

## Running the Tests

```bash
pytest test_default_builder.py -v
```

Expected output: All tests should pass, verifying that the implementation matches these manual calculations within numerical tolerance.
