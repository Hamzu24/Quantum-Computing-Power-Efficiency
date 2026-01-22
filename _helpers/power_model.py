"""
Relative cryogenic cost model for superconducting quantum computers.

This module calculates relative cryogenic cooling costs based on the T² scaling
of dilution refrigerator cooling power.

Physics basis:
    Dilution refrigerators cool by exploiting the enthalpy of mixing between
    ³He and ⁴He. The cooling power at the mixing chamber is:

        Q̇ = 84 × ṅ₃ × T²  [W]

    where ṅ₃ is the ³He circulation rate (mol/s) and T is temperature (K).

Source:
    Pobell, F. "Matter and Methods at Low Temperatures"
    Springer, 3rd edition (2007), Chapter 7

This T² scaling means operating at higher temperatures requires proportionally
less cooling power, enabling direct comparison of cryogenic overhead across
different operating points.
"""

from typing import List, Dict
from dataclasses import dataclass


def relative_cryo_cost(T_operating_mK: float, T_reference_mK: float = 15.0) -> float:
    """
    Relative cryogenic cooling cost based on T² scaling.

    From: Pobell, "Matter and Methods at Low Temperatures", Ch. 7

    Args:
        T_operating_mK: Operating temperature in milliKelvin
        T_reference_mK: Reference temperature (default: 15 mK)

    Returns:
        Relative cost factor (< 1 means cheaper than reference)

    Example:
        >>> relative_cryo_cost(50, 15)  # 50 mK vs 15 mK
        0.09  # ~11x reduction in cooling cost
    """
    return (T_reference_mK / T_operating_mK) ** 2


def cryo_cost_reduction_factor(T_operating_mK: float, T_reference_mK: float = 15.0) -> float:
    """
    How many times cheaper is cooling at T_operating vs T_reference.

    Args:
        T_operating_mK: Operating temperature in milliKelvin
        T_reference_mK: Reference temperature (default: 15 mK)

    Returns:
        Reduction factor (> 1 means cheaper than reference)

    Example:
        >>> cryo_cost_reduction_factor(50, 15)
        11.1  # Cooling is ~11x cheaper at 50 mK than 15 mK
    """
    return (T_operating_mK / T_reference_mK) ** 2


@dataclass
class PerformanceEfficiencyPoint:
    """A single point on the performance-efficiency tradeoff curve."""
    temperature_mK: float
    relative_cryo_cost: float
    cryo_reduction_factor: float
    performance_metric: float = None
    metric_name: str = None


def temperature_sweep(
    T_range_mK: List[float],
    T_reference_mK: float = 15.0,
    performance_values: List[float] = None,
    metric_name: str = None
) -> List[PerformanceEfficiencyPoint]:
    """
    Calculate relative cryogenic costs across a temperature range.

    Args:
        T_range_mK: List of temperatures in milliKelvin
        T_reference_mK: Reference temperature for relative comparison
        performance_values: Optional list of performance metrics (same length as T_range_mK)
        metric_name: Name of the performance metric (e.g., "quantum_volume")

    Returns:
        List of PerformanceEfficiencyPoint objects

    Example:
        >>> temps = [15, 20, 30, 50, 100]
        >>> results = temperature_sweep(temps)
        >>> for r in results:
        ...     print(f"{r.temperature_mK} mK: {r.cryo_reduction_factor:.1f}x cheaper")
    """
    results = []

    for i, T in enumerate(T_range_mK):
        perf = performance_values[i] if performance_values else None

        point = PerformanceEfficiencyPoint(
            temperature_mK=T,
            relative_cryo_cost=relative_cryo_cost(T, T_reference_mK),
            cryo_reduction_factor=cryo_cost_reduction_factor(T, T_reference_mK),
            performance_metric=perf,
            metric_name=metric_name
        )
        results.append(point)

    return results


def print_cryo_cost_table(T_range_mK: List[float] = None, T_reference_mK: float = 15.0):
    """
    Print a formatted table of relative cryogenic costs.

    Args:
        T_range_mK: List of temperatures (default: [15, 20, 30, 50, 100])
        T_reference_mK: Reference temperature
    """
    if T_range_mK is None:
        T_range_mK = [15, 20, 30, 50, 100]

    print(f"\nRelative Cryogenic Cost (reference: {T_reference_mK} mK)")
    print("=" * 50)
    print(f"{'Temperature':<15} {'Relative Cost':<15} {'Reduction':<15}")
    print("-" * 50)

    for T in T_range_mK:
        rel_cost = relative_cryo_cost(T, T_reference_mK)
        reduction = cryo_cost_reduction_factor(T, T_reference_mK)
        print(f"{T:>8.0f} mK     {rel_cost:>10.3f}x      {reduction:>10.1f}x cheaper")

    print("=" * 50)
    print("Based on T² scaling of dilution refrigerator cooling power")
    print("Source: Pobell, 'Matter and Methods at Low Temperatures' (2007)")


def format_for_paper(
    T_range_mK: List[float],
    performance_values: List[float],
    metric_name: str = "Quantum Volume",
    T_reference_mK: float = 15.0
) -> str:
    """
    Generate a formatted table suitable for a paper.

    Args:
        T_range_mK: List of temperatures
        performance_values: Corresponding performance metric values
        metric_name: Name of the metric
        T_reference_mK: Reference temperature

    Returns:
        Formatted string table
    """
    lines = []
    lines.append(f"Temperature (mK) | Relative Cryo Cost | {metric_name}")
    lines.append("-" * 60)

    ref_perf = performance_values[0] if performance_values else None

    for i, T in enumerate(T_range_mK):
        rel_cost = relative_cryo_cost(T, T_reference_mK)
        perf = performance_values[i] if performance_values else "N/A"

        if ref_perf and performance_values:
            perf_rel = performance_values[i] / ref_perf
            lines.append(f"{T:>15.0f}  | {rel_cost:>18.3f} | {perf} ({perf_rel:.2%} of ref)")
        else:
            lines.append(f"{T:>15.0f}  | {rel_cost:>18.3f} | {perf}")

    return "\n".join(lines)
