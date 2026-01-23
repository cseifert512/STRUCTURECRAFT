# app/components/design_card.py
"""
Design Report Card component - displays pass/fail summary for all design checks.
"""

import streamlit as st
from typing import Dict, Any


def render_design_card(
    deflection_check: Dict[str, Any],
    utilization_check: Dict[str, Any],
    metrics: Dict[str, Any],
    shipping_limit: float = 6.0  # meters
) -> bool:
    """
    Render a pass/fail report card for the design.
    
    Args:
        deflection_check: Result from compute_deflection_check()
        utilization_check: Result from compute_utilization()
        metrics: Design metrics dictionary
        shipping_limit: Maximum shippable member length (m)
    
    Returns:
        bool: True if all checks pass
    """
    st.markdown("### 📋 Design Report Card")
    
    # Collect all checks
    checks = []
    
    # 1. Deflection check
    defl_ratio = deflection_check['checks']['L/240']['ratio']
    checks.append({
        'name': 'Deflection (L/240)',
        'value': f"{defl_ratio:.0%}",
        'pass': deflection_check['overall_pass'],
        'detail': f"{deflection_check['checks']['L/240']['actual_mm']:.1f} mm / {deflection_check['checks']['L/240']['limit_mm']:.1f} mm limit"
    })
    
    # 2. Member strength check
    worst_util = utilization_check['worst_util']
    checks.append({
        'name': 'Member Strength',
        'value': f"{worst_util*100:.0f}%",
        'pass': utilization_check['all_pass'],
        'detail': f"Worst: Bar {utilization_check['worst_bar']} ({utilization_check['n_pass']}/{utilization_check['n_pass'] + utilization_check['n_fail']} pass)"
    })
    
    # 3. Shipping limit check
    max_len = metrics.get('max_member_length', 0)
    checks.append({
        'name': f'Shipping (<{shipping_limit}m)',
        'value': f"{max_len:.2f} m",
        'pass': max_len <= shipping_limit,
        'detail': f"Longest member must fit in standard transport"
    })
    
    # Display each check as a row
    for check in checks:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{check['name']}**")
        with col2:
            st.write(check['value'])
        with col3:
            if check['pass']:
                st.markdown("✅ **PASS**")
            else:
                st.markdown("❌ **FAIL**")
    
    st.divider()
    
    # Overall status
    all_pass = all(c['pass'] for c in checks)
    
    if all_pass:
        st.success("🎉 **ALL CHECKS PASS** — Design is acceptable")
    else:
        failing = [c['name'] for c in checks if not c['pass']]
        st.error(f"⚠️ **DESIGN NEEDS REVISION** — Failing: {', '.join(failing)}")
    
    return all_pass


def render_compact_checks(
    deflection_check: Dict[str, Any],
    utilization_check: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    """
    Render compact inline check indicators (for the metrics panel).
    
    Args:
        deflection_check: Result from compute_deflection_check()
        utilization_check: Result from compute_utilization()
        metrics: Design metrics dictionary
    """
    # Deflection indicator
    defl_ratio = deflection_check['checks']['L/240']['ratio']
    if deflection_check['overall_pass']:
        st.success(f"✓ Deflection OK (L/240: {defl_ratio:.0%})")
    else:
        st.error(f"✗ Deflection FAIL (L/240: {defl_ratio:.0%})")
    
    # Utilization indicator
    util_pct = utilization_check['worst_util'] * 100
    if util_pct <= 80:
        st.success(f"✓ Utilization: {util_pct:.0f}% (Bar {utilization_check['worst_bar']})")
    elif util_pct <= 100:
        st.warning(f"⚠ Utilization: {util_pct:.0f}% (Bar {utilization_check['worst_bar']})")
    else:
        st.error(f"✗ Utilization: {util_pct:.0f}% OVERSTRESSED")
    
    st.caption(f"{utilization_check['n_pass']}/{utilization_check['n_pass'] + utilization_check['n_fail']} members pass")


