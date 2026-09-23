"""Explicit IPC broad phase shared by contact admission, forces and path proofs.

The pinned toolkit's platform-dependent default omitted overlapping primitives
in the Linux ARM source build. HashGrid is explicit and separately checked
against exhaustive AABB overlap and native BruteForce regression oracles.
"""

CONTACT_BROAD_PHASE_PROFILE = "ipctk-HashGrid-explicit-v1"


def contact_broad_phase():
    import ipctk

    # Broad phases retain their last build; each independent query owns its state.
    return ipctk.HashGrid()
