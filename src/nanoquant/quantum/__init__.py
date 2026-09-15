"""Quantum proof of concept: polish one tile of stored sign bits with warm-started recursive QAOA.

``ising`` and ``rqaoa`` are numpy-only (no torch, no qiskit); ``sign_tile`` and ``pick_layer`` bridge to the
compression pipeline; ``ionq_backend`` is import-guarded on ``qiskit-ionq``.
"""
