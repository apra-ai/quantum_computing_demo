## RSA classical vs quantum demo

This workspace contains two small presentation-oriented Python demos:

- `brute_force_demo.py` for classical trial-division factorization on a CPU
- `quanten_demo.py` for the current Qiskit environment check around Shor factoring

The workspace now uses a local virtual environment in `.venv` with the current packages:

- `qiskit==2.4.1`
- `qiskit-aer==0.17.2`
- `qiskit-ibm-runtime==0.46.1`

### Run the classical demo

```powershell
.venv\Scripts\python.exe brute_force_demo.py
```

### Run the quantum demo

Use the local simulator:

```powershell
.venv\Scripts\python.exe quanten_demo.py
```

Use an IBM Quantum backend:

```powershell
$env:IBM_QUANTUM_TOKEN="<your-token>"
.venv\Scripts\python.exe quanten_demo.py --backend-name ibm_brisbane
```

### Qiskit note

The currently installed Qiskit stack does not expose a built-in high-level `Shor` class anymore.

That means `quanten_demo.py` can verify the selected simulator or IBM backend, but it cannot factor `N = 15` through `qiskit.algorithms.Shor` because that API no longer exists in current Qiskit.

If you need an actual Shor factorization demo, you need one of these two paths:

- pin an older Qiskit stack that still shipped the high-level Shor implementation
- replace the file with a manual order-finding circuit demo for `N = 15`
