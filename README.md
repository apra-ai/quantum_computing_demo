# Classical Brute Force vs. Grover Demo

This project is a small educational demo for presentations about quantum attacks on classical search problems. It compares:

- classical CPU brute-force search with linear scaling $O(N)$
- Grover's algorithm with quadratic speedup $O(\sqrt{N})$

The demo searches for the same hidden target bitstring in both modes.

## Important framing

This is **not** a wall-clock speed demo.

Real IBM Quantum hardware is usually slower than a laptop for these tiny examples because of queueing, transpilation, calibration overhead, measurement overhead, and hardware noise. The meaningful comparison here is the algorithmic scaling:

- classical search: $O(N)$ checked candidates
- Grover search: $O(\sqrt{N})$ oracle rounds

## Project structure

```text
src/
	classical_bruteforce.py
	grover.py
	ibm_backend.py
	visualization.py
	main.py
README.md
requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your IBM token only if you want to use real hardware:

```bash
export IBM_QUANTUM_TOKEN=your_token_here
```

PowerShell equivalent:

```powershell
$env:IBM_QUANTUM_TOKEN="your_token_here"
```

## Run examples

Local simulator:

```bash
python src/main.py --mode simulator --n-qubits 3 --target 101 --shots 1024
```

IBM Quantum hardware:

```bash
python src/main.py --mode ibm --n-qubits 3 --target 101 --shots 1024
```

If IBM credentials are missing or the IBM runtime package is unavailable, `--mode ibm` automatically falls back to the local simulator and prints the reason.

## Command-line options

- `--mode simulator|ibm`
- `--n-qubits 2|3|4|5`
- `--target bitstring`
- `--shots 1024`

Valid search spaces are:

- `n = 2` for $N = 4$
- `n = 3` for $N = 8$
- `n = 4` for $N = 16$
- `n = 5` for $N = 32$ (optional larger example)

## What the demo reports

For each run, the program prints:

- classical number of checked candidates
- Grover iteration count using $\left\lfloor \frac{\pi}{4}\sqrt{2^n} \right\rfloor$
- measured result distribution
- success probability for the target bitstring

It also saves two plots in `outputs/`:

- a bar chart of measured quantum states
- a comparison bar chart of brute-force checks vs. Grover iterations
