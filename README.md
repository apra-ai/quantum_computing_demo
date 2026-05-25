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

Create a `.env` file in the project root if you want to use real IBM hardware:

```env
IBM_QUANTUM_TOKEN=your_token_here
IBM_QUANTUM_INSTANCE=your_instance_or_crn_here
IBM_QUANTUM_BACKEND=ibm_brisbane
```

The app loads `.env` automatically at startup, so you do not need to set `export` or `$env:` manually.

`IBM_QUANTUM_INSTANCE` is optional in code, but in practice it is often needed when your IBM account has no default instance or when you want to force a specific one.

`IBM_QUANTUM_BACKEND` is also optional. If you set it, the demo will try to use exactly that IBM hardware backend. If you leave it unset, the code automatically picks a hardware backend with enough qubits and a low queue.

You can copy the template file first:

```bash
cp .env.example .env
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

The demo also saves a structured JSON execution log in `outputs/logs/`. Each log contains:

- the request you sent to the quantum execution layer
- the selected backend and transpiled circuit summary
- the backend response counts and success probability
- the fallback reason when IBM mode had to use the local simulator instead

## Experiment runner

The project also includes a separate batch runner in `src/experiment_runner.py`. It is independent from `src/main.py` and is meant for collecting timing data across many qubit counts.

The runner can execute both CPU brute-force and Grover runs for a configurable list of bit sizes, save CSV and JSON summaries, and generate a combined runtime scaling plot.

Example:

```bash
python src/experiment_runner.py --mode ibm --backend ibm_kingston --shots 1 --max-qubits 36 --target-mode last
```

Resume an existing batch and only run missing sizes:

```bash
python src/experiment_runner.py --mode ibm --shots 1 --max-qubits 36 --resume true
```

Run only a specific size and merge it into the existing CSV and JSON:

```bash
python src/experiment_runner.py --mode ibm --shots 1 --qubits 36 --resume true
```

By default, the runner writes its outputs into a dedicated folder under `outputs/evaluation_runtimes/`.

Important notes:

- Small `n_qubits` values can run on real IBM hardware.
- Large `n_qubits` values may fail or become impractical because of circuit depth, transpilation limits, queueing, shot costs, and hardware constraints.
- The runner catches per-size execution errors and continues unless `--stop-on-error true` is set.
- With `--resume true`, the runner loads the existing CSV, skips already completed qubit sizes, updates the CSV and JSON, and rebuilds the runtime plot from the updated CSV.
