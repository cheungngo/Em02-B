# Em02-B

Simulated evaluation of selected Em02-A-v6 formulas. Not a physical admission.

```bash
cd Em02-B
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python -m em02b.run_audit
```

The runner writes `results/tables/audit_manifest.json` and rejects `physical_identified`.

Current audit, source id `1eeba3536183`:

- \(c_R=0.625\), \(c_{47}=0.150451030065\)
- fixed-\(\epsilon\) mesh residual shrinks from \(1.3\times 10^{-4}\) to \(1.1\times 10^{-5}\)
- atomic \(5a\) Gauss \(4.851387\) versus Sobol \(4.845938\)
- scalar non-tip \(P_{5a}\) integral \(-0.00463365\)

Manuscripts: `reports/Em02-B-v1.tex`, `reports/Em02-B-scripts-v1.tex`.
