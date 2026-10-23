# Can AI Replace Camera Color Filters? — ColorChecker experiment

Reproducible **simulation** code for the ColorChecker portion of the YouTube video. The scripts were taken from the original experiment archives.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python compare_models.py
python smoothed_lut_experiment.py
```

`monochrome_color_simulation.py` is the initial baseline. The comparison scripts use 24 Macbeth patches for training and two independent 130- and 140-patch test sets. The `.mat` spectral data files are included.

**Models:** affine mapping, degree-3 polynomial, PCHIP LUT, regularized MLP, and smoothed PCHIP LUT.

**Important limitations:** the monochrome sensitivity is an illustrative model, not measured sensor quantum efficiency; D65/CMFs are from the original experiment; a single monochrome value cannot uniquely recover RGB. The original PCHIP LUT looks more colorful in some displays, but has worse mean ΔE00 on the 130-patch set than other methods. The DDColor Colab notebook is **not** included here, since it is already maintained separately.

To reproduce the full AI segment, share your existing DDColor Colab notebook separately and credit the upstream DDColor repository.
