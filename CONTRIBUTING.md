# Contributing

For a bug report, include the command, Python and package versions, device, expected behavior, and complete error message. Include a small input that you are allowed to share when it is needed to reproduce the issue.

Keep pull requests focused and describe any change to model behavior, data preprocessing, sampling, or metrics. Preserve existing checkpoint compatibility unless the change explicitly requires a new model version.

Install the pinned dependencies and run the CPU checks before submitting a change:

```bash
python -m pip install -r requirements.txt
python scripts/check_installation.py
```

The checks cover the model and latent objective, fixed partitions and sampling, metric/depth validation, and image-directory inference. For changes to a command-line interface, also run that command with `--help` and update its README example.
