<h1 align="center">lelab-omx</h1>

<p align="center">
  <b>Robotis OMX support for LeLab and LeRobot workflows.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](pyproject.toml)

</div>

`lelab-omx` extends the browser-based LeLab workflow with Robotis OMX-AI
hardware support, no-code data collection, and bimanual teleoperation. The
project also targets an open simulation workflow that generates demonstrations
with reinforcement learning and exports them as LeRobotDataset data for
imitation learning.

The current base includes the LeLab web UI and backend together with the OMX-AI
and dual-arm work developed in the first-stage repository. It supports the
calibration, teleoperation, recording, training, replay, and inference workflow
used by LeRobot. The simulation-based demonstration generator is planned beyond
this imported baseline.

## Development setup

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and Node.js/npm.

```bash
git clone https://github.com/avatar196kc/lelab-omx.git
cd lelab-omx

uv sync --extra dev --extra test
npm ci --prefix frontend
npm run build --prefix frontend
uv run lelab --dev
```

The backend uses port `8000`; the Vite development server uses port `8080`.

## Technical references

- [LeLab documentation](https://huggingface.co/docs/lerobot/lelab)
- [Hugging Face LeLab](https://github.com/huggingface/leLab)
- [First-stage LeLab repository](https://github.com/orocapangyo/leLab)
- [First-stage installation notes](https://github.com/orocapangyo/leLab/blob/main/study/install.md)

## Upstream and license

The application base was imported from
[`orocapangyo/leLab`](https://github.com/orocapangyo/leLab) at commit
[`6fc977e`](https://github.com/orocapangyo/leLab/commit/6fc977e), which is based
on Hugging Face LeLab. Source copyright and Apache-2.0 notices are retained.
See [NOTICE](NOTICE) for provenance details.

This repository is licensed under the [Apache License 2.0](LICENSE).
