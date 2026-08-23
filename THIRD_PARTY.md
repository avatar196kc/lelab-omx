# Third-party components

Open source components used by `lelab-omx`. Versions are the ones actually
resolved in `uv.lock` and `frontend/package-lock.json`.

Components authored by this team are not listed here; see [NOTICE](NOTICE) for
provenance of the imported application base.

| #   | Component   | Version                  | License      | Repository                                      | Purpose                                                                                             |
| --- | ----------- | ------------------------ | ------------ | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1   | LeRobot     | 0.6.0 (git tag `v0.6.0`) | Apache-2.0   | https://github.com/huggingface/lerobot          | Robot learning core — teleoperation, recording, training, replay, inference. Imported as a library. |
| 2   | PyTorch     | 2.11.0                   | BSD-3-Clause | https://github.com/pytorch/pytorch              | Policy training and inference backend. Pulled in via LeRobot.                                       |
| 3   | FastAPI     | 0.139.2                  | MIT          | https://github.com/fastapi/fastapi              | Backend REST and WebSocket API.                                                                     |
| 4   | Uvicorn     | 0.51.0                   | BSD-3-Clause | https://github.com/encode/uvicorn               | ASGI server hosting the FastAPI app.                                                                |
| 5   | websockets  | 16.1.1                   | BSD-3-Clause | https://github.com/python-websockets/websockets | Real-time joint state streaming to the browser.                                                     |
| 6   | NumPy       | 2.2.6                    | BSD-3-Clause | https://github.com/numpy/numpy                  | Numerical handling of joint and trajectory arrays.                                                  |
| 7   | psutil      | 7.2.2                    | BSD-3-Clause | https://github.com/giampaolo/psutil             | Training job process and resource management.                                                       |
| 8   | React       | 18.3.1                   | MIT          | https://github.com/facebook/react               | Web UI rendering.                                                                                   |
| 9   | three.js    | 0.177.0                  | MIT          | https://github.com/mrdoob/three.js              | 3D rendering of the robot during teleoperation and episode review.                                  |
| 10  | urdf-loader | 0.12.7                   | Apache-2.0   | https://github.com/gkjohnson/urdf-loaders       | Loads the OMX URDF model into the browser 3D viewer.                                                |
| 11  | Vite        | 8.0.16                   | MIT          | https://github.com/vitejs/vite                  | Frontend build and development server.                                                              |

## Platform-specific

| Component | Version | License | Repository                            | Purpose                                                                                                                                 |
| --------- | ------- | ------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| pygrabber | 0.2     | MIT     | https://github.com/bunkahle/pygrabber | Windows only (`sys_platform == 'win32'`). Resolves DirectShow camera names so the frontend can match a camera to its browser device id. |

## 3D model assets

The OMX URDF and mesh files under `frontend/public/omx-urdf/` originate from the
first-stage repository and carry their own Apache-2.0 license file at
`frontend/public/omx-urdf/LICENSE`.

## Notes

- The full transitive dependency tree is visible in `uv.lock` and
  `frontend/package-lock.json`. This document lists direct dependencies that the
  project itself declares.
- `@react-three/fiber` 8.18.0 (MIT) is used as the React binding for three.js.
