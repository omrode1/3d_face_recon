# FLAME 3DMM Fitting & Export Pipeline: Bug Report and Fixes

## 1. FLAME Model Not Loading (Dummy Model Used)
- **Symptom:** Output: 'Creating a simplified dummy model instead', mesh has only 18 landmarks, OBJ/UVs do not match face.
- **Root Cause:** Numpy/chumpy compatibility issues (`np.bool`, `np.object`, etc. missing in recent numpy).
- **Fix:** Add numpy compatibility patch at the top of all scripts that load the FLAME model:
  ```python
  import numpy as np
  if not hasattr(np, 'bool'): np.bool = bool
  if not hasattr(np, 'object'): np.object = object
  if not hasattr(np, 'str'): np.str = str
  if not hasattr(np, 'int'): np.int = int
  if not hasattr(np, 'float'): np.float = float
  if not hasattr(np, 'complex'): np.complex = complex
  if not hasattr(np, 'unicode'): np.unicode = str
  ```

## 2. Camera Parameters Not Optimized (Mesh Not Aligned)
- **Symptom:** Camera parameters remain at initial values (e.g., `[[5. 0. 0. 0.]]`), mesh appears in top left, loss does not decrease.
- **Root Cause:** Loss computation broke the computation graph by converting tensors to numpy before loss calculation.
- **Fix:** Keep all variables as PyTorch tensors until after the loss and backward pass. Only convert to numpy for visualization or saving, never for loss computation.

## 3. In-place Assignment Error for cam_params
- **Symptom:** RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
- **Root Cause:** In-place assignment to a leaf tensor with requires_grad=True.
- **Fix:** Initialize cam_params with the correct values at creation, e.g.:
  ```python
  cam_params = torch.tensor([[5.0, 0.0, 0.0, 0.0]], device=self.device, requires_grad=True)
  ```

## 4. Landmark Count Mismatch
- **Symptom:** Warnings about landmark count mismatch (e.g., model has 18, GT has 68), poor fitting.
- **Root Cause:** Dummy model loaded (see bug #1), or landmark mapping not matching detected landmarks.
- **Fix:** Ensure real FLAME model loads and that landmark indices match the detected landmark convention (usually 68-point for MediaPipe/dlib).

## 5. cam_params.grad is None
- **Symptom:** Printout shows cam_params.grad: None, so optimizer cannot update camera parameters.
- **Root Cause:** Loss computation broke the computation graph (see bug #2).
- **Fix:** Same as bug #2—keep all variables as PyTorch tensors until after the loss and backward pass.

---

**Summary:**
- Always keep the computation graph intact for optimization.
- Never convert tensors to numpy before loss/backward.
- Patch numpy for legacy FLAME/chumpy compatibility.
- Initialize all learnable parameters with correct values at creation.
- Ensure landmark mapping matches your detector and model. 