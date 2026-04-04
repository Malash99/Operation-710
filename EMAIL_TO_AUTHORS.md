# Email to DINO-VO Authors

**To**: Yassine Azhari, Dongwon Shim
**Subject**: Implementation questions regarding DINO-VO (arXiv:2507.13145) — SVD backward through Essential matrix

---

Dear Dr. Azhari and Prof. Shim,

I am reimplementing DINO-VO from your paper "DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model" (IEEE RA-L, July 2025). I have successfully implemented all pipeline components (Phases 1-9) and am training on TartanAir as described in Section IV-A.

The matching loss converges well during epochs 1-4 (from ~6.3 to ~4.5), but when the pose loss is introduced in epoch 5, it remains flat at ~540 across multiple epochs and never decreases. I have traced the root cause to a gradient flow issue and have a few specific questions.

## Question 1: Differentiating through the Essential matrix decomposition

The Essential matrix always has degenerate singular values (sigma, sigma, 0). When differentiating through the SVD decomposition of E to recover (R, t), the standard PyTorch SVD backward involves terms of the form 1/(s_i^2 - s_j^2), which become undefined when s_1 = s_2.

In Section III-D, you reference Brachmann et al. [5] (DSAC) for the SVD decomposition. Could you clarify how you handle the SVD backward pass in this degenerate case? Specifically:

- Do you use DSAC-style gradient clamping (bounding the 1/(s_i^2 - s_j^2) terms)?
- Do you implement a custom autograd function for the SVD?
- Or do you use a different approach entirely?

This is the critical blocker in our reimplementation — without proper gradient flow through the Essential matrix decomposition, the pose loss cannot backpropagate to the matching network.

## Question 2: Essential matrix projection

After solving the weighted 8-point algorithm (Eq. 11) for E, do you project E onto the Essential manifold (i.e., enforce singular values to (1, 1, 0) via SVD) before decomposing it into (R, t)? Or do you decompose the raw E from the null space directly?

This matters because the projection step introduces an additional SVD with guaranteed degenerate singular values, compounding the backward pass issue from Question 1.

## Question 3: TartanAir training resolution

The paper specifies image resolution 476 x 742 for EuRoC (Section IV-A), but does not state the resolution used for TartanAir training. Since TartanAir's native 640 x 480 is not divisible by 14 (DINOv2 patch size), what resolution do you resize TartanAir images to?

## Question 4: TartanAir training environments

Which TartanAir environments and difficulty levels did you use for training? All available environments, or a specific subset?

---

Thank you very much for your time. Your paper presents an elegant approach to visual odometry and I am keen to get the reimplementation working correctly.

Best regards,
[Your Name]
