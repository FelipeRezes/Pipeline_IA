# Pipeline_IA — Scientific Ablation & SOTA Benchmark Results (A1, A2, A3, A4, A5, A6, A7)

| Experiment | Description | Key Metric |
| :--- | :--- | :--- |
| **A1** | Comparison of 5 kinematic filters across straight, evasive turn, and blind-zone outage regimes. | Adaptive EKF Maneuver RMSE = 0.3418 m |
| **A2** | Stabilization of fading factor lambda_k with circular heading smoothing and NIS persistence under increasing noise. | Completed |
| **A3** | Ground-plane projection bias reduction via 3D perspective bounding-box contact-point correction. | Bias Reduction = 80.31% |
| **A4** | Cross-camera vehicle association and forensic route backtracking across 4 fusion modes. | Hybrid IDF1 = 1.0 |
| **A5** | HMM/Viterbi map-matching vs Euclidean snapping, v_crit corner pruning, and CICC blockade recommendation. | HMM Accuracy = 1.0 |
| **A6** | State-of-the-art comparison: EKF/IMM vs SIR Particle Filter, LSTM, GRU (5 seeds), and Road-Graph Route Prediction vs HMM/Viterbi and Temporal GNN (T-GNN). | Adaptive EKF Blind RMSE = 7.156 m; Proposed Tree Top-1 = 0.75 |
| **A7** | City-scale multi-camera benchmark scaling from 4 to 24 non-overlapping cameras and 20 to 150 concurrent vehicles across 4 association modes. | 24 Cams / 150 Veh Hybrid IDF1 = 1.0 |
