# Automated Sim-to-Real Deployment Framework for Reinforcement Learning Control on Resource-Constrained Microcontrollers

**Authors:** Nidharshan V., Sulochana Sooriyaarachchi  
**Affiliation:** Department of Computer Science & Engineering, University of Moratuwa, Sri Lanka  
**Keywords:** Sim-to-Real Transfer, Reinforcement Learning, Microcontroller Deployment, TinyML, Dynamic Self-Righting

---

## Abstract

Deploying deep reinforcement learning (RL) policies onto resource-constrained microcontrollers (MCUs) such as the ESP32 (<520 kB RAM, bare-metal) traditionally requires error-prone, manual firmware translation of network weights, inference kernels, and peripheral drivers. We present **`simtoreal`**, an open-source framework that automates end-to-end policy deployment from simulation to embedded hardware. The framework features: (1) a dual-target code generator that exports trained Stable-Baselines3 and PyTorch actor networks into either zero-allocation, heap-free C headers or precompiled MicroPython bytecode (`.mpy`); and (2) a hardware abstraction synthesizer that generates the complete deterministic on-board control loop from a declarative YAML configuration.

We benchmark the pipeline on **_Robo1_**, a 94.8 g, 2-DOF dynamic self-righting robot paired with a CAD-grounded MuJoCo digital twin. Across 4,609 evaluation inputs, the generated C and MicroPython policies achieve single-precision numerical parity with PyTorch (maximum absolute errors of $2.9 \times 10^{-6}$ and $1.2 \times 10^{-6}$, respectively). In 1,000 closed-loop simulation runs under nominal and domain-randomized physics, the synthesized C controller replicates PyTorch trajectories identically, attaining a 97.2% self-righting success rate. Microcontroller profiling on an ESP32 reveals that while sensor and actuator I/O execute in 4.2 ms (well within the 20 ms / 50 Hz deadline), interpreted MicroPython inference takes ~155.2 ms (limiting the interpreted loop rate to 5.4 Hz), demonstrating that compiled zero-heap C headers are essential for hard real-time dynamic robotic control.
