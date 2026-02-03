import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="HeteroSim-X", layout="wide")

st.title("⚙️ HeteroSim-X: Heterogeneous Computing Simulator & Profiler")
st.caption("FI 9070 – Heterogeneous Computing | Anna University (R2021)")

st.markdown("""
**HeteroSim-X** simulates how modern heterogeneous systems dynamically distribute workloads
between **multi-core CPUs (task parallelism)** and **GPU-style data parallel execution**.
""")

# Sidebar
st.sidebar.header("Simulation Parameters")
workload_size = st.sidebar.slider("Workload Size", 100_000, 5_000_000, 1_000_000, step=100_000)
cpu_threads = st.sidebar.slider("CPU Threads", 1, 16, 4)
auto_ratio = st.sidebar.slider("Heterogeneous CPU Load (%)", 10, 90, 50)

# Workload
data = np.arange(workload_size, dtype=np.float64)

def cpu_task(chunk):
    s = 0
    for x in chunk:
        s += x * x
    return s

def gpu_task(arr):
    return np.sum(arr * arr)

# Timing containers
timeline = []

# CPU Execution
t0 = time.perf_counter()
chunks = np.array_split(data, cpu_threads)
cpu_sum = 0
with ThreadPoolExecutor(max_workers=cpu_threads) as executor:
    futures = []
    for i, chunk in enumerate(chunks):
        start = time.perf_counter()
        futures.append(executor.submit(cpu_task, chunk))
        timeline.append(("CPU Task", i, start))
    for f in as_completed(futures):
        cpu_sum += f.result()
cpu_time = time.perf_counter() - t0

# GPU-style Execution
t1 = time.perf_counter()
gpu_sum = gpu_task(data)
gpu_time = time.perf_counter() - t1
timeline.append(("GPU Kernel", 0, t1))

# Heterogeneous Execution
split = int((auto_ratio / 100) * workload_size)
cpu_data = data[:split]
gpu_data = data[split:]

t2 = time.perf_counter()
hetero_sum = 0

chunks = np.array_split(cpu_data, cpu_threads)
with ThreadPoolExecutor(max_workers=cpu_threads) as executor:
    futures = []
    for i, chunk in enumerate(chunks):
        start = time.perf_counter()
        futures.append(executor.submit(cpu_task, chunk))
        timeline.append(("Hetero CPU", i, start))
    for f in as_completed(futures):
        hetero_sum += f.result()

hetero_sum += gpu_task(gpu_data)
hetero_time = time.perf_counter() - t2

# Metrics
speedup_gpu = cpu_time / gpu_time
speedup_hetero = cpu_time / hetero_time

# Results Display
st.subheader("📊 Performance Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("CPU Time (s)", f"{cpu_time:.4f}")
c2.metric("GPU-style Time (s)", f"{gpu_time:.4f}")
c3.metric("Heterogeneous Time (s)", f"{hetero_time:.4f}")

st.subheader("🚀 Speedup Analysis")
c4, c5 = st.columns(2)
c4.metric("GPU Speedup", f"{speedup_gpu:.2f}×")
c5.metric("Heterogeneous Speedup", f"{speedup_hetero:.2f}×")

# Timeline Visualization
st.subheader("🕒 Execution Timeline")

labels = ["CPU", "GPU-style", "Heterogeneous"]
times = [cpu_time, gpu_time, hetero_time]

fig, ax = plt.subplots()
ax.bar(labels, times)
ax.set_ylabel("Execution Time (seconds)")
ax.set_title("Execution Timeline Comparison")

st.pyplot(fig)

# Concepts
st.subheader("🧠 Concepts Demonstrated")
st.markdown("""
- **ILP & TLP:** Multi-threaded CPU execution  
- **Data Parallelism:** Vectorized GPU-style kernel  
- **OpenMP Analogy:** Thread pools & work sharing  
- **OpenCL Analogy:** Kernel-based data execution  
- **Heterogeneous Scheduling:** Dynamic CPU–GPU workload split  
- **Profiling:** Runtime measurement & speedup analysis  
""")

st.caption("This simulator abstracts real heterogeneous hardware behavior for academic analysis.")
