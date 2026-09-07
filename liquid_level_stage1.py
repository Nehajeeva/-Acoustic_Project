import numpy as np
import matplotlib.pyplot as plt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder


# ============================================================
# 1. PHYSICAL PARAMETERS
# ============================================================

TOTAL_HEIGHT = 1.0          # Total column height (m)
LIQUID_HEIGHT = 0.70        # Liquid height (m)
COLUMN_WIDTH = 0.10         # Column width (m)

AIR_HEIGHT = TOTAL_HEIGHT - LIQUID_HEIGHT

# Air properties
AIR_SPEED = 343.0           # m/s
AIR_DENSITY = 1.2           # kg/m^3

# Liquid properties (water-like)
LIQUID_SPEED = 1480.0       # m/s
LIQUID_DENSITY = 1000.0     # kg/m^3

# First test frequency
frequency = 500             # Hz


# ============================================================
# 2. CREATE COMPUTATIONAL GRID
# ============================================================

Nx = 300
Ny = 60

dx = TOTAL_HEIGHT / Nx
dy = COLUMN_WIDTH / Ny

kgrid = kWaveGrid(
    [Nx, Ny],
    [dx, dy]
)


# ============================================================
# 3. CREATE AIR + LIQUID MEDIUM
# ============================================================

sound_speed = np.ones((Nx, Ny)) * AIR_SPEED
density = np.ones((Nx, Ny)) * AIR_DENSITY

# Grid position where liquid starts
liquid_start = int(AIR_HEIGHT / dx)

# Change lower part of column to liquid
sound_speed[liquid_start:, :] = LIQUID_SPEED
density[liquid_start:, :] = LIQUID_DENSITY

medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density
)


# ============================================================
# 4. CREATE TIME ARRAY
# ============================================================

kgrid.makeTime(
    np.max(sound_speed),
    t_end=0.005
)

time = kgrid.t_array.flatten()


# ============================================================
# 5. CREATE SOURCE
# ============================================================

source = kSource()

source_position = 30
centre_y = Ny // 2

# Source mask
source_mask = np.zeros((Nx, Ny), dtype=bool)
source_mask[source_position, centre_y] = True

source.p_mask = source_mask


# ============================================================
# 6. CREATE 500 Hz SOURCE SIGNAL
# ============================================================

# Number of time points
Nt = len(time)

# Create a short 500 Hz sinusoidal signal
source_signal = np.sin(
    2 * np.pi * frequency * time
)

# Apply a smooth envelope so the signal starts and stops cleanly
envelope = np.zeros(Nt)

burst_time = 0.002       # 2 ms burst

burst_samples = np.where(time <= burst_time)[0]

if len(burst_samples) > 0:

    n = len(burst_samples)

    envelope[burst_samples] = (
        0.5
        - 0.5 * np.cos(
            2 * np.pi * np.arange(n) / (n - 1)
        )
    )

source_signal = source_signal * envelope


# k-Wave expects:
# (number of source points, number of time steps)

source.p = source_signal.reshape(1, -1)


# ============================================================
# 7. CREATE SENSOR
# ============================================================

sensor = kSensor()

sensor_mask = np.zeros((Nx, Ny), dtype=bool)

sensor_mask[source_position, centre_y] = True

sensor.mask = sensor_mask


# ============================================================
# 8. RUN SIMULATION
# ============================================================

print()
print("======================================")
print("Starting k-Wave simulation")
print("Frequency:", frequency, "Hz")
print("======================================")
print()

result = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    backend="python",
    device="cpu",
    quiet=False
)

print()
print("Simulation completed!")


# ============================================================
# 9. GET SENSOR SIGNAL
# ============================================================

sensor_signal = result["p"]

print("Original sensor signal shape:", sensor_signal.shape)

# Convert (1, Nt) into (Nt,)
if sensor_signal.ndim == 2:
    sensor_signal = sensor_signal[0, :]

print("Corrected sensor signal shape:", sensor_signal.shape)

print("Time array shape:", time.shape)


# ============================================================
# 10. PLOT RECEIVED SIGNAL
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time * 1000,
    sensor_signal
)

plt.xlabel("Time (ms)")
plt.ylabel("Pressure")

plt.title(
    "Received Signal at Sensor - 500 Hz"
)

plt.grid(True)

plt.tight_layout()

plt.show()