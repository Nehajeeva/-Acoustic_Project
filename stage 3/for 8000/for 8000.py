# ================================================================
# STAGE 1
# SOUND PROPAGATION IN LIQUID COLUMN
#
# Reflection at liquid surface
# Transmission into liquid
# Reflection from liquid bottom
# Automatic surface and bottom echo detection
#
# k-Wave 0.6.2
# Python 3.13
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder


# ================================================================
# 1. PHYSICAL PARAMETERS
# ================================================================

AIR_DISTANCE = 0.300       # m
LIQUID_DEPTH = 0.700       # m

C_AIR = 343.0              # m/s
C_LIQUID = 1480.0          # m/s

RHO_AIR = 1.2              # kg/m^3
RHO_LIQUID = 1000.0        # kg/m^3

FREQUENCY = 8000.0          # Hz

SOURCE_TIME = 0.000300     # s


# ================================================================
# 2. GRID
# ================================================================

dx = 0.005
dy = 0.005

Nx = 240
Ny = 30

kgrid = kWaveGrid(
    [Nx, Ny],
    [dx, dy]
)


# ================================================================
# 3. POSITIONS
# ================================================================

SOURCE_X = 20
SURFACE_X = 80
BOTTOM_X = 220


# ================================================================
# 4. PML
# ================================================================

PML_SIZE = [10, 6]


# ================================================================
# 5. SOURCE / SENSOR WIDTH
# ================================================================

Y1 = 6
Y2 = 24

NUMBER_OF_POINTS = Y2 - Y1


# ================================================================
# 6. CREATE MEDIUM
# ================================================================

sound_speed = np.ones((Nx, Ny)) * C_AIR
density = np.ones((Nx, Ny)) * RHO_AIR

# AIR
sound_speed[:SURFACE_X, :] = C_AIR
density[:SURFACE_X, :] = RHO_AIR

# LIQUID
sound_speed[SURFACE_X:BOTTOM_X, :] = C_LIQUID
density[SURFACE_X:BOTTOM_X, :] = RHO_LIQUID

# HIGH-IMPEDANCE BOTTOM REFLECTOR
BOTTOM_SOUND_SPEED = 3000.0
BOTTOM_DENSITY = 10000.0

sound_speed[BOTTOM_X:, :] = BOTTOM_SOUND_SPEED
density[BOTTOM_X:, :] = BOTTOM_DENSITY

medium = kWaveMedium(
    sound_speed=sound_speed,
    density=density
)


# ================================================================
# 7. TIME SETTINGS
# ================================================================

dt = 1.0e-7

SIM_TIME = 0.004

Nt = int(np.ceil(SIM_TIME / dt)) + 1

kgrid.setTime(Nt, dt)


# ================================================================
# 8. SOURCE
# ================================================================

source = kSource()

source_mask = np.zeros((Nx, Ny), dtype=bool)
source_mask[SOURCE_X, Y1:Y2] = True

source.p_mask = source_mask


# ================================================================
# 9. SHORT GAUSSIAN PULSE AT 600 Hz
# ================================================================

time = np.arange(Nt) * dt

sigma = 0.000055

gaussian = np.exp(
    -((time - SOURCE_TIME) / sigma) ** 2
)

carrier = np.sin(
    2 * np.pi * FREQUENCY * (time - SOURCE_TIME)
)

source_signal = gaussian * carrier

# Normalize source
source_max = np.max(np.abs(source_signal))

if source_max > 0:
    source_signal = source_signal / source_max

# Copy the same signal to every source point
source.p = np.tile(
    source_signal,
    (NUMBER_OF_POINTS, 1)
)

source.p_mode = "additive"


# ================================================================
# 10. SENSOR
# ================================================================

sensor = kSensor()

sensor_mask = np.zeros((Nx, Ny), dtype=bool)
sensor_mask[SOURCE_X, Y1:Y2] = True

sensor.mask = sensor_mask
sensor.record = ["p"]


# ================================================================
# 11. THEORETICAL TIMES
# ================================================================

# Surface reflection: source -> surface -> sensor
air_round_trip = 2 * AIR_DISTANCE / C_AIR

surface_expected = SOURCE_TIME + air_round_trip

# Bottom reflection: surface -> bottom -> surface
liquid_round_trip = 2 * LIQUID_DEPTH / C_LIQUID

bottom_expected = surface_expected + liquid_round_trip


# ================================================================
# 12. PRINT INFORMATION
# ================================================================

print()
print("=" * 70)
print("STAGE 1 - REFLECTION + TRANSMISSION + BOTTOM REFLECTION")
print("=" * 70)

print()
print("GEOMETRY")
print("-" * 70)

print(f"Frequency               : {FREQUENCY:.1f} Hz")
print(f"Source / Sensor position: {SOURCE_X}")
print(f"Liquid surface position : {SURFACE_X}")
print(f"Liquid bottom position  : {BOTTOM_X}")
print(f"Air distance            : {AIR_DISTANCE:.3f} m")
print(f"Liquid depth            : {LIQUID_DEPTH:.3f} m")

print()
print("EXPECTED ECHO TIMES")
print("-" * 70)

print(f"Source emission      : {SOURCE_TIME * 1000:.4f} ms")
print(f"Surface reflection   : {surface_expected * 1000:.4f} ms")
print(f"Liquid round trip    : {liquid_round_trip * 1000:.4f} ms")
print(f"Bottom reflection    : {bottom_expected * 1000:.4f} ms")

print()
print("TIME SETTINGS")
print("-" * 70)

print(f"Time step            : {dt:.3e} s")
print(f"Simulation time      : {SIM_TIME * 1000:.3f} ms")
print(f"Number of time steps : {Nt}")
print(f"PML                  : {PML_SIZE}")


# ================================================================
# 13. RUN K-WAVE
# ================================================================

print()
print("=" * 70)
print("RUNNING K-WAVE")
print("=" * 70)

print()
print("Please wait...")

result = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    pml_size=PML_SIZE,
    pml_inside=True,
    quiet=False
)

print()
print("=" * 70)
print("K-WAVE COMPLETED")
print("=" * 70)


# ================================================================
# 14. EXTRACT RAW PRESSURE
# ================================================================

if isinstance(result, dict):
    raw_pressure = np.asarray(result["p"])
else:
    raw_pressure = np.asarray(result)

print()
print("Raw pressure shape:", raw_pressure.shape)


# ================================================================
# 15. PROCESS SENSOR DATA
# ================================================================

if raw_pressure.ndim == 2:
    pressure = np.mean(raw_pressure, axis=0)
else:
    pressure = raw_pressure.flatten()

print("Processed signal length:", len(pressure))


# ================================================================
# 16. ACTUAL TIME ARRAY
# ================================================================

time = np.arange(len(pressure)) * dt


# ================================================================
# 17. PRESSURE ENVELOPE
# ================================================================

analytic_signal = hilbert(pressure)
envelope = np.abs(analytic_signal)

maximum = np.max(envelope)

if maximum > 0:
    envelope = envelope / maximum


# ================================================================
# 18. PEAK DETECTION
# ================================================================

peaks, properties = find_peaks(
    envelope,
    distance=int(50e-6 / dt),
    prominence=0.0001
)

print()
print("=" * 70)
print("DETECTED SIGNAL PEAKS")
print("=" * 70)

for p in peaks:
    if envelope[p] > 0.0001:
        print(
            f"Time = {time[p] * 1000:.4f} ms   "
            f"Envelope = {envelope[p]:.6f}"
        )


# ================================================================
# 19. ECHO DETECTION FUNCTION
# ================================================================

def find_echo(
    expected_time,
    before,
    after,
    minimum_amplitude
):
    start = expected_time - before
    end = expected_time + after

    indices = np.where(
        (time >= start) &
        (time <= end)
    )[0]

    if len(indices) == 0:
        return None

    local_envelope = envelope[indices]

    local_peaks, _ = find_peaks(
        local_envelope,
        prominence=minimum_amplitude
    )

    if len(local_peaks) == 0:
        return None

    candidate_indices = indices[local_peaks]
    candidate_values = envelope[candidate_indices]

    strongest = np.argmax(candidate_values)
    selected = candidate_indices[strongest]

    if envelope[selected] < minimum_amplitude:
        return None

    return selected


# ================================================================
# 20. DETECT SURFACE
# ================================================================

surface_index = find_echo(
    expected_time=surface_expected,
    before=0.25e-3,
    after=0.25e-3,
    minimum_amplitude=0.005
)


# ================================================================
# 21. DETECT BOTTOM
# ================================================================

bottom_index = find_echo(
    expected_time=bottom_expected,
    before=0.25e-3,
    after=0.25e-3,
    minimum_amplitude=0.0005
)


# ================================================================
# 22. DETECTED TIMES
# ================================================================

if surface_index is not None:
    surface_detected = time[surface_index]
else:
    surface_detected = None

if bottom_index is not None:
    bottom_detected = time[bottom_index]
else:
    bottom_detected = None


# ================================================================
# 23. AUTOMATIC DETECTION RESULT
# ================================================================

print()
print("=" * 70)
print("AUTOMATIC ECHO DETECTION")
print("=" * 70)

print()
print(f"Source reference time : {SOURCE_TIME * 1000:.4f} ms")
print()
print(f"Surface expected      : {surface_expected * 1000:.4f} ms")

if surface_detected is not None:
    print(f"Surface detected      : {surface_detected * 1000:.4f} ms")
else:
    print("Surface reflection NOT detected.")

print()
print(f"Bottom expected       : {bottom_expected * 1000:.4f} ms")

if bottom_detected is not None:
    print(f"Bottom detected       : {bottom_detected * 1000:.4f} ms")
else:
    print("Bottom reflection NOT detected.")


# ================================================================
# 24. DISTANCE CALCULATION
# ================================================================

if (
    surface_detected is not None
    and bottom_detected is not None
):
    # AIR: surface echo minus source emission
    measured_air_round_trip = (
        surface_detected - SOURCE_TIME
    )

    calculated_air_distance = (
        C_AIR * measured_air_round_trip / 2
    )

    # LIQUID: bottom echo minus surface echo
    measured_liquid_round_trip = (
        bottom_detected - surface_detected
    )

    calculated_liquid_depth = (
        C_LIQUID * measured_liquid_round_trip / 2
    )

    calculated_total = (
        calculated_air_distance + calculated_liquid_depth
    )

    air_error = abs(
        calculated_air_distance - AIR_DISTANCE
    )

    liquid_error = abs(
        calculated_liquid_depth - LIQUID_DEPTH
    )

    # ============================================================
    # 25. FINAL RESULTS
    # ============================================================

    print()
    print("=" * 70)
    print("FINAL DISTANCE RESULTS")
    print("=" * 70)

    print()
    print(f"Source emission time      : {SOURCE_TIME * 1000:.4f} ms")
    print(f"Surface echo time         : {surface_detected * 1000:.4f} ms")
    print(f"Bottom echo time          : {bottom_detected * 1000:.4f} ms")

    print()
    print(
        f"Measured air round trip   : "
        f"{measured_air_round_trip * 1000:.4f} ms"
    )

    print(
        f"Measured liquid round trip: "
        f"{measured_liquid_round_trip * 1000:.4f} ms"
    )

    print()
    print(f"Calculated air distance   : {calculated_air_distance:.4f} m")
    print(f"Calculated liquid depth   : {calculated_liquid_depth:.4f} m")
    print(f"Calculated total          : {calculated_total:.4f} m")

    print()
    print(f"Actual air distance       : {AIR_DISTANCE:.4f} m")
    print(f"Actual liquid depth       : {LIQUID_DEPTH:.4f} m")

    print()
    print(f"Air distance error        : {air_error:.4f} m")
    print(f"Liquid depth error        : {liquid_error:.4f} m")

else:
    print()
    print("=" * 70)
    print("DISTANCE CALCULATION NOT PERFORMED")
    print("=" * 70)

    if surface_detected is None:
        print("Surface reflection was not detected.")

    if bottom_detected is None:
        print("Bottom reflection was not detected.")


# ================================================================
# 26. SAVE DATA FOR STAGE 2
# ================================================================

np.save("stage1_time.npy", time)
np.save("stage1_pressure.npy", pressure)
np.save("stage1_envelope.npy", envelope)


# ================================================================
# 27. SAVE RESULTS
# ================================================================

with open("stage1_results.txt", "w") as f:
    f.write("STAGE 1 - REFLECTION + TRANSMISSION\n")
    f.write("====================================\n\n")

    f.write(f"Frequency = {FREQUENCY:.2f} Hz\n")
    f.write(f"Air distance = {AIR_DISTANCE:.6f} m\n")
    f.write(f"Liquid depth = {LIQUID_DEPTH:.6f} m\n")
    f.write(f"Source time = {SOURCE_TIME * 1000:.6f} ms\n")
    f.write(f"Expected surface = {surface_expected * 1000:.6f} ms\n")
    f.write(f"Expected bottom = {bottom_expected * 1000:.6f} ms\n")

    if surface_detected is not None:
        f.write(
            f"Detected surface = "
            f"{surface_detected * 1000:.6f} ms\n"
        )

    if bottom_detected is not None:
        f.write(
            f"Detected bottom = "
            f"{bottom_detected * 1000:.6f} ms\n"
        )

    if (
        surface_detected is not None
        and bottom_detected is not None
    ):
        f.write(
            f"\nCalculated air = "
            f"{calculated_air_distance:.6f} m\n"
        )

        f.write(
            f"Calculated liquid = "
            f"{calculated_liquid_depth:.6f} m\n"
        )

        f.write(
            f"Calculated total = "
            f"{calculated_total:.6f} m\n"
        )

        f.write(
            f"\nAir error = "
            f"{air_error:.6f} m\n"
        )

        f.write(
            f"Liquid error = "
            f"{liquid_error:.6f} m\n"
        )


# ================================================================
# 28. GRAPH
# ================================================================

plt.figure(figsize=(14, 7))

time_ms = time * 1000

pressure_plot = pressure.copy()

max_pressure = np.max(np.abs(pressure_plot))

if max_pressure > 0:
    pressure_plot = pressure_plot / max_pressure

plt.plot(
    time_ms,
    pressure_plot,
    label="Received pressure"
)

plt.plot(
    time_ms,
    envelope,
    label="Pressure envelope"
)

plt.axvline(
    SOURCE_TIME * 1000,
    linestyle="--",
    label="Source"
)

plt.axvline(
    surface_expected * 1000,
    linestyle="--",
    label="Expected surface"
)

plt.axvline(
    bottom_expected * 1000,
    linestyle="--",
    label="Expected bottom"
)

if surface_detected is not None:
    plt.axvline(
        surface_detected * 1000,
        linestyle=":",
        label="Detected surface"
    )

if bottom_detected is not None:
    plt.axvline(
        bottom_detected * 1000,
        linestyle=":",
        label="Detected bottom"
    )

plt.xlabel("Time (ms)")
plt.ylabel("Normalized pressure")
plt.title(
    "Stage 1 - Reflection + Transmission + Bottom Reflection "
    "(600 Hz)"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "stage1_echo_signal.png",
    dpi=200
)

plt.show()


# ================================================================
# 29. FINISHED
# ================================================================

print()
print("=" * 70)
print("STAGE 1 FINISHED")
print("=" * 70)

print()
print("Saved files:")
print("stage1_time.npy")
print("stage1_pressure.npy")
print("stage1_envelope.npy")
print("stage1_results.txt")
print("stage1_echo_signal.png")

print("=" * 70)