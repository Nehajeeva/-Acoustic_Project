import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. COLUMN PARAMETERS
# ============================================================

TOTAL_HEIGHT = 1.0          # 1 metre = 100 cm
LIQUID_HEIGHT = 0.70        # 70 cm

AIR_HEIGHT = TOTAL_HEIGHT - LIQUID_HEIGHT

SOUND_SPEED = 343.0         # m/s

FREQUENCY = 500              # Hz


# ============================================================
# 2. CALCULATE EXPECTED ECHO
# ============================================================

# Distance from sensor to liquid surface
true_distance = AIR_HEIGHT

# Sound travels down and comes back
echo_time = (2 * true_distance) / SOUND_SPEED

# Convert to milliseconds
echo_time_ms = echo_time * 1000


# Calculate liquid level
measured_distance = (
    SOUND_SPEED * echo_time / 2
)

measured_liquid_height = (
    TOTAL_HEIGHT - measured_distance
)


print()
print("================================================")
print("       ULTRASONIC LIQUID LEVEL SIMULATION")
print("================================================")
print()

print(f"Frequency              : {FREQUENCY} Hz")

print(
    f"Total column height    : "
    f"{TOTAL_HEIGHT * 100:.2f} cm"
)

print(
    f"Actual liquid height   : "
    f"{LIQUID_HEIGHT * 100:.2f} cm"
)

print(
    f"Actual air distance    : "
    f"{AIR_HEIGHT * 100:.2f} cm"
)

print(
    f"Expected echo time     : "
    f"{echo_time_ms:.3f} ms"
)

print()
print("---------- MEASURED RESULT ----------")

print(
    f"Echo time              : "
    f"{echo_time_ms:.3f} ms"
)

print(
    f"Sensor to liquid       : "
    f"{measured_distance * 100:.2f} cm"
)

print(
    f"Liquid level           : "
    f"{measured_liquid_height * 100:.2f} cm"
)

print(
    f"Remaining air          : "
    f"{measured_distance * 100:.2f} cm"
)

print()
print("================================================")
print()


# ============================================================
# 3. ANIMATION PARAMETERS
# ============================================================

# Animation duration slightly longer than echo time
ANIMATION_TIME = 0.0022

# Number of animation frames
NUMBER_OF_FRAMES = 100

times = np.linspace(
    0,
    ANIMATION_TIME,
    NUMBER_OF_FRAMES
)


# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 10)
)


# ============================================================
# 5. DRAW COLUMN
# ============================================================

# Column walls
ax.plot(
    [0, 0],
    [0, 100],
    linewidth=5
)

ax.plot(
    [10, 10],
    [0, 100],
    linewidth=5
)


# ============================================================
# 6. LIQUID REGION
# ============================================================

ax.fill_between(
    [0, 10],
    30,
    100,
    alpha=0.25
)


# Liquid surface
ax.plot(
    [0, 10],
    [30, 30],
    linestyle="--",
    linewidth=3
)


# ============================================================
# 7. SENSOR
# ============================================================

ax.scatter(
    5,
    3,
    s=300,
    marker="s"
)

ax.text(
    5,
    -1,
    "ULTRASONIC\nSENSOR",
    ha="center",
    va="top",
    fontsize=11,
    fontweight="bold"
)


# ============================================================
# 8. DISTANCE LABEL
# ============================================================

ax.annotate(
    "",
    xy=(2, 30),
    xytext=(2, 3),

    arrowprops=dict(
        arrowstyle="<->",
        linewidth=2
    )
)

ax.text(
    1.1,
    16.5,
    "30 cm\nAIR",
    ha="center",
    va="center",
    fontsize=10
)


# ============================================================
# 9. LIQUID LABEL
# ============================================================

ax.text(
    5,
    65,
    "LIQUID\n70 cm",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)


# ============================================================
# 10. WAVE OBJECTS
# ============================================================

wave_line, = ax.plot(
    [],
    [],
    linewidth=5
)

wave_arrow = ax.annotate(
    "",
    xy=(5, 10),
    xytext=(5, 5),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=3
    )
)


# Reflection wave
reflection_line, = ax.plot(
    [],
    [],
    linewidth=5
)


# ============================================================
# 11. TEXT INFORMATION
# ============================================================

status_text = ax.text(
    5,
    105,
    "",
    ha="center",
    fontsize=13,
    fontweight="bold"
)


time_text = ax.text(
    5,
    110,
    "",
    ha="center",
    fontsize=11
)


# ============================================================
# 12. AXIS SETTINGS
# ============================================================

ax.set_xlim(
    -1,
    11
)

ax.set_ylim(
    115,
    -8
)

ax.set_xticks([])

ax.set_ylabel(
    "Height from top (cm)"
)

ax.set_title(
    "Ultrasonic Liquid Level Measurement\n500 Hz",
    fontsize=15,
    fontweight="bold"
)


# ============================================================
# 13. ANIMATION FUNCTION
# ============================================================

def update(frame):

    t = times[frame]

    # --------------------------------------------------------
    # PHYSICAL POSITION OF WAVE
    # --------------------------------------------------------

    # Wave leaves sensor at t = 0
    distance_travelled = SOUND_SPEED * t

    distance_cm = distance_travelled * 100


    # --------------------------------------------------------
    # CASE 1: TRANSMITTED WAVE MOVING DOWN
    # --------------------------------------------------------

    if distance_cm < 27:

        y = 3 + distance_cm

        wave_line.set_data(
            [5, 5],
            [y - 3, y]
        )

        reflection_line.set_data(
            [],
            []
        )

        wave_arrow.set_position(
            (5, y)
        )

        wave_arrow.xy = (
            5,
            y + 1
        )

        wave_arrow.set_position(
            (5, y - 3)
        )

        status_text.set_text(
            "TRANSMITTED WAVE ↓"
        )

        time_text.set_text(
            f"Time = {t * 1000:.3f} ms"
        )


    # --------------------------------------------------------
    # CASE 2: WAVE HITS LIQUID SURFACE
    # --------------------------------------------------------

    elif distance_cm < 33:

        wave_line.set_data(
            [4, 5, 6],
            [30, 29, 30]
        )

        reflection_line.set_data(
            [],
            []
        )

        status_text.set_text(
            "WAVE REACHES LIQUID SURFACE"
        )

        time_text.set_text(
            f"Time = {t * 1000:.3f} ms"
        )


    # --------------------------------------------------------
    # CASE 3: REFLECTED WAVE MOVING UP
    # --------------------------------------------------------

    else:

        # Round trip distance
        return_distance = (
            distance_cm - 30
        )

        # Position of reflected wave
        y = 30 - return_distance

        if y < 3:
            y = 3


        reflection_line.set_data(
            [5, 5],
            [y, y + 3]
        )


        wave_line.set_data(
            [],
            []
        )


        wave_arrow.xy = (
            5,
            y
        )

        wave_arrow.set_position(
            (5, y + 3)
        )


        status_text.set_text(
            "REFLECTED WAVE ↑"
        )

        time_text.set_text(
            f"Time = {t * 1000:.3f} ms"
        )


    # --------------------------------------------------------
    # CASE 4: WAVE RETURNS TO SENSOR
    # --------------------------------------------------------

    if t >= echo_time:

        wave_line.set_data(
            [4.5, 5, 5.5],
            [3, 2, 3]
        )

        reflection_line.set_data(
            [4.5, 5, 5.5],
            [3, 2, 3]
        )

        status_text.set_text(
            "✓ ECHO RECEIVED"
        )

        time_text.set_text(
            f"Echo time = {echo_time_ms:.3f} ms"
        )


    return [
        wave_line,
        reflection_line,
        wave_arrow,
        status_text,
        time_text
    ]


# ============================================================
# 14. CREATE ANIMATION
# ============================================================

animation = FuncAnimation(

    fig,

    update,

    frames=NUMBER_OF_FRAMES,

    interval=40,

    blit=False,

    repeat=True
)


# ============================================================
# 15. SHOW
# ============================================================

plt.tight_layout()

plt.show()