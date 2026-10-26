"""
Self-contained DBS validation for the real 1920 x 1080 DMD center scan point.

No external pattern/image files are required.

Running this script will:
1) Generate the original direct binary grating.
2) Optimize its exact 45 x 96 periodic supercell using harmonic-aware
   Direct Binary Search (DBS).
3) Tile the optimized supercell to the full 1080 x 1920 DMD.
4) Compute full-DMD Fourier-plane intensities.
5) Save separate 600-dpi PNG figures:
   - full direct DMD pattern
   - full DBS-optimized DMD pattern
   - identical 100 x 100 local DMD zooms
   - direct +1-order linear intensity
   - DBS +1-order linear intensity
   - direct +3-order linear intensity
   - DBS +3-order linear intensity
   - higher-order / +1 intensity-ratio curve
6) Save binary TIFF patterns and a quantitative text report.

The output folder is created beside this .py file, so the code can be run
directly on Windows without editing any input file paths.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. OUTPUT DIRECTORY
# ============================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUTPUT_DIR = SCRIPT_DIR / "DBS_self_contained_output1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REAL DMD / OPTICAL PARAMETERS
# ============================================================

HEIGHT = 1080
WIDTH = 1920

WAVELENGTH = 532.47e-9       # m
FOCAL_LENGTH = 300e-3        # m
DMD_PIXEL_PITCH = 10.8e-6    # m

# Original scan-grid settings
INTERVAL_X = 11 / 4          # 2.75 Fourier-bin units
INTERVAL_Y = 20 / 4          # 5.00 Fourier-bin units

# Center scan point of the original 101 x 101 scanning array
I_CENTER = 96
J_CENTER = 92

# Full-DMD Fourier carrier offsets
X1 = I_CENTER * INTERVAL_X   # 264
Y1 = J_CENTER * INTERVAL_Y   # 460

# Exact rational reduction:
# 264 / 1080 = 11 / 45
# 460 / 1920 = 23 / 96
CELL_H = 45
CELL_W = 96
KX = 11
KY = 23

assert HEIGHT % CELL_H == 0
assert WIDTH % CELL_W == 0

TILE_H = HEIGHT // CELL_H    # 24
TILE_W = WIDTH // CELL_W     # 20
N_CELL = CELL_H * CELL_W


# ============================================================
# 3. DBS SETTINGS
# ============================================================

DBS_SWEEPS = 10
RANDOM_SEED = 1

# Explicitly penalized higher diffraction orders
HARMONIC_ORDERS = (3, 5, 7, 9)

HARMONIC_WEIGHTS = {
    3: 1.0,
    5: 1.0,
    7: 1.0,
    9: 0.5,
}

# Weak penalty on total residual non-target Fourier energy
BETA_TOTAL_OTHER = 0.01

# Preserve at least ~75% of the original first-order intensity
MIN_FIRST_ORDER_FRACTION = 0.75
GAMMA_FIRST_ORDER = 200.0

# Keep overall binary fill factor near 50%
DELTA_FILL = 0.20


# ============================================================
# 4. FIGURE SETTINGS
# ============================================================

DPI = 600

# Same DMD region for direct / DBS zoom comparison
ZOOM_ROW0 = 490
ZOOM_ROW1 = 590
ZOOM_COL0 = 910
ZOOM_COL1 = 1010

# Half-width around +1 / +3 Fourier peaks
FOCAL_HALF_SIZE = 28

# Ratio-plot orders
RATIO_ORDERS = (1, 3, 5, 7, 9)


# ============================================================
# 5. FOURIER / GEOMETRY HELPERS
# ============================================================

def centered_fft2(x):
    """Centered 2D Fourier transform."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def wrap_centered_offset(offset, length):
    """
    Map an integer Fourier offset into the principal centered FFT interval.
    """
    return int(
        (int(round(offset)) + length // 2) % length
        - length // 2
    )


def full_order_offset(order):
    """
    Aliased +order diffraction position in the full 1080 x 1920 FFT plane.
    """
    dr = wrap_centered_offset(order * int(round(X1)), HEIGHT)
    dc = wrap_centered_offset(order * int(round(Y1)), WIDTH)
    return dr, dc


def full_order_pixel(order):
    """
    Pixel coordinate of the +order diffraction spot in centered FFT output.
    """
    dr, dc = full_order_offset(order)
    return HEIGHT // 2 + dr, WIDTH // 2 + dc


def focal_plane_sampling():
    """
    Fourier-plane sample spacing using the same relation as the scan code.
    """
    dx = (
        WAVELENGTH
        * FOCAL_LENGTH
        / (DMD_PIXEL_PITCH * HEIGHT)
    )

    dy = (
        WAVELENGTH
        * FOCAL_LENGTH
        / (DMD_PIXEL_PITCH * WIDTH)
    )

    return dx, dy


def focal_plane_position_mm(order):
    """Physical focal-plane position of a diffraction order."""
    dx, dy = focal_plane_sampling()
    dr, dc = full_order_offset(order)
    return dr * dx * 1e3, dc * dy * 1e3


def crop_around(arr, center_rc, half_size):
    """Crop a square region around a specified pixel."""
    r, c = center_rc

    r0 = max(0, r - half_size)
    r1 = min(arr.shape[0], r + half_size + 1)

    c0 = max(0, c - half_size)
    c1 = min(arr.shape[1], c + half_size + 1)

    return arr[r0:r1, c0:c1]


def save_binary_tif(pattern, path):
    """Save a binary 0/1 pattern as a 1-bit TIFF."""
    arr = (pattern > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").convert("1").save(path)


# ============================================================
# 6. DIRECT BINARY GRATING
# ============================================================

def make_direct_supercell():
    """
    Generate the exact 45 x 96 periodic supercell corresponding to

        pattern = cos[
            2*pi*(264/1080*row + 460/1920*col)
        ] > 0

    using the exact rational frequencies 11/45 and 23/96.
    """
    r, c = np.indices(
        (CELL_H, CELL_W),
        dtype=float
    )

    phase = 2 * np.pi * (
        KX * r / CELL_H
        + KY * c / CELL_W
    )

    return (np.cos(phase) > 0).astype(np.uint8)


def tile_to_full_dmd(cell):
    """Tile one supercell to the complete 1080 x 1920 DMD."""
    return np.tile(
        cell,
        (TILE_H, TILE_W)
    )


# ============================================================
# 7. FAST INCREMENTAL FOURIER MODEL FOR DBS
# ============================================================

TRACKED_ORDERS = (1,) + HARMONIC_ORDERS

TRACKED_FREQS = [
    (
        (order * KX) % CELL_H,
        (order * KY) % CELL_W
    )
    for order in TRACKED_ORDERS
]

_rr, _cc = np.indices(
    (CELL_H, CELL_W),
    dtype=float
)

# Fourier-coefficient change produced by a unit flip at every supercell pixel
PHASE_MAPS = np.asarray([
    np.exp(
        -2j * np.pi * (
            kx * _rr / CELL_H
            + ky * _cc / CELL_W
        )
    )
    for kx, ky in TRACKED_FREQS
])


def initial_tracked_coefficients(cell):
    """Extract Fourier coefficients at all monitored diffraction orders."""
    F = np.fft.fft2(cell)

    return np.asarray([
        F[kx, ky]
        for kx, ky in TRACKED_FREQS
    ])


def evaluate_objective(Ftrack, n_on, I1_ref):
    """
    Harmonic-aware DBS objective:

        J =
            sum_n w_n * I_n / I_1
            + beta * E_other / (2 I_1)
            + gamma *
              max(0, eta_min - I_1/I_1,ref)^2
            + delta * (fill - 0.5)^2

    Lower J is better.
    """
    powers = np.abs(Ftrack) ** 2
    I1 = float(powers[0])

    if I1 <= 1e-30:
        return np.inf, None

    harmonic_ratios = {}
    harmonic_cost = 0.0

    for j, order in enumerate(
        HARMONIC_ORDERS,
        start=1
    ):
        ratio = float(
            powers[j] / I1
        )

        harmonic_ratios[order] = ratio

        harmonic_cost += (
            HARMONIC_WEIGHTS[order]
            * ratio
        )

    # Parseval for unnormalized FFT:
    # total Fourier energy = N * number_of_ON_pixels
    total_fourier_energy = N_CELL * n_on

    # DC Fourier energy for a binary array
    dc_energy = n_on ** 2

    # Remove DC and desired +/-1 contribution
    other_energy = max(
        0.0,
        float(
            total_fourier_energy
            - dc_energy
            - 2 * I1
        )
    )

    other_ratio = (
        other_energy
        / (2 * I1)
    )

    first_order_fraction = (
        I1 / I1_ref
    )

    first_order_deficit = max(
        0.0,
        MIN_FIRST_ORDER_FRACTION
        - first_order_fraction
    )

    fill = (
        n_on / N_CELL
    )

    J = (
        harmonic_cost
        + BETA_TOTAL_OTHER * other_ratio
        + GAMMA_FIRST_ORDER
          * first_order_deficit ** 2
        + DELTA_FILL
          * (fill - 0.5) ** 2
    )

    metrics = {
        "I1": I1,
        "I1_fraction": first_order_fraction,
        "harmonic_ratios": harmonic_ratios,
        "other_energy_ratio": other_ratio,
        "fill": fill,
    }

    return J, metrics


# ============================================================
# 8. DIRECT BINARY SEARCH
# ============================================================

def optimize_supercell_dbs(initial_cell):
    """
    Optimize the 45 x 96 periodic binary supercell using DBS.
    """
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    cell = initial_cell.copy()

    Ftrack = initial_tracked_coefficients(
        cell
    )

    n_on = int(
        np.sum(cell)
    )

    I1_ref = float(
        np.abs(Ftrack[0]) ** 2
    )

    best_J, best_metrics = evaluate_objective(
        Ftrack,
        n_on,
        I1_ref
    )

    history = [{
        "sweep": 0,
        "accepted": 0,
        "J": best_J,
        **best_metrics,
    }]

    print("=" * 78)
    print("DBS OPTIMIZATION")
    print("=" * 78)

    print(
        f"Supercell:                  "
        f"{CELL_H} x {CELL_W}"
    )

    print(
        f"Independent binary pixels: "
        f"{N_CELL}"
    )

    print(
        f"Full DMD:                   "
        f"{HEIGHT} x {WIDTH}"
    )

    print(
        f"Carrier offset:             "
        f"({X1:.0f}, {Y1:.0f}) FFT bins"
    )

    print(
        f"Initial objective:          "
        f"{best_J:.8e}"
    )

    for order in HARMONIC_ORDERS:
        print(
            f"Initial I{order}/I1:          "
            f"{100 * best_metrics['harmonic_ratios'][order]:.6f}%"
        )

    # --------------------------------------------------------
    # DBS sweeps
    # --------------------------------------------------------
    for sweep in range(
        1,
        DBS_SWEEPS + 1
    ):

        accepted = 0

        # Random pixel visitation order
        for linear_index in rng.permutation(
            N_CELL
        ):

            r = int(
                linear_index // CELL_W
            )

            c = int(
                linear_index % CELL_W
            )

            old_value = int(
                cell[r, c]
            )

            # 0 -> 1 gives +1
            # 1 -> 0 gives -1
            delta = (
                +1
                if old_value == 0
                else -1
            )

            candidate_F = (
                Ftrack
                + delta
                * PHASE_MAPS[:, r, c]
            )

            candidate_n_on = (
                n_on + delta
            )

            new_J, new_metrics = evaluate_objective(
                candidate_F,
                candidate_n_on,
                I1_ref
            )

            # Accept only if objective decreases
            if new_J < best_J:

                cell[r, c] = (
                    1 - old_value
                )

                Ftrack = candidate_F
                n_on = candidate_n_on

                best_J = new_J
                best_metrics = new_metrics

                accepted += 1

        history.append({
            "sweep": sweep,
            "accepted": accepted,
            "J": best_J,
            **best_metrics,
        })

        harmonic_text = ", ".join([
            (
                f"I{o}/I1="
                f"{100 * best_metrics['harmonic_ratios'][o]:.4f}%"
            )
            for o in HARMONIC_ORDERS
        ])

        print(
            f"Sweep {sweep:2d}: "
            f"accepted={accepted:4d}, "
            f"I1 retained="
            f"{100 * best_metrics['I1_fraction']:.3f}%, "
            f"{harmonic_text}"
        )

        # Early stopping
        if accepted == 0:
            break

    return cell, history


# ============================================================
# 9. FULL-DMD FOURIER VALIDATION
# ============================================================

def full_dmd_metrics(pattern):
    """
    Calculate full 1080 x 1920 FFT and diffraction-order metrics.
    """
    F = centered_fft2(pattern)
    I = np.abs(F) ** 2

    p1_plus = full_order_pixel(+1)
    p1_minus = full_order_pixel(-1)

    I1_plus = float(
        I[p1_plus]
    )

    I1_minus = float(
        I[p1_minus]
    )

    # Average symmetric +/-1 intensities
    I1 = 0.5 * (
        I1_plus
        + I1_minus
    )

    order_metrics = {}

    for order in HARMONIC_ORDERS:

        p_plus = full_order_pixel(
            +order
        )

        p_minus = full_order_pixel(
            -order
        )

        value = 0.5 * (
            float(I[p_plus])
            + float(I[p_minus])
        )

        order_metrics[order] = {
            "intensity": value,
            "ratio": (
                value
                / (I1 + 1e-30)
            ),
            "plus_pixel": p_plus,
            "minus_pixel": p_minus,
            "plus_position_mm":
                focal_plane_position_mm(
                    +order
                ),
            "minus_position_mm":
                focal_plane_position_mm(
                    -order
                ),
        }

    # Strongest non-target Fourier pixel
    mask = np.ones(
        I.shape,
        dtype=bool
    )

    excluded = [
        (
            HEIGHT // 2,
            WIDTH // 2
        ),
        p1_plus,
        p1_minus,
    ]

    for p in excluded:
        mask[p] = False

    masked = np.where(
        mask,
        I,
        -np.inf
    )

    max_other_pixel = (
        np.unravel_index(
            np.argmax(masked),
            I.shape
        )
    )

    max_other = float(
        I[max_other_pixel]
    )

    return {
        "F": F,
        "I": I,
        "I1": I1,
        "I1_plus": I1_plus,
        "I1_minus": I1_minus,
        "order_metrics": order_metrics,
        "max_other_ratio":
            max_other
            / (I1 + 1e-30),
        "max_other_pixel":
            max_other_pixel,
    }


# ============================================================
# 10. VISUALIZATION HELPERS
# ============================================================

def save_full_dmd_pattern(
    pattern,
    filename,
    title
):
    """
    Full 1920 x 1080 DMD pattern.
    aspect='equal' preserves the true rectangular geometry.
    """
    fig, ax = plt.subplots(
        figsize=(12, 6.75)
    )

    ax.imshow(
        pattern,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    ax.set_title(title)
    ax.set_xlabel("DMD column")
    ax.set_ylabel("DMD row")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_dmd_zoom(
    pattern_zoom,
    filename,
    title
):
    """
    Local 100 x 100 binary DMD region.
    """
    fig, ax = plt.subplots(
        figsize=(5.2, 5.2)
    )

    ax.imshow(
        pattern_zoom,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    ax.set_title(title)
    ax.set_xlabel("Local column")
    ax.set_ylabel("Local row")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_focal_order(
    intensity_crop,
    filename,
    title
):
    """
    Save one +1 or +3 focal-order image using a linear intensity scale.

    All intensity maps are normalized to Direct +1.
    The same display range [0,1] is used for all maps.
    """
    fig, ax = plt.subplots(
        figsize=(5.0, 4.5)
    )

    im = ax.imshow(
        intensity_crop,
        cmap="hot",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal"
    )

    ax.set_title(title)
    ax.set_xlabel(
        "FFT column offset"
    )
    ax.set_ylabel(
        "FFT row offset"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "Linear intensity\nnormalized to Direct +1"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_order_ratio_plot(
    orders,
    direct_ratios,
    optimized_ratios
):
    """
    Plot I_n / I_1 versus diffraction order.

    Direct ratios are normalized by Direct I1.
    DBS ratios are normalized by DBS I1.
    """
    fig, ax = plt.subplots(
        figsize=(5.8, 4.7)
    )

    ax.plot(
        orders,
        direct_ratios,
        "-o",
        linewidth=1.6,
        markersize=5,
        label="Direct"
    )

    ax.plot(
        orders,
        optimized_ratios,
        "-s",
        linewidth=1.6,
        markersize=5,
        label="DBS"
    )

    ax.set_yscale("log")

    ax.set_xticks(
        orders
    )

    ax.set_xlabel(
        "Diffraction order"
    )

    ax.set_ylabel(
        r"$I_n/I_1$"
    )

    ax.set_title(
        "Higher-order to +1 intensity ratio"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "higher_order_to_plus1_ratio.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_convergence_plot(history):
    """
    Optional DBS convergence plot.
    """
    sweeps = [
        h["sweep"]
        for h in history
    ]

    objective = [
        h["J"]
        for h in history
    ]

    fig, ax = plt.subplots(
        figsize=(5.5, 4.2)
    )

    ax.plot(
        sweeps,
        objective,
        "-o"
    )

    ax.set_xlabel(
        "DBS sweep"
    )

    ax.set_ylabel(
        "Objective J"
    )

    ax.set_title(
        "DBS convergence"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "dbs_convergence.png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR.resolve()}\n"
    )

    # --------------------------------------------------------
    # Step 1: Generate Direct binary grating
    # --------------------------------------------------------
    direct_cell = (
        make_direct_supercell()
    )

    direct_full = (
        tile_to_full_dmd(
            direct_cell
        )
    )

    # --------------------------------------------------------
    # Step 2: DBS optimization
    # --------------------------------------------------------
    optimized_cell, history = (
        optimize_supercell_dbs(
            direct_cell
        )
    )

    optimized_full = (
        tile_to_full_dmd(
            optimized_cell
        )
    )

    # --------------------------------------------------------
    # Step 3: Full-DMD Fourier analysis
    # --------------------------------------------------------
    print(
        "\nComputing full 1080 x 1920 "
        "Fourier-plane intensities..."
    )

    direct_metrics = (
        full_dmd_metrics(
            direct_full
        )
    )

    optimized_metrics = (
        full_dmd_metrics(
            optimized_full
        )
    )

    I_direct = (
        direct_metrics["I"]
    )

    I_opt = (
        optimized_metrics["I"]
    )

    # --------------------------------------------------------
    # Step 4: Quantitative summary
    # --------------------------------------------------------
    changed_fraction = float(
        np.mean(
            direct_full
            != optimized_full
        )
    )

    first_retention = (
        optimized_metrics["I1"]
        / direct_metrics["I1"]
    )

    print("\n" + "=" * 78)
    print("FULL-DMD RESULTS")
    print("=" * 78)

    print(
        f"Changed pixels: "
        f"{100 * changed_fraction:.4f}%"
    )

    print(
        f"+1 intensity retained: "
        f"{100 * first_retention:.4f}%"
    )

    for order in HARMONIC_ORDERS:

        before = (
            direct_metrics[
                "order_metrics"
            ][order]["ratio"]
        )

        after = (
            optimized_metrics[
                "order_metrics"
            ][order]["ratio"]
        )

        print(
            f"I{order}/I1: "
            f"{100 * before:.6f}% "
            f"-> "
            f"{100 * after:.6f}% "
            f"("
            f"{before/(after+1e-30):.2f}x "
            f"suppression)"
        )

    # --------------------------------------------------------
    # Step 5: Save binary DMD patterns
    # --------------------------------------------------------
    save_binary_tif(
        direct_full,
        OUTPUT_DIR
        / "pattern_direct_1080x1920.tif"
    )

    save_binary_tif(
        optimized_full,
        OUTPUT_DIR
        / "pattern_DBS_1080x1920.tif"
    )

    save_binary_tif(
        direct_cell,
        OUTPUT_DIR
        / "supercell_direct_45x96.tif"
    )

    save_binary_tif(
        optimized_cell,
        OUTPUT_DIR
        / "supercell_DBS_45x96.tif"
    )

    # --------------------------------------------------------
    # Step 6: DMD full-field figures
    # --------------------------------------------------------
    save_full_dmd_pattern(
        direct_full,
        "01_direct_binary_DMD_full.png",
        "Direct binary DMD pattern"
    )

    save_full_dmd_pattern(
        optimized_full,
        "02_DBS_binary_DMD_full.png",
        "DBS-optimized binary DMD pattern"
    )

    # --------------------------------------------------------
    # Step 7: Identical 100 x 100 DMD zoom regions
    # --------------------------------------------------------
    direct_zoom = direct_full[
        ZOOM_ROW0:ZOOM_ROW1,
        ZOOM_COL0:ZOOM_COL1
    ]

    optimized_zoom = optimized_full[
        ZOOM_ROW0:ZOOM_ROW1,
        ZOOM_COL0:ZOOM_COL1
    ]

    save_dmd_zoom(
        direct_zoom,
        "03_direct_DMD_zoom_100x100.png",
        "Direct DMD zoom (100×100)"
    )

    save_dmd_zoom(
        optimized_zoom,
        "04_DBS_DMD_zoom_100x100.png",
        "DBS DMD zoom (100×100)"
    )

    # --------------------------------------------------------
    # Step 8: +1 / +3 positions and intensities
    # --------------------------------------------------------
    p1 = full_order_pixel(+1)
    p3 = full_order_pixel(+3)

    I1_direct = float(
        I_direct[p1]
    )

    I1_opt = float(
        I_opt[p1]
    )

    I3_direct = float(
        I_direct[p3]
    )

    I3_opt = float(
        I_opt[p3]
    )

    # Common display normalization:
    # Direct +1 intensity = 1
    I_direct_norm = (
        I_direct
        / I1_direct
    )

    I_opt_norm = (
        I_opt
        / I1_direct
    )

    d1 = crop_around(
        I_direct_norm,
        p1,
        FOCAL_HALF_SIZE
    )

    o1 = crop_around(
        I_opt_norm,
        p1,
        FOCAL_HALF_SIZE
    )

    d3 = crop_around(
        I_direct_norm,
        p3,
        FOCAL_HALF_SIZE
    )

    o3 = crop_around(
        I_opt_norm,
        p3,
        FOCAL_HALF_SIZE
    )

    # --------------------------------------------------------
    # Step 9: Save +1 / +3 individual focal-order figures
    # --------------------------------------------------------
    save_focal_order(
        d1,
        "05_direct_plus1_linear.png",
        (
            "Direct +1 order\n"
            f"I / I(+1, direct) = "
            f"{I1_direct/I1_direct:.3f}"
        )
    )

    save_focal_order(
        o1,
        "06_DBS_plus1_linear.png",
        (
            "DBS +1 order\n"
            f"I / I(+1, direct) = "
            f"{I1_opt/I1_direct:.3f}"
        )
    )

    save_focal_order(
        d3,
        "07_direct_plus3_linear.png",
        (
            "Direct +3 order\n"
            f"I / I(+1, direct) = "
            f"{I3_direct/I1_direct:.4f};  "
            f"I3/I1 = "
            f"{I3_direct/I1_direct:.4f}"
        )
    )

    save_focal_order(
        o3,
        "08_DBS_plus3_linear.png",
        (
            "DBS +3 order\n"
            f"I / I(+1, direct) = "
            f"{I3_opt/I1_direct:.4f};  "
            f"I3/I1 = "
            f"{I3_opt/I1_opt:.4f}"
        )
    )

    # --------------------------------------------------------
    # Step 10: Higher-order / +1 ratio curve
    # --------------------------------------------------------
    ratio_orders = np.asarray(
        RATIO_ORDERS,
        dtype=int
    )

    direct_ratios = []
    optimized_ratios = []

    for order in ratio_orders:

        p = full_order_pixel(
            int(order)
        )

        if order == 1:
            direct_ratio = 1.0
            optimized_ratio = 1.0
        else:
            direct_ratio = (
                float(I_direct[p])
                / I1_direct
            )

            optimized_ratio = (
                float(I_opt[p])
                / I1_opt
            )

        direct_ratios.append(
            direct_ratio
        )

        optimized_ratios.append(
            optimized_ratio
        )

    direct_ratios = np.asarray(
        direct_ratios
    )

    optimized_ratios = np.asarray(
        optimized_ratios
    )

    save_order_ratio_plot(
        ratio_orders,
        direct_ratios,
        optimized_ratios
    )

    # --------------------------------------------------------
    # Step 11: DBS convergence figure
    # --------------------------------------------------------
    save_convergence_plot(
        history
    )

    # --------------------------------------------------------
    # Step 12: Quantitative text report
    # --------------------------------------------------------
    report_path = (
        OUTPUT_DIR
        / "DBS_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DBS validation for the "
            "1080 x 1920 DMD center scan point\n"
        )

        f.write(
            "=" * 78
            + "\n"
        )

        f.write(
            f"DMD: "
            f"{HEIGHT} x {WIDTH}\n"
        )

        f.write(
            f"Supercell: "
            f"{CELL_H} x {CELL_W}\n"
        )

        f.write(
            f"Carrier offset: "
            f"({X1:.0f}, {Y1:.0f})\n\n"
        )

        f.write(
            f"Changed DMD pixels: "
            f"{100 * changed_fraction:.8f}%\n"
        )

        f.write(
            f"+1 intensity retained: "
            f"{100 * first_retention:.8f}%\n\n"
        )

        f.write(
            f"Direct +1 intensity: "
            f"{I1_direct:.12e}\n"
        )

        f.write(
            f"DBS +1 intensity: "
            f"{I1_opt:.12e}\n"
        )

        f.write(
            f"Direct +3 intensity: "
            f"{I3_direct:.12e}\n"
        )

        f.write(
            f"DBS +3 intensity: "
            f"{I3_opt:.12e}\n\n"
        )

        f.write(
            f"DBS +1 / Direct +1: "
            f"{I1_opt/I1_direct:.12e}\n"
        )

        f.write(
            f"Direct I3/I1: "
            f"{I3_direct/I1_direct:.12e}\n"
        )

        f.write(
            f"DBS I3/I1: "
            f"{I3_opt/I1_opt:.12e}\n\n"
        )

        f.write(
            "Higher-order intensity ratios\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        for (
            order,
            dr,
            obr
        ) in zip(
            ratio_orders,
            direct_ratios,
            optimized_ratios
        ):

            f.write(
                f"Order {order:2d}: "
                f"Direct In/I1 = {dr:.12e}, "
                f"DBS In/I1 = {obr:.12e}\n"
            )

    print("\n" + "=" * 78)
    print("FINISHED")
    print("=" * 78)

    print(
        "All output files were written to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":
    main()
