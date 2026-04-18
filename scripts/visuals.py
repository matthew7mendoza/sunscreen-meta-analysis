"""
Styled per RevMan / Cochrane Handbook for Systematic Reviews of Interventions.
Outputs PRISMA 2020 flow diagrams, Forest Plots, Risk of Bias charts,
and Summary of Findings (SoF) tables.

Visuals by Claude Code.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# Publication palette & typography
# ══════════════════════════════════════════════════════════════════════════════
_SERIF = [
    "Palatino Linotype", "Palatino", "Book Antiqua",
    "Georgia", "DejaVu Serif",
]

_DARK = "#1A1A2E"
_MID = "#5D6D7E"
_LIGHT = "#BDC3C7"
_SHADE = "#F5F7FA"
_COBALT = "#1B4F72"

_ROB_COLORS = {
    "Low": "#27AE60",
    "Some concerns": "#F39C12",
    "High": "#C0392B",
}


def _apply_pub_style():
    """Set rcParams for publication-quality serif figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


class BaseVisualizer:
    """Parent class for saving figures."""

    @staticmethod
    def save_plot(filename):
        """Create output directory if needed and save as SVG."""
        folder_path = "figures"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")

        full_path = os.path.join(folder_path, filename)
        plt.savefig(full_path, bbox_inches = "tight", dpi = 300)
        print(f"Saved figure to: {full_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PRISMA 2020 Flow Diagram
# ══════════════════════════════════════════════════════════════════════════════
class PRISMAFlowchart(BaseVisualizer):
    """PRISMA 2020 flowchart adapted for AI-assisted systematic review.

    AI-assisted deep research bypassed traditional title/abstract screening,
    producing a streamlined three-phase flow: Identification -> Eligibility ->
    Included.
    """

    _PHASE_BG = {
        "Identification": "#D6E4F0",
        "Eligibility": "#FDEBD0",
        "Included": "#E8DAEF",
    }

    @staticmethod
    def _arrow(ax, x1, y1, x2, y2, *, main = True):
        """Draw a clean annotation arrow."""
        color = _DARK if main else _MID
        lw = 1.8 if main else 1.3
        ax.annotate(
            "",
            xy = (x2, y2),
            xytext = (x1, y1),
            arrowprops = {
                "arrowstyle": "-|>",
                "color": color,
                "lw": lw,
                "mutation_scale": 16,
            },
            annotation_clip = False,
        )

    def draw(self, save_filename = "prisma_flowchart.svg"):
        """Draw and save the PRISMA 2020 AI-adapted flowchart."""
        _apply_pub_style()
        _, ax = plt.subplots(figsize = (12, 11))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.get_figure().patch.set_facecolor("white")

        box_main = {
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": _DARK,
            "linewidth": 2.2,
        }
        box_excl = {
            "boxstyle": "round,pad=0.50",
            "facecolor": "#FAFAFA",
            "edgecolor": _MID,
            "linewidth": 1.8,
        }
        box_note = {
            "boxstyle": "round,pad=0.40",
            "facecolor": "#FEF9E7",
            "edgecolor": "#D4AC0D",
            "linewidth": 1.5,
        }

        phases = [
            ("Identification", 0.76, 0.97),
            ("Eligibility", 0.34, 0.74),
            ("Included", 0.06, 0.32),
        ]
        for phase, ylo, yhi in phases:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, ylo), 0.09, yhi - ylo,
                boxstyle = "round,pad=0.005",
                facecolor = self._PHASE_BG[phase],
                edgecolor = _LIGHT, linewidth = 0.7, zorder = 0,
            ))
            ax.text(
                0.055, (ylo + yhi) / 2, phase,
                ha = "center", va = "center", fontsize = 13,
                fontweight = "bold", rotation = 90, color = _DARK,
            )

        # Identification
        ax.text(
            0.42, 0.87,
            "Records identified via AI-assisted\n"
            "deep research model mining of\n"
            "full-text databases\n"
            "(PubMed, Academic Search Complete)\n"
            "n = 11",
            ha = "center", va = "center", bbox = box_main, fontsize = 13,
        )

        ax.text(
            0.80, 0.73,
            "AI-assisted deep research model\n"
            "bypassed traditional title /\n"
            "abstract screening phase",
            ha = "center", va = "center", bbox = box_note,
            fontsize = 11, style = "italic", color = _MID,
        )

        # Eligibility
        ax.text(
            0.42, 0.55,
            "Full-text articles assessed\n"
            "for eligibility\n"
            "k = 11",
            ha = "center", va = "center", bbox = box_main, fontsize = 13,
        )

        ax.text(
            0.80, 0.53,
            "Full-text articles excluded (k = 7)\n\n"
            "\u2022 Incorrect Metric / Not measured\n"
            "   in mg/cm\u00b2 (k = 4)\n"
            "\u2022 Incorrect Intervention /\n"
            "   Layering with makeup (k = 1)\n"
            "\u2022 Incorrect Outcome / Measuring\n"
            "   time instead of quantity (k = 1)\n"
            "\u2022 Incorrect Study Design /\n"
            "   In vitro instead of in vivo (k = 1)",
            ha = "center", va = "center", bbox = box_excl, fontsize = 12,
        )

        # Included
        ax.text(
            0.42, 0.19,
            "Studies included in\n"
            "quantitative synthesis\n"
            "(Meta-Analysis)\n"
            "k = 4 studies\n"
            "(N = 116 participants)",
            ha = "center", va = "center", bbox = box_main,
            fontsize = 14, fontweight = "bold",
        )

        # Arrows
        self._arrow(ax, 0.42, 0.810, 0.42, 0.615)
        self._arrow(ax, 0.42, 0.490, 0.42, 0.260)
        self._arrow(ax, 0.545, 0.55, 0.635, 0.55, main = False)

        if save_filename:
            self.save_plot(save_filename)


# ══════════════════════════════════════════════════════════════════════════════
# Forest Plot (RevMan Style)
# ══════════════════════════════════════════════════════════════════════════════
class ForestPlotVisualizer(BaseVisualizer):
    """RevMan-style random-effects forest plot with optional RoB column."""

    def draw(self, group_data, pooled_result, title = "Forest Plot",
             save_filename = None, rob_data = None):
        """Draw and save the forest plot.

        Parameters
        ----------
        group_data : list[dict]
            Per-study results with keys: id, g, se, w.
        pooled_result : dict
            Pooled effect with keys: k, mean, lower, upper, i2, i2_lower,
            i2_upper, tau2.
        title : str
            Figure title.
        save_filename : str, optional
            Output filename.
        rob_data : dict, optional
            Risk of Bias judgments keyed by study_id. When provided, a
            traffic-light column is appended to the right of the plot.
        """
        _apply_pub_style()

        study_ids = [r["id"] for r in group_data]
        effects = [r["g"] for r in group_data]
        se_vals = [r["se"] for r in group_data]
        weights = [r["w"] for r in group_data]
        k = pooled_result["k"]

        total_w = sum(weights)
        weight_pcts = [(w / total_w) * 100 for w in weights]
        lowers = [g - 1.96 * se for g, se in zip(effects, se_vals)]
        uppers = [g + 1.96 * se for g, se in zip(effects, se_vals)]

        max_w = max(weights) if weights else 1
        vis_w = [max(40, (w / max_w) * 180) for w in weights]

        test_lbl, p_val = self._compute_test_stat(pooled_result, k)

        n = len(study_ids)
        _, ax = plt.subplots(figsize = (16, 3.8 + n * 0.8))
        y_pos = np.arange(n)[::-1]

        for i, yc in enumerate(y_pos):
            if i % 2 == 0:
                ax.axhspan(yc - 0.45, yc + 0.45, color = _SHADE, zorder = 0)

        ax.grid(axis = "x", color = _LIGHT, linestyle = ":", linewidth = 0.5, zorder = 1)

        ax.scatter(
            effects, y_pos, marker = "s", s = vis_w,
            color = _COBALT, zorder = 4, linewidths = 0,
        )

        cap = 0.15
        for lo, hi, yc in zip(lowers, uppers, y_pos):
            ax.hlines(yc, lo, hi, color = _DARK, linewidth = 0.9, zorder = 3)
            ax.vlines(
                [lo, hi], yc - cap, yc + cap,
                color = _DARK, linewidth = 0.9, zorder = 3,
            )

        py = -1.5
        ax.fill(
            [pooled_result["lower"], pooled_result["mean"],
             pooled_result["upper"], pooled_result["mean"]],
            [py, py + 0.30, py, py - 0.30],
            color = _DARK, zorder = 4,
        )

        ax.axvline(x = 0, color = _DARK, linestyle = "-", linewidth = 1.0, zorder = 2)

        min_e = min(lowers + [pooled_result["lower"], -0.5])
        max_e = max(uppers + [pooled_result["upper"], 0.5])
        rng = max_e - min_e
        vis_scalar = max(rng, 6.0)

        col_s = min_e - vis_scalar * 0.55
        col_w = max_e + vis_scalar * 0.15
        col_c = max_e + vis_scalar * 0.40

        rob_domains_full = [
            "Domain 1: Randomisation",
            "Domain 2: Deviations",
            "Domain 3: Missing Outcome",
            "Domain 4: Measurement",
            "Domain 5: Selective Reporting",
        ]
        rob_short = ["D1", "D2", "D3", "D4", "D5"]

        if rob_data:
            col_rob = col_c + vis_scalar * 0.30
            rob_sp = vis_scalar * 0.08
            right_edge = col_rob + len(rob_short) * rob_sp + vis_scalar * 0.10
        else:
            right_edge = col_c + vis_scalar * 0.25

        ax.set_xlim(col_s, right_edge)

        hdr_y = max(y_pos) + 1.2
        ax.text(
            col_s, hdr_y, "Study or Subgroup",
            fontweight = "bold", ha = "left", va = "center", fontsize = 10,
        )
        ax.text(
            col_w, hdr_y, "Weight",
            fontweight = "bold", ha = "center", va = "center", fontsize = 10,
        )
        ax.text(
            col_c, hdr_y, "Hedges' g [95% CI]",
            fontweight = "bold", ha = "center", va = "center", fontsize = 10,
        )
        ax.text(
            col_c, hdr_y - 0.50, "IV, Random, 95% CI",
            ha = "center", va = "center", fontsize = 8.5, color = _MID,
        )

        if rob_data:
            mid_rob = col_rob + 2 * rob_sp
            ax.text(
                mid_rob, hdr_y + 0.5, "Risk of Bias",
                ha = "center", va = "center", fontsize = 9, fontweight = "bold",
            )
            for j, d in enumerate(rob_short):
                ax.text(
                    col_rob + j * rob_sp, hdr_y, d,
                    ha = "center", va = "center", fontsize = 7.5, color = _MID,
                )

        rule_kw = {"color": _DARK, "xmin": 0.01, "xmax": 0.99}
        ax.axhline(y = hdr_y + 0.70, linewidth = 1.5, **rule_kw)
        ax.axhline(y = hdr_y - 0.85, linewidth = 0.8, **rule_kw)
        ax.axhline(y = py + 0.75, linewidth = 0.8, **rule_kw)
        ax.axhline(y = py - 1.30, linewidth = 1.5, **rule_kw)

        for i, yc in enumerate(y_pos):
            label = study_ids[i].replace("_", " ")
            ax.text(col_s, yc, label, ha = "left", va = "center", fontsize = 9.5)
            ax.text(
                col_w, yc, f"{weight_pcts[i]:.1f}%",
                ha = "center", va = "center", fontsize = 9.5,
            )
            ci_txt = f"{effects[i]:.2f} [{lowers[i]:.2f}, {uppers[i]:.2f}]"
            ax.text(col_c, yc, ci_txt, ha = "center", va = "center", fontsize = 9.5)

            if rob_data and study_ids[i] in rob_data:
                for j, domain in enumerate(rob_domains_full):
                    judgment = rob_data[study_ids[i]].get(domain, "")
                    color = _ROB_COLORS.get(judgment, "gray")
                    ax.scatter(
                        col_rob + j * rob_sp, yc,
                        marker = "o", s = 55, color = color,
                        edgecolors = "white", linewidths = 0.5, zorder = 5,
                    )

        ax.text(
            col_s, py, "Total (95% CI)",
            fontweight = "bold", ha = "left", va = "center", fontsize = 10,
        )
        ax.text(
            col_w, py, "100.0%",
            fontweight = "bold", ha = "center", va = "center", fontsize = 10,
        )
        p_txt = (
            f"{pooled_result['mean']:.2f} "
            f"[{pooled_result['lower']:.2f}, {pooled_result['upper']:.2f}]"
        )
        ax.text(
            col_c, py, p_txt,
            fontweight = "bold", ha = "center", va = "center", fontsize = 10,
        )

        p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
        het = (
            f"Heterogeneity: \u03c4\u00b2 = {pooled_result['tau2']:.2f};  "
            f"I\u00b2 = {pooled_result['i2']:.1f}% "
            f"[95% CI: {pooled_result.get('i2_lower', 0):.1f}%, "
            f"{pooled_result.get('i2_upper', 0):.1f}%]"
        )
        tst = f"Test for overall effect: {test_lbl}  (P {p_str})"

        ax.text(
            col_s, py - 0.65, het,
            fontsize = 8.5, ha = "left", va = "center", color = _MID,
        )
        ax.text(
            col_s, py - 1.00, tst,
            fontsize = 8.5, ha = "left", va = "center", color = _MID,
        )

        lbl_y = py - 1.65
        ax.text(
            -0.05, lbl_y, "\u2190 Favours Single Application",
            ha = "right", va = "top", fontsize = 8.5, style = "italic", color = _MID,
        )
        ax.text(
            0.05, lbl_y, "Favours Double Application \u2192",
            ha = "left", va = "top", fontsize = 8.5, style = "italic", color = _MID,
        )

        ax.set_ylim(bottom = py - 2.2)
        ax.set_yticks([])
        ax.set_xlabel(
            "Standardized Mean Difference (Hedges' g)",
            labelpad = 22, fontweight = "bold",
        )
        ax.set_title(title, pad = 18, fontweight = "bold", fontsize = 12)

        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.spines["bottom"].set_color(_DARK)

        t_lo = np.floor(min_e * 2) / 2
        t_hi = np.ceil(max_e * 2) / 2
        ax.set_xticks(np.arange(t_lo, t_hi + 0.01, 0.5))
        ax.tick_params(axis = "x", color = _DARK, length = 3.5, width = 0.6)
        ax.spines["bottom"].set_bounds(t_lo, t_hi)

        if save_filename:
            self.save_plot(save_filename)

    @staticmethod
    def _compute_test_stat(pooled_result, k):
        """Return (label, p_value) for the overall test statistic."""
        if k > 1:
            t_crit = stats.t.ppf(0.975, df = k - 1)
            kh_se = (pooled_result["upper"] - pooled_result["mean"]) / t_crit
            t_stat = pooled_result["mean"] / kh_se if kh_se else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df = k - 1))
            return f"t = {t_stat:.2f}", p_val

        std_se = (pooled_result["upper"] - pooled_result["mean"]) / 1.96
        z_stat = pooled_result["mean"] / std_se if std_se else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return f"Z = {z_stat:.2f}", p_val


# ══════════════════════════════════════════════════════════════════════════════
# Risk of Bias 2 (RoB 2) Visualizations
# ══════════════════════════════════════════════════════════════════════════════
class RiskOfBiasVisualizer(BaseVisualizer):
    """Cochrane RoB 2 traffic light and weighted summary bar charts."""

    def __init__(self, rob_judgments):
        self.rob_data = rob_judgments
        self.studies = list(self.rob_data.keys())
        self.domains = [
            "Domain 1: Randomisation",
            "Domain 2: Deviations",
            "Domain 3: Missing Outcome",
            "Domain 4: Measurement",
            "Domain 5: Selective Reporting",
        ]
        self.color_map = _ROB_COLORS
        self._symbols = {
            "Low": ("+", "white"),
            "Some concerns": ("?", _DARK),
            "High": ("\u2013", "white"),
        }

    def _overall_judgment(self, study):
        """Derive RoB 2 overall judgment (worst-case across domains)."""
        judgments = [self.rob_data[study][d] for d in self.domains]
        if "High" in judgments:
            return "High"
        if "Some concerns" in judgments:
            return "Some concerns"
        return "Low"

    def _draw_circle(self, ax, x, y, judgment, radius = 0.33):
        """Render a single traffic-light circle with Cochrane symbol."""
        color = self.color_map.get(judgment, "gray")
        symbol, txt_color = self._symbols.get(judgment, ("?", "black"))
        ax.add_patch(mpatches.Circle(
            (x, y), radius,
            facecolor = color, edgecolor = "white",
            linewidth = 1.0, zorder = 2,
        ))
        ax.text(
            x, y, symbol,
            va = "center", ha = "center",
            fontsize = 13, fontweight = "bold", color = txt_color, zorder = 3,
        )

    def draw_traffic_light_plot(self, save_filename = "rob_traffic_light.svg"):
        """Draw and save the RoB 2 traffic light plot with Overall column."""
        _apply_pub_style()

        n_s = len(self.studies)
        n_d = len(self.domains)
        all_cols = self.domains + ["Overall"]
        n_cols = len(all_cols)

        _, ax = plt.subplots(figsize = (10.5, n_s * 1.0 + 2.5))
        ax.set_xlim(0, n_cols + 3.2)
        ax.set_ylim(-1.2, n_s + 1.0)
        ax.axis("off")
        ax.get_figure().patch.set_facecolor("white")

        for xi, label in enumerate(all_cols):
            ax.text(
                xi + 3.5, n_s + 0.55, label,
                rotation = 45, ha = "left", va = "bottom",
                fontsize = 8.5, color = _DARK,
            )

        ax.axhline(
            y = n_s + 0.15, color = _DARK, linewidth = 1.0,
            xmin = 0.01, xmax = 0.99,
        )

        for yi, study in enumerate(reversed(self.studies)):
            yc = yi
            if yi % 2 == 0:
                ax.axhspan(yc, yc + 1, color = _SHADE, zorder = 0)

            ax.text(
                0.15, yc + 0.5, study.replace("_", " "),
                va = "center", ha = "left", fontsize = 10,
                fontweight = "bold", color = _DARK,
            )

            for xi, domain in enumerate(self.domains):
                self._draw_circle(
                    ax, xi + 3.5, yc + 0.5,
                    self.rob_data[study][domain],
                )

            self._draw_circle(
                ax, n_d + 3.5, yc + 0.5,
                self._overall_judgment(study),
            )

        ax.axhline(y = 0, color = _DARK, linewidth = 1.0, xmin = 0.01, xmax = 0.99)

        legend_els = [
            Line2D(
                [0], [0], marker = "o", color = "w",
                markerfacecolor = c, markersize = 10,
                label = lbl, markeredgecolor = "white",
            )
            for lbl, c in self.color_map.items()
        ]
        ax.legend(
            handles = legend_els, loc = "lower center",
            bbox_to_anchor = (0.5, -0.10), ncol = 3,
            frameon = False, fontsize = 9,
        )

        if save_filename:
            self.save_plot(save_filename)

    def draw_summary_bar_chart(self, save_filename = "rob_summary_bar.svg"):
        """Draw and save the RoB 2 weighted summary bar chart."""
        _apply_pub_style()
        _, ax = plt.subplots(figsize = (10, 4.5))
        y_pos = np.arange(len(self.domains))

        low_pcts, some_pcts, high_pcts = [], [], []
        for domain in self.domains:
            counts = {"Low": 0, "Some concerns": 0, "High": 0}
            total_w = 0
            for _, data in self.rob_data.items():
                w = data["Weight"]
                counts[data[domain]] += w
                total_w += w
            low_pcts.append(counts["Low"] / total_w * 100)
            some_pcts.append(counts["Some concerns"] / total_w * 100)
            high_pcts.append(counts["High"] / total_w * 100)

        bh = 0.6
        ax.barh(
            y_pos, low_pcts,
            color = self.color_map["Low"], edgecolor = "white",
            linewidth = 0.6, label = "Low risk of bias", height = bh,
        )
        ax.barh(
            y_pos, some_pcts, left = low_pcts,
            color = self.color_map["Some concerns"], edgecolor = "white",
            linewidth = 0.6, label = "Some concerns", height = bh,
        )
        ax.barh(
            y_pos, high_pcts, left = np.add(low_pcts, some_pcts),
            color = self.color_map["High"], edgecolor = "white",
            linewidth = 0.6, label = "High risk of bias", height = bh,
        )

        thresh = 10
        for i, (lo, so, hi) in enumerate(zip(low_pcts, some_pcts, high_pcts)):
            if lo >= thresh:
                ax.text(
                    lo / 2, i, f"{lo:.0f}%",
                    ha = "center", va = "center", fontsize = 8,
                    color = "white", fontweight = "bold",
                )
            if so >= thresh:
                ax.text(
                    lo + so / 2, i, f"{so:.0f}%",
                    ha = "center", va = "center", fontsize = 8,
                    color = _DARK, fontweight = "bold",
                )
            if hi >= thresh:
                ax.text(
                    lo + so + hi / 2, i, f"{hi:.0f}%",
                    ha = "center", va = "center", fontsize = 8,
                    color = "white", fontweight = "bold",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.domains, fontsize = 9.5)
        ax.set_xlabel(
            "Percentage (weighted by study size)",
            fontweight = "bold", labelpad = 8,
        )
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis = "x", color = _LIGHT, linestyle = ":", linewidth = 0.5)

        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.75)

        ax.legend(
            loc = "upper center", bbox_to_anchor = (0.5, -0.18),
            ncol = 3, frameon = False, fontsize = 9,
        )
        plt.tight_layout()

        if save_filename:
            self.save_plot(save_filename)


# ══════════════════════════════════════════════════════════════════════════════
# Summary of Findings (SoF) Table
# ══════════════════════════════════════════════════════════════════════════════
class SummaryOfFindingsTable(BaseVisualizer):
    """GRADE Summary of Findings table for the primary outcome."""

    def draw(self, pooled_result, n_participants = None,
             outcome = "Sunscreen applied (mg/cm\u00b2)",
             comparison = "Double vs. Single Application",
             certainty = None, certainty_reasons = None,
             save_filename = "sof_table.svg"):
        """Draw and save the Summary of Findings table."""
        _apply_pub_style()
        _, ax = plt.subplots(figsize = (14, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.get_figure().patch.set_facecolor("white")

        k = pooled_result["k"]
        mean = pooled_result["mean"]
        lower = pooled_result["lower"]
        upper = pooled_result["upper"]
        i2 = pooled_result["i2"]

        ax.text(
            0.50, 0.95, "Summary of Findings",
            ha = "center", fontsize = 14, fontweight = "bold", color = _DARK,
        )
        ax.text(0.50, 0.90, comparison, ha = "center", fontsize = 11, color = _MID)
        ax.axhline(y = 0.87, color = _DARK, linewidth = 1.5, xmin = 0.02, xmax = 0.98)

        cc = [0.15, 0.34, 0.47, 0.65, 0.82, 0.94]
        hdrs = [
            "Outcome",
            "Studies\n(k)",
            "Participants\n(N)",
            "Effect Estimate\nSMD [95% CI]",
            "Heterogeneity\n(I\u00b2)",
            "Certainty\n(GRADE)",
        ]

        hdr_y = 0.79
        for x, h in zip(cc, hdrs):
            ax.text(
                x, hdr_y, h,
                ha = "center", va = "center",
                fontsize = 9.5, fontweight = "bold", color = _DARK,
            )

        ax.axhline(y = 0.71, color = _DARK, linewidth = 0.8, xmin = 0.02, xmax = 0.98)

        dy = 0.61
        n_str = str(n_participants) if n_participants else "\u2014"
        cert = certainty if certainty else "Not assessed"
        smd = f"{mean:.2f} [{lower:.2f}, {upper:.2f}]"

        grade_symbols = {
            "High": "\u25CF\u25CF\u25CF\u25CF",
            "Moderate": "\u25CF\u25CF\u25CF\u25CB",
            "Low": "\u25CF\u25CF\u25CB\u25CB",
            "Very low": "\u25CF\u25CB\u25CB\u25CB",
        }

        vals = [outcome, str(k), n_str, smd, f"{i2:.1f}%", cert]
        for x, v in zip(cc, vals):
            ax.text(
                x, dy, v,
                ha = "center", va = "center", fontsize = 10, color = _DARK,
            )

        if certainty in grade_symbols:
            ax.text(
                cc[-1], dy - 0.06, grade_symbols[certainty],
                ha = "center", va = "center", fontsize = 11, color = _DARK,
            )

        ax.axhline(y = 0.51, color = _DARK, linewidth = 1.5, xmin = 0.02, xmax = 0.98)

        if mean > 0 and lower > 0:
            interp = (
                "The pooled standardized mean difference indicates a "
                "statistically significant effect\n"
                "favouring double application of sunscreen."
            )
        elif mean > 0:
            interp = (
                "The pooled standardized mean difference suggests a "
                "non-significant trend\n"
                "favouring double application of sunscreen."
            )
        else:
            interp = ""

        if interp:
            ax.text(
                0.50, 0.44, interp,
                ha = "center", va = "center",
                fontsize = 9.5, style = "italic", color = _MID,
            )

        if certainty_reasons:
            reasons = "GRADE downgrades:  " + ";  ".join(certainty_reasons)
            ax.text(
                0.50, 0.36, reasons,
                ha = "center", va = "center", fontsize = 8.5, color = _MID,
            )

        ax.text(
            0.03, 0.25,
            "SMD = Standardized Mean Difference (Hedges' g);  "
            "CI = Confidence Interval;  "
            "IV = Inverse Variance;  "
            "Random = Random-effects model (REML)",
            fontsize = 8, color = _LIGHT, ha = "left",
        )

        if save_filename:
            self.save_plot(save_filename)
