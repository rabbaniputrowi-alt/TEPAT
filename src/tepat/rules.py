"""Fixed legal rules, each traceable to an Evidence Bank row (see README).

Nothing here is learned. These are the regulations TEPAT encodes on purpose.
"""
from __future__ import annotations

DAYS_PER_MONTH = 30          # simplification: 30-day months, day 0 = 1 Oct 2024
HORIZON_MONTHS = 24          # Oct 2024 .. Sep 2026
HORIZON_DAYS = DAYS_PER_MONTH * HORIZON_MONTHS
SCORING_DAY = HORIZON_DAYS - 1

KS_RATE = 0.05               # E04: 5% of wage (4% employer, 1% worker)
WAGE_CAP = 12_000_000        # E05 / E05c: Perpres 59/2024 Pasal 32(1)
DUE_DAY_OF_MONTH = 9         # E40: iuran due by the 10th (0-indexed day 9)
LEGAL_REG_LAG_DAYS = 10      # E39: PP 86/2013 Pasal 3(4) = 7 working days ~= 10 calendar days
PHK_FREE_DAYS = 180          # E44: up to 6 months of free cover after PHK (Pasal 27)
ARREARS_MAX_MONTHS = 24      # E55: reactivation needs <=24 months of arrears (Pasal 42(3))
SMALL_CLASSES = ("micro", "small")  # E05c: Pasal 32(4) floor exemption


def month_of(day: int) -> int:
    return day // DAYS_PER_MONTH


def year_of_month(month: int) -> int:
    """Month 0 = Oct 2024; months 3..14 = 2025; 15..23 = 2026. Pre-horizon clips to 2024."""
    if month < 0:
        return 2024
    return 2024 + (month + 9) // 12


def year_of_day(day: int) -> int:
    return year_of_month(month_of(day))


def wage_base(wage: float, umk: float, umkm_class: str) -> float:
    """Iuran base: floor = UMK (E05), except micro/small employers (E05c), cap = Rp12M."""
    floor = 0.0 if umkm_class in SMALL_CLASSES else umk
    return min(max(wage, floor), WAGE_CAP)


def iuran_ks(wage: float, umk: float, umkm_class: str) -> float:
    return KS_RATE * wage_base(wage, umk, umkm_class)


def below_umk_is_violation(umkm_class: str) -> bool:
    """A reported wage below UMK is only a red flag for medium/large employers (E05c)."""
    return umkm_class not in SMALL_CLASSES
