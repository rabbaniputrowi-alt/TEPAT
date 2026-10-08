---
version: alpha
name: TEPAT
description: Daylit inspection workspace with petrol navigation and teal emphasis.
colors:
  primary: "#087f82"
  primary-hover: "#06696c"
  canvas: "#f3f7f7"
  paper: "#ffffff"
  ink: "#153e43"
  muted: "#587078"
  border: "#dce6e7"
  soft: "#e4f3f1"
  nav: "#123b40"
  nav-muted: "#bad3d4"
  nav-selected: "#b7e9df"
  nav-selected-ink: "#133e41"
  capacity: "#e5f0ef"
  chart-secondary: "#72aaa9"
  chart-warning: "#d48b4a"
  chart-baseline: "#98aeb5"
typography:
  headline:
    fontFamily: "Public Sans, Segoe UI, sans-serif"
    fontSize: "2rem"
    fontWeight: 760
    lineHeight: 1.18
    letterSpacing: "-0.035em"
  section:
    fontFamily: "Public Sans, Segoe UI, sans-serif"
    fontSize: "1.1rem"
    fontWeight: 700
    letterSpacing: "-0.025em"
  metric:
    fontFamily: "Public Sans, Segoe UI, sans-serif"
    fontSize: "1.7rem"
    fontWeight: 750
    letterSpacing: "-0.025em"
  navigation:
    fontFamily: "Public Sans, Segoe UI, sans-serif"
    fontSize: "0.85rem"
    fontWeight: 550
  metric-label:
    fontFamily: "Public Sans, Segoe UI, sans-serif"
    fontSize: "0.77rem"
    fontWeight: 600
rounded:
  navigation: "7px"
  control: "8px"
  table: "9px"
  container: "10px"
  pill: "100px"
spacing:
  compact: "0.5rem"
  group: "1rem"
  section: "2rem"
components:
  workspace:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "{colors.paper}"
  navigation:
    backgroundColor: "{colors.nav}"
    textColor: "{colors.nav-muted}"
    typography: "{typography.navigation}"
  navigation-selected:
    backgroundColor: "{colors.nav-selected}"
    textColor: "{colors.nav-selected-ink}"
    rounded: "{rounded.navigation}"
  metric:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.container}"
    padding: "1rem 1.1rem"
  metric-emphasis:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.paper}"
    rounded: "{rounded.container}"
  caption:
    textColor: "{colors.muted}"
  divider:
    backgroundColor: "{colors.border}"
  soft-status:
    backgroundColor: "{colors.soft}"
  capacity-panel:
    backgroundColor: "{colors.capacity}"
---

# Design System: TEPAT

## Overview

**Creative North Star: "Daylit inspection desk"**

The implemented system is a clear office workspace, not a promotional landing page. A deep petrol rail anchors navigation; pale surfaces keep evidence readable; teal marks selection and the primary action. Typography and data hierarchy carry the identity rather than emoji or decorative animation.

**Key Characteristics:**
- Public Sans with tabular numerals for money and ranks.
- Left-aligned evidence, compact controls, flat surfaces.
- A single high-emphasis summary cell; other metrics remain quiet.
- Native Streamlit controls with visible interaction states.

This document records the shipped implementation. `app/assets/workspace.css`, `.streamlit/config.toml`, and chart settings in `app/streamlit_app.py` remain the executable sources. Surface-specific composition lives in `.impeccable/surfaces/workspace.md`.

`tokens.json` is the CLI-generated DTCG export. The exporter currently omits the unitless headline lineHeight; the 1.18 value in this frontmatter and the stylesheet remains authoritative. The app does not consume the export.

## Colors

Primary teal is used for buttons, selected data, and the first summary cell. Petrol is reserved for navigation. Canvas, paper, border, ink, and muted text form the reading surface. Pale mint indicates selected navigation and synthetic-data status. Amber separates secondary chart series; it is not an extra brand accent.

**The Evidence Rule.** Color distinguishes state and comparison; labels carry the meaning. Do not make the user infer compliance from color alone.

## Typography

Public Sans is embedded from the local variable Latin WOFF2, with Segoe UI and sans-serif fallbacks. Provenance and OFL license are in `app/assets/FONT-SOURCE.txt` and `FONT-LICENSE.txt`; there is no runtime Google Fonts request.

The headline, section, metric, navigation, and label roles use the frontmatter tokens. Metric numerals are tabular. On smaller screens, headlines reduce to 1.6rem and summary values to 1.25rem. Mobile metric labels wrap instead of hiding their meaning behind ellipses.

## Layout

The desktop rail is 264px wide. Main content has a 1600px maximum width and 2rem vertical / 2.6rem horizontal padding. Gaps group related controls; evidence remains left aligned. The queue table and first-case preview have a 3:1 split; case evidence and charts have a 3:2 split.

At 1100px, side padding and metric sizes tighten. At 760px, content columns stack, summaries form a two-column grid, and navigation uses the native expandable sidebar. Mobile main padding leaves room for that control. Keep wide data inside its table scroller rather than expanding the whole page.

## Elevation & Depth

Surfaces are flat: border, spacing, and tonal separation convey structure. The system adds no decorative shadow or blurred glass layer. Native popover behavior is preserved.

## Shapes

Containers and tables have gently curved corners; compact controls are slightly tighter. Status pills use a fully rounded silhouette. The small authored focus-target SVG is the only brand mark; Material icons provide native browser/app affordances.

## Components

### Navigation

Plain Indonesian labels on the petrol rail. Mint fill and stronger type identify the current view. Hover changes surface tone. Keyboard focus receives an outline. The hidden radio input remains accessible; only its decorative radio dot is concealed.

### Summary

White metric cells with border and compact padding. Only the first cell receives solid teal. On mobile, two columns preserve scanability; labels wrap and values remain legible.

### Buttons and fields

Primary actions use teal with a darker hover state. Buttons, text inputs, and select controls have a minimum 44px height. Preserve native validation, help, select, and disabled behavior. "Buka kasus" transfers the ranked employer selection into the case view; "Kirim laporan demo" describes a local-only action.

### Tables and charts

Tables retain sorting and their internal horizontal scroller. Probability uses a labelled progress cell. Altair charts use Public Sans, light grids, teal for TEPAT, subdued comparison series, and amber where a second category must be separated. Tooltips carry exact data values.

### Evidence and notices

Reasons use simple separated rows with a legal-basis line, not icon tiles. Documents are an ordinary list. Benchmarks keep their single-seed synthetic warning; worker checks keep their demo disclosure. The empty queue explains which filters to change.

## Do's and Don'ts

### Do
- Do keep synthetic-data disclosure visible and human decisions explicit.
- Do use sentence-case Indonesian labels and actions that name the result.
- Do retain keyboard focus, native sidebar controls, and mobile wrapping.
- Do use only short state transitions; remove them under reduced motion.

### Don't
- Don't replace the evidence workspace with a marketing hero or decorative feature grid.
- Don't use emoji as navigation, status, or decoration.
- Don't hide Streamlit's entire toolbar: it also contains mobile sidebar access.
- Don't change scoring, legal meaning, or benchmark claims as a styling task.
