// TEPAT design tokens, copied from the app and proposal deck (proposal/TEPAT Proposal.html :root).
// Every scene reads colours and type from here so the video, the deck and the app share one look.
import { loadFont } from "@remotion/fonts";
import { staticFile } from "remotion";

export const C = {
  ink: "#153e43",
  muted: "#587078",
  primary: "#087f82",
  primaryDark: "#06696c",
  nav: "#123b40",
  navMuted: "#bad3d4",
  soft: "#e4f3f1",
  canvas: "#f3f7f7",
  paper: "#ffffff",
  border: "#dce6e7",
  amber: "#d48b4a",
  baseline: "#98aeb5",
} as const;

export const FONT = "Public Sans";

export const fontReady = loadFont({
  family: FONT,
  url: staticFile("public-sans-latin.woff2"),
  weight: "100 900",
});

export const W = 1920;
export const H = 1080;
export const FPS = 30;
