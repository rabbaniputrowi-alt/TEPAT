// Render spike: proves fonts, tokens, frame-driven motion and 1080p30 output on this machine.
// One fictional employer card travels across the five template stages. Not the final scene.
import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT } from "./tokens";

const STAGES = ["Input", "Proses", "Analisis AI", "Output", "Tindak lanjut"];

const Stage: React.FC<{ label: string; i: number; active: number }> = ({ label, i, active }) => {
  const on = Math.max(0, 1 - Math.abs(active - i));
  return (
    <div
      style={{
        width: 300,
        height: 120,
        borderRadius: 14,
        border: `2px solid ${on > 0.5 ? C.primary : C.border}`,
        background: on > 0.5 ? C.soft : C.paper,
        color: C.ink,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 34,
        fontWeight: 650,
        opacity: 0.45 + 0.55 * on,
      }}
    >
      {i + 1}. {label}
    </div>
  );
};

export const Spike: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const t = interpolate(frame, [0.4 * fps, durationInFrames - 0.6 * fps], [0, 4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(0.45, 0, 0.2, 1),
  });
  return (
    <AbsoluteFill style={{ background: C.canvas, fontFamily: FONT, color: C.ink }}>
      <div style={{ position: "absolute", top: 120, left: 120, fontSize: 64, fontWeight: 750 }}>
        Alur proses TEPAT
      </div>
      <div style={{ position: "absolute", top: 520, left: 120, display: "flex", gap: 45 }}>
        {STAGES.map((s, i) => (
          <Stage key={s} label={s} i={i} active={t} />
        ))}
      </div>
      <div
        style={{
          position: "absolute",
          top: 360,
          left: 120,
          translate: `${t * 345}px 0px`,
          width: 300,
          padding: "18px 22px",
          borderRadius: 12,
          background: C.nav,
          color: C.paper,
          fontSize: 26,
          lineHeight: 1.3,
        }}
      >
        <div style={{ fontWeight: 700 }}>PT Baru Sejahtera</div>
        <div style={{ color: C.navMuted, fontSize: 22 }}>fiktif, usaha kecil</div>
      </div>
    </AbsoluteFill>
  );
};
