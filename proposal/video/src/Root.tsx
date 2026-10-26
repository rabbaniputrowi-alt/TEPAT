import { Composition } from "remotion";
import { Spike } from "./Spike";
import { FPS, H, W } from "./tokens";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition id="Spike" component={Spike} durationInFrames={4 * FPS} fps={FPS} width={W} height={H} />
    </>
  );
};
