import os
import io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import logging
import imageio

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[

        logging.FileHandler('logs', mode='a'),
        logging.StreamHandler()
    ]

)
log = logging.getLogger(__name__)


# ========= Save the video log =========
def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        print(f"Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"No frames to save. File not created: {filepath}")

def to_png_bytes(img_np):
    img = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

class EpisodeParquetWriter:
    def __init__(self, root, chunk_id=0):
        self.chunk_dir = os.path.join(root, "data", f"chunk-{chunk_id:03d}")
        self.video_dir = os.path.join(root, "videos", f"chunk-{chunk_id:03d}")
        os.makedirs(self.chunk_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self.rows = []
        self.episode_index = 0
        self.frames_agent = []
        self.frames_wrist = []

    def set_episode_metadata(self, metadata):
        """Set per-episode metadata that will be merged into every row.

        Args:
            metadata: dict of column_name -> value pairs.  Values can be
                      scalars or lists (stored as-is in parquet).
        """
        self._episode_metadata = dict(metadata) if metadata else {}

    def add_step(self, state, action, task_index, frame_agent, frame_wrist):

        self.frames_agent.append(frame_agent)
        self.frames_wrist.append(frame_wrist)

        row = {
            "observation.state": state.tolist(),
            "action": action.tolist(),
            "timestamp": float(len(self.frames_agent) * 0.05),
            "frame_index": int(len(self.frames_agent)),
            "episode_index": int(self.episode_index),
            "index": len(self.rows)-1,
            "task_index": int(task_index)
        }
        if hasattr(self, "_episode_metadata"):
            row.update(self._episode_metadata)
        self.rows.append(row)



    def save_episode(self):
        if len(self.rows) == 0:
            return

        df = pd.DataFrame(self.rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        filename = f"episode_{self.episode_index:06d}.parquet"
        pq.write_table(table, os.path.join(self.chunk_dir, filename))

        save_video(self.frames_agent, f"episode_{self.episode_index:06d}.mp4", fps=30, save_dir=f"{self.video_dir}/observation.images.image")
        save_video(self.frames_wrist, f"episode_{self.episode_index:06d}.mp4", fps=30, save_dir=f"{self.video_dir}/observation.images.wrist")

        # clear buffers for next episode
        self.rows = []
        self.frames_agent = []
        self.frames_wrist = []
        self._episode_metadata = {}
        self.episode_index += 1

    def return_episode_index(self):
        return self.episode_index

    def return_video_dir(self):
        return self.video_dir
