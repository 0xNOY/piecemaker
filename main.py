import functools
import os
import pickle
import shutil
import sys
from base64 import b32encode
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple, Union

import cv2
import gradio as gr
import numpy as np
import requests

sys.path.append(str(Path(__file__).parent.absolute() / "track_anything"))

from track_anything import TrackingAnything

SAM_CHECKPOINT_DICT = {
    "vit_h": {
        "name": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "name": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "name": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
}

VIDEO_SUFFIX = [".mp4", ".avi", ".mov", ".mkv"]

TMPL_SUFFIX = ".tmpl.pkl"


def timed_lru_cache(seconds=60, maxsize=128, typed=False):
    def wrapper_cache(func):
        func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)
        func.ttl = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.ttl

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            with Lock():
                if datetime.utcnow() >= func.expiration:
                    func.cache_clear()
                    func.expiration = datetime.utcnow() + func.ttl
                return func(*args, **kwargs)

        wrapped_func.clear_cache = func.cache_clear
        wrapped_func.cache_info = func.cache_info
        return wrapped_func

    return wrapper_cache


@timed_lru_cache(seconds=60, maxsize=None)
def get_first_frame_from_video(path: str) -> cv2.Mat:
    cap = cv2.VideoCapture(str(path))
    _, frame = cap.read()
    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


@dataclass
class TemplateFrame:
    name: str
    clicks: Tuple[List[Tuple[int, int]], List[int]]
    img: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    painted_img: Optional[np.ndarray] = None
    logit: Optional[np.ndarray] = None


class PieceMaker:
    def __init__(
        self,
        data_dir: Path = Path(__file__).parent / "data",
        sam_model_type: str = "vit_h",
        xmem_checkpoint_name: Optional[str] = None,
        xmem_checkpoint_url: str = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth",
        device: str = "cuda:0",
        port: int = 8080,
    ) -> "PieceMaker":
        self.port = port

        self.data_dir = data_dir

        self.src_dir = self.data_dir / "src"
        self.src_video_dir = self.src_dir / "video"
        self.src_tmpl_dir = self.src_dir / "tmpl"

        self.dst_dir = self.data_dir / "dst"
        self.dst_video_dir = self.dst_dir / "video"
        self.dst_piece_dir = self.dst_dir / "piece"

        checkpoints_dir = self.data_dir / "checkpoints"

        self.data_dir.mkdir(exist_ok=True)
        self.src_dir.mkdir(exist_ok=True)
        self.src_video_dir.mkdir(exist_ok=True)
        self.src_tmpl_dir.mkdir(exist_ok=True)
        self.dst_dir.mkdir(exist_ok=True)
        self.dst_video_dir.mkdir(exist_ok=True)
        self.dst_piece_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)

        if xmem_checkpoint_name is None or xmem_checkpoint_name == "":
            xmem_checkpoint_name = f"xmem-{b32encode(sha256(xmem_checkpoint_url.encode()).digest())[:16].decode()}.pth"

        sam_checkpoint_path = download_checkpoint(
            SAM_CHECKPOINT_DICT[sam_model_type]["url"],
            checkpoints_dir,
            SAM_CHECKPOINT_DICT[sam_model_type]["name"],
        )
        xmem_checkpoint_path = download_checkpoint(
            xmem_checkpoint_url, checkpoints_dir, xmem_checkpoint_name
        )

        self.track_anything = TrackingAnything(
            sam_checkpoint_path,
            xmem_checkpoint_path,
            device,
            sam_model_type,
        )

    def name2src_video_path(self, name: str) -> Path:
        for f in self.src_video_dir.glob(f"{name}.*"):
            if f.suffix in VIDEO_SUFFIX:
                return f

        raise FileNotFoundError(f"Video {name} not found.")

    def store_video(self, name: str, path: Union[str, Path], source_gallery):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video {path} not found.")

        source_gallery = [(i[0]["name"], i[1]) for i in source_gallery]
        thumbnail = get_first_frame_from_video(path)
        source_gallery.append((thumbnail, name))

        shutil.move(path, self.src_video_dir / f"{name}{path.suffix}")

        return *([gr.update(value=None)] * 2), source_gallery

    def load_source_videos_for_gallery(self):
        frames = []
        tmpl_names = set(
            path.name.split(".")[0]
            for path in self.src_tmpl_dir.glob(f"*{TMPL_SUFFIX}")
        )
        for path in self.src_video_dir.glob("*"):
            if not path.suffix in VIDEO_SUFFIX:
                continue

            name = path.name.split(".")[0]

            if name in tmpl_names:
                with open(self.src_tmpl_dir / f"{name}{TMPL_SUFFIX}", "rb") as f:
                    tmpl_state: TemplateFrame = pickle.load(f)
                    frame = tmpl_state.painted_img
            else:
                frame = get_first_frame_from_video(path)

            frames.append((frame, name))

        return frames

    def delete_source(self):
        shutil.rmtree(self.src_dir)
        self.src_dir.mkdir()
        self.src_video_dir.mkdir()
        self.src_tmpl_dir.mkdir()
        return gr.update(value=None)

    def set_tmpl_img(self, srcs, select_data: gr.SelectData):
        tmpl_img = srcs[select_data.index][0]["name"]
        tmpl_name = srcs[select_data.index][1]

        tmpl_state = TemplateFrame(tmpl_name, ([], []))

        if (self.src_tmpl_dir / f"{tmpl_name}{TMPL_SUFFIX}").exists():
            with open(self.src_tmpl_dir / f"{tmpl_name}{TMPL_SUFFIX}", "rb") as f:
                tmpl_state: TemplateFrame = pickle.load(f)
                tmpl_img = tmpl_state.painted_img

        return (
            gr.update(value=tmpl_img, label=tmpl_name),
            tmpl_state,
            [(img[0]["name"], img[1]) for img in srcs],
        )

    def _sam_refine(
        self,
        tmpl_state: TemplateFrame,
    ):
        self.track_anything.samcontroler.sam_controler.reset_image()
        self.track_anything.samcontroler.sam_controler.set_image(tmpl_state.img)

        mask, _, painted_img = self.track_anything.first_frame_click(
            tmpl_state.img,
            np.array(tmpl_state.clicks[0]),
            np.array(tmpl_state.clicks[1]),
        )

        tmpl_state.mask = mask
        tmpl_state.painted_img = painted_img

        return tmpl_state

    def undo_click(self, tmpl_state: TemplateFrame):
        tmpl_state.clicks[0].pop()
        tmpl_state.clicks[1].pop()

        tmpl_state = self._sam_refine(tmpl_state)

        return tmpl_state, gr.update(value=tmpl_state.painted_img)

    def sam_refine(
        self,
        tmpl_img,
        point_prompt,
        tmpl_state: TemplateFrame,
        select_data: gr.SelectData,
    ):
        if tmpl_state.img is None:
            tmpl_state.img = tmpl_img

        point = (
            select_data.index,
            int(point_prompt == "Positive"),
        )

        tmpl_state.clicks[0].append(point[0])
        tmpl_state.clicks[1].append(point[1])

        tmpl_state = self._sam_refine(tmpl_state)

        return tmpl_state, gr.update(value=tmpl_state.painted_img)

    def add_to_queue(self, tmpl_state: TemplateFrame, queue: List[TemplateFrame]):
        with open(self.src_tmpl_dir / f"{tmpl_state.name}{TMPL_SUFFIX}", "wb") as f:
            pickle.dump(tmpl_state, f)

        queue.append(tmpl_state)

        queue_gallery = []
        for tmpl_state in queue:
            queue_gallery.append((tmpl_state.painted_img, tmpl_state.name))

        return (gr.update(value=None, label=None), queue, queue_gallery)

    def make_pieces(
        self,
        queue: List[TemplateFrame],
        max_short_side_size: int,
        max_fps: int,
        sequential_num: bool,
    ):
        n = len(queue)
        for i, tmpl_state in enumerate(queue):
            print(f"Making Pieces: {tmpl_state.name} ({i+1}/{n})")

            if tmpl_state.mask is None:
                continue

            video_path = self.name2src_video_path(tmpl_state.name)
            cap = cv2.VideoCapture(str(video_path))

            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            max_fps = min(max_fps, fps)
            fps_ratio = int(fps / max_fps)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            short_side = min(height, width)
            max_short_side_size = min(max_short_side_size, short_side)
            scale = max_short_side_size / short_side
            j = -1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                j += 1
                if j % fps_ratio != 0:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if scale != 1:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                frames.append(frame)

            cap.release()

            tmpl_mask = cv2.resize(
                tmpl_state.mask,
                (frames[0].shape[1], frames[0].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            self.track_anything.xmem.clear_memory()
            masks, _, _ = self.track_anything.generator(frames, tmpl_mask)

            data_name = tmpl_state.name
            if sequential_num:
                num = str(i).zfill(len(str(n)))
                data_name = f"{num}_{data_name}"

            piece_dir = self.dst_piece_dir / data_name
            piece_dir.mkdir(exist_ok=True)

            len_n_frames = len(str(len(frames)))
            for j, (frame, mask) in enumerate(zip(frames, masks)):
                num = str(j).zfill(len_n_frames)
                piece = frame.copy()
                bound_rect = cv2.boundingRect(mask)
                piece = piece[
                    bound_rect[1] : bound_rect[1] + bound_rect[3],
                    bound_rect[0] : bound_rect[0] + bound_rect[2],
                ]

                if (
                    piece is None
                    or len(piece) < 1
                    or piece.shape[0] < 1
                    or piece.shape[1] < 1
                ):
                    continue

                cv2.imwrite(str(piece_dir / f"{num}.png"), piece)

        return gr.update(value=None), []

    def build_ui(self):
        with gr.Blocks(title="PieceMaker") as root:
            tmpl_state = gr.State(TemplateFrame("", ([], [])))
            queue = gr.State([])

            with gr.Column():
                with gr.Row():
                    with gr.Column() as box_send_video:
                        gr.Markdown("## Step1: Input Videos")
                        data_name = gr.Textbox(lines=1, label="Data Name")
                        input_video = gr.Video(label="Input Video")
                        btn_send_video = gr.Button("Send", variant="primary")

                    with gr.Column() as box_source_library:
                        gr.Markdown("## Step2: Select Sources")
                        source_gallery = gr.Gallery(
                            self.load_source_videos_for_gallery,
                            label="Source Thumbnails",
                        ).style(
                            columns=4,
                            object_fit="scale-down",
                        )
                        btn_delete_source = gr.Button(
                            "Delete All Sources", variant="stop"
                        )

                with gr.Column() as box_make_tmpl:
                    gr.Markdown("## Step3: Make Template Frames")
                    with gr.Row():
                        radio_point_prompt = gr.Radio(
                            label="Point prompt",
                            choices=["Positive", "Negative"],
                            value="Positive",
                        )
                        btn_undo_click = gr.Button("Undo")
                        btn_clear_clicks = gr.Button("Clear Clicks")
                    img_tmpl_preview = gr.Image(label="Template Frame")
                    btn_add_to_queue = gr.Button("Add to Queue", variant="primary")

                with gr.Row():
                    with gr.Column() as box_queue:
                        gr.Markdown("## Queue")
                        queue_gallery = gr.Gallery(label="Queue Thumbnails").style(
                            columns=4, object_fit="scale-down"
                        )
                        btn_clear_queue = gr.Button("Clear Queue", variant="stop")

                    with gr.Column() as box_make_piece:
                        gr.Markdown("## Step4: Make Pieces")
                        slider_max_short_side_size = gr.Slider(
                            label="Max Short Side Size",
                            minimum=144,
                            maximum=1080,
                            value=480,
                            step=1,
                        )
                        slider_max_fps = gr.Slider(
                            label="Max FPS", minimum=1, maximum=30, value=10, step=1
                        )
                        checkbox_sequential_num = gr.Checkbox(
                            label="Sequential numbering of Dirs", value=True
                        )
                        btn_make_pieces = gr.Button("Make Pieces", variant="primary")

            btn_send_video.click(
                self.store_video,
                [data_name, input_video, source_gallery],
                [data_name, input_video, source_gallery],
            )

            btn_delete_source.click(self.delete_source, outputs=source_gallery)

            source_gallery.select(
                self.set_tmpl_img,
                inputs=[source_gallery],
                outputs=[img_tmpl_preview, tmpl_state, source_gallery],
            )

            btn_undo_click.click(
                self.undo_click,
                inputs=[tmpl_state],
                outputs=[tmpl_state, img_tmpl_preview],
            )

            btn_clear_clicks.click(
                lambda tmpl_state: (
                    TemplateFrame(tmpl_state.name, ([], []), tmpl_state.img),
                    gr.update(value=tmpl_state.img),
                ),
                inputs=[tmpl_state],
                outputs=[tmpl_state, img_tmpl_preview],
            )

            img_tmpl_preview.select(
                self.sam_refine,
                inputs=[img_tmpl_preview, radio_point_prompt, tmpl_state],
                outputs=[tmpl_state, img_tmpl_preview],
            )

            btn_add_to_queue.click(
                self.add_to_queue,
                inputs=[tmpl_state, queue],
                outputs=[img_tmpl_preview, queue, queue_gallery],
            )

            btn_clear_queue.click(
                lambda: (gr.update(value=None), []),
                outputs=[queue_gallery, queue],
            )

            btn_make_pieces.click(
                self.make_pieces,
                inputs=[
                    queue,
                    slider_max_short_side_size,
                    slider_max_fps,
                    checkbox_sequential_num,
                ],
                outputs=[queue_gallery, queue],
            )

        return root

    def run(self):
        self.build_ui().launch(server_port=self.port)


if __name__ == "__main__":
    PieceMaker().run()
