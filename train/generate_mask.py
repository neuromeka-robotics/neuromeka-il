"""Manual SAM2 video labeling, migrated from ETC/train/process/generate_mask.py.

Load raw H5 episodes directly or upload the collector's MP4. Add named masks,
click positive/negative points or box corners, propagate, inspect and save.
"""
import argparse
from pathlib import Path
import sys
import tempfile

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from perception.zero_shot_detection import add_sam2_arguments, build_predictor, inference_context


class MaskSession:
    def __init__(self, predictor, output_dir, camera="wrist", overwrite=False):
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.camera = camera
        self.overwrite = overwrite
        self.temp = None
        self.state = None
        self.frames = []
        self.objects = {}
        self.masks = {}
        self.prompts = {}
        self.exclusions = {}
        self.tracked = False

    def load(self, source):
        self.close()
        source = Path(source)
        self.source = source
        self.objects, self.masks, self.prompts = {}, {}, {}
        self.exclusions = {}
        self.tracked = False
        self.fps = 20.
        if source.suffix.lower() == ".h5":
            with h5py.File(source, "r") as root:
                self.frames = list(root[f"images.rgb.{self.camera}"][:])
            self.output_name = f"{source.stem}_{self.camera}"
        else:
            self.frames = []
            cap = cv2.VideoCapture(str(source))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 20.
            try:
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    self.frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            self.output_name = source.stem
        if not self.frames:
            raise ValueError(f"No RGB frames in {source}")
        self.temp = tempfile.TemporaryDirectory(prefix="visual_servo_masks_")
        for index, rgb in enumerate(self.frames):
            path = Path(self.temp.name) / f"{index:06d}.jpg"
            if not cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise OSError(f"Cannot write {path}")
        with torch.inference_mode(), inference_context():
            self.state = self.predictor.init_state(video_path=self.temp.name,
                offload_video_to_cpu=True, offload_state_to_cpu=True)
        self.add_object("target")
        return self.paint(0)

    def add_object(self, name):
        name = name.strip()
        if not name or "/" in name:
            raise ValueError("Use a nonempty mask name without slashes")
        if name in self.objects:
            return
        if self.state is not None and self.state.get("tracking_has_started", False):
            raise ValueError("Add all objects before tracking, or Clear all prompts before adding another object")
        self.objects[name] = len(self.objects) + 1
        self.masks[name] = {}
        self.tracked = False

    def rename_object(self, old, new):
        new = new.strip()
        if old not in self.objects or not new or "/" in new or new in self.objects:
            raise ValueError("Select a mask and give it a new, unique name")
        self.objects[new] = self.objects.pop(old)
        self.masks[new] = self.masks.pop(old)

    def paint(self, frame_index):
        rgb = self.frames[int(frame_index)].copy()
        for name, masks in self.masks.items():
            if int(frame_index) in masks:
                mask = masks[int(frame_index)]
                rgb[mask] = (rgb[mask] * .5 + np.array([0, 127, 0])).astype(np.uint8)
        return rgb

    def _store(self, frame_index, ids, logits):
        names = {value: key for key, value in self.objects.items()}
        for index, obj_id in enumerate(ids):
            mask = (
                logits[index] > 0).detach().cpu().numpy().reshape(self.frames[0].shape[:2])
            for x0, y0, x1, y1 in self.exclusions.get((int(frame_index), obj_id), []):
                mask[y0:y1, x0:x1] = False
            self.masks[names[obj_id]][int(frame_index)] = mask

    @torch.inference_mode()
    def prompt(self, frame_index, name, kind, xy):
        if self.state is None or name not in self.objects:
            raise ValueError("Load an episode and select a mask first")
        frame_index = int(frame_index)
        obj_id = self.objects[name]
        prompt = self.prompts.setdefault((frame_index, obj_id), {"points": [], "labels": [], "box": [], "negative_box": []})
        if kind == "Negative box":
            if frame_index not in self.masks[name]:
                raise ValueError("Generate a mask on this frame before erasing an area")
            if len(prompt["negative_box"]) == 2:
                prompt["negative_box"] = []
            prompt["negative_box"].append(list(xy))
            if len(prompt["negative_box"]) == 1:
                return self.paint(frame_index), "Click the opposite corner of the area to erase."
            box = np.asarray(prompt["negative_box"], dtype=int)
            x0, y0 = box.min(axis=0)
            x1, y1 = box.max(axis=0)
            self.exclusions.setdefault((frame_index, obj_id), []).append((x0, y0, x1, y1))
            self.masks[name][frame_index][y0:y1, x0:x1] = False
            self.tracked = False
            return self.paint(frame_index), "Area erased on this frame. Track again to update the preview, then Save."
        if kind == "Box":
            if len(prompt["box"]) == 2:
                prompt["box"] = []
            prompt["box"].append(list(xy))
            if len(prompt["box"]) < 2:
                return self.paint(frame_index), "Click the opposite box corner."
        else:
            prompt["points"].append(list(xy))
            prompt["labels"].append(1 if kind == "Positive point" else 0)
        kwargs = {}
        if prompt["points"]:
            kwargs.update(points=np.array(prompt["points"], dtype=np.float32),
                          labels=np.array(prompt["labels"], dtype=np.int32))
        if len(prompt["box"]) == 2:
            box = np.asarray(prompt["box"])
            kwargs["box"] = np.concatenate((box.min(axis=0), box.max(axis=0))).astype(np.float32)
        with inference_context():
            _, ids, logits = self.predictor.add_new_points_or_box(
                inference_state=self.state, frame_idx=frame_index, obj_id=obj_id, **kwargs)
        self._store(frame_index, ids, logits)
        self.tracked = False
        painted = self.paint(frame_index)
        for point, label in zip(prompt["points"], prompt["labels"]):
            cv2.circle(painted, tuple(map(int, point)), 4, (0, 255, 0) if label else (255, 0, 0), -1)
        return painted, "Prompt updated. Track again to propagate this correction."

    @torch.inference_mode()
    def track(self, frame_index):
        if not self.prompts:
            raise ValueError("Add at least one prompt first")
        prompted = {obj_id for (_, obj_id), prompt in self.prompts.items()
                    if prompt["points"] or len(prompt["box"]) == 2}
        if prompted != set(self.objects.values()):
            raise ValueError("Every named mask needs a prompt before tracking")
        with inference_context():
            for reverse in (False, True):
                for frame, ids, logits in self.predictor.propagate_in_video(
                        self.state, start_frame_idx=int(frame_index), reverse=reverse):
                    self._store(frame, ids, logits)
        self.tracked = all(len(masks) == len(self.frames) for masks in self.masks.values())
        if not self.tracked:
            raise RuntimeError("Some frames are unlabeled; add prompts and track again")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        preview = self.output_dir / f"{self.output_name}_preview.mp4"
        height, width = self.frames[0].shape[:2]
        # Bundled ffmpeg produces browser-compatible H.264, without a system ffmpeg install.
        from imageio_ffmpeg import write_frames
        writer = write_frames(str(preview), (width, height), fps=self.fps,
                              codec="libx264", output_params=["-movflags", "+faststart"])
        writer.send(None)
        try:
            for index in range(len(self.frames)):
                writer.send(np.ascontiguousarray(self.paint(index)))
        finally:
            writer.close()
        return self.paint(frame_index), str(preview), "Tracked all frames. Inspect the video/slider, then Save."

    def save(self):
        if not self.tracked:
            raise ValueError("Track all frames after your latest edit before saving")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.output_name}.h5"
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace reviewed masks.")
        temporary = path.with_suffix(".h5.tmp")
        try:
            with h5py.File(temporary, "w") as out:
                for name, masks in self.masks.items():
                    shape = (len(self.frames), *self.frames[0].shape[:2], 1)
                    dataset = out.create_dataset(name, shape=shape, dtype=np.uint8, compression="lzf")
                    for index in range(len(self.frames)):
                        dataset[index, ..., 0] = masks[index]
                out.attrs["camera"] = self.camera
                if self.source.suffix.lower() == ".h5":
                    out.attrs["source_episode"] = self.source.name
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return f"Saved {path}"

    def clear(self, frame_index):
        if self.state is not None:
            self.predictor.reset_state(self.state)
        self.prompts = {}
        self.exclusions = {}
        self.masks = {name: {} for name in self.objects}
        self.tracked = False
        return self.paint(frame_index), "All prompts and masks cleared."

    def close(self):
        if self.state is not None:
            self.predictor.reset_state(self.state)
            self.state = None
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None


def build_app(session, episodes):
    import gradio as gr
    with gr.Blocks(title="SAM2 mask labeling") as app:
        gr.Markdown("# SAM2 mask labeling\nLoad an episode, prompt the target, track, inspect, and save.")
        with gr.Row():
            source = gr.Dropdown(choices=[str(path) for path in episodes], label="Raw H5 episode")
            load_button = gr.Button("Load episode")
            upload = gr.File(label="Or upload H5 / collector MP4", file_types=[".h5", ".mp4"], type="filepath")
        with gr.Row():
            with gr.Column():
                canvas = gr.Image(type="numpy", label="Click to prompt", interactive=False)
                frame = gr.Slider(0, 1, step=1, value=0, label="Frame")
                kind = gr.Radio(["Positive point", "Negative point", "Box", "Negative box"], value="Positive point", label="Prompt")
            with gr.Column():
                selected = gr.Dropdown(choices=[], label="Mask")
                name = gr.Textbox(value="target", label="Mask name")
                with gr.Row():
                    add = gr.Button("Add mask")
                    rename = gr.Button("Rename mask")
                    clear = gr.Button("Clear all prompts")
                preview = gr.Video(label="Tracked preview", format="mp4")
                with gr.Row():
                    track = gr.Button("Track")
                    save = gr.Button("Save masks")
        status = gr.Textbox(label="Status", interactive=False)

        def load(path):
            image = session.load(path)
            return image, gr.update(maximum=len(session.frames)-1, value=0), gr.update(
                choices=list(session.objects), value="target"), None, f"Loaded {len(session.frames)} frames"

        def add_mask(value):
            session.add_object(value)
            return gr.update(choices=list(session.objects), value=value.strip())

        def rename_mask(old, value):
            session.rename_object(old, value)
            return gr.update(choices=list(session.objects), value=value.strip())

        def click(index, obj, prompt_kind, evt: gr.SelectData):
            return session.prompt(index, obj, prompt_kind, evt.index)

        outputs = [canvas, frame, selected, preview, status]
        event_options = {"concurrency_id": "sam2_session", "concurrency_limit": 1}
        load_button.click(load, [source], outputs, **event_options)
        upload.upload(load, [upload], outputs, **event_options)
        frame.release(session.paint, [frame], [canvas], **event_options)
        canvas.select(click, [frame, selected, kind], [canvas, status], **event_options)
        add.click(add_mask, [name], [selected], **event_options)
        rename.click(rename_mask, [selected, name], [selected], **event_options)
        clear.click(session.clear, [frame], [canvas, status], **event_options)
        track.click(session.track, [frame], [canvas, preview, status], **event_options)
        save.click(session.save, [], [status], **event_options)
    # A local, single-user labeling session; serialize all SAM2 state mutations.
    return app.queue(default_concurrency_limit=1)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="syringe_visual_servo")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--port", type=int, default=5800)
    parser.add_argument("--overwrite", action="store_true")
    add_sam2_arguments(parser)
    args = parser.parse_args()
    predictor = build_predictor(args.sam2_checkpoint, args.sam2_config, args.sam2_repo, video=True)
    session = MaskSession(predictor, args.output_dir or root / "data_mask" / args.task,
                          args.camera, args.overwrite)
    episodes = sorted((args.data_dir or root / "data" / args.task).glob("*.h5"))
    try:
        app = build_app(session, episodes)
        app.launch(server_name="127.0.0.1", server_port=args.port, share=False)
    finally:
        session.close()


if __name__ == "__main__":
    main()
