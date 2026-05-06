"""yomitoku の Gradio ベース GUI。

起動: `yomitoku_gui` (エントリーポイント) または `python -m yomitoku.cli.gui`。

オプション extra `gui` が必要: `pip install yomitoku[gui]`。
"""

import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from ..data.functions import load_epub, load_image, load_pdf
from ..document_analyzer import DocumentAnalyzer
from ..utils.searchable_epub import create_searchable_epub
from ..utils.searchable_pdf import create_searchable_pdf

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "gradio is not installed. Install with: pip install yomitoku[gui]"
    ) from e


_ANALYZER = None


def _get_analyzer(device: str, lite: bool) -> DocumentAnalyzer:
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER

    configs = {}
    if lite:
        configs = {
            "ocr": {
                "text_recognizer": {"model_name": "parseq-tiny"},
            },
        }
        if device == "cpu" or not torch.cuda.is_available():
            configs["ocr"]["text_detector"] = {"infer_onnx": True}

    _ANALYZER = DocumentAnalyzer(
        configs=configs,
        visualize=False,
        device=device,
        ignore_meta=False,
        reading_order="auto",
    )
    return _ANALYZER


def _process(
    file_obj,
    output_format: str,
    device: str,
    lite: bool,
    max_pages: int,
    progress=None,
) -> Tuple[str, str]:
    """アップロードファイルに yomitoku を実行し (出力パス, ログ) を返す。"""
    if file_obj is None:
        return None, "No file uploaded."

    src = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    suffix = src.suffix.lower().lstrip(".")
    if suffix not in {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "pdf", "epub"}:
        return None, f"Unsupported input format: .{suffix}"

    tmpdir = Path(tempfile.mkdtemp(prefix="yomitoku-gui-"))
    log: List[str] = [
        f"Input: {src.name} ({suffix})",
        f"Output format: {output_format}",
    ]

    analyzer = _get_analyzer(device, lite)
    log.append(f"Device: {device}, Lite: {lite}")

    if progress is not None:
        progress(0.05, desc="Loading input")

    book = None
    if suffix == "epub":
        book = load_epub(src)
        imgs = list(book)
        log.append(f"EPUB pages: {len(book.page_refs)} total, {len(imgs)} OCR-eligible")
        if output_format != "searchable-epub":
            return None, (
                "EPUB input currently only supports 'searchable-epub' output."
            )
    elif suffix == "pdf":
        imgs = list(load_pdf(src))
        log.append(f"PDF pages: {len(imgs)}")
    else:
        imgs = load_image(src)
        log.append(f"Image pages: {len(imgs)}")

    if max_pages > 0 and len(imgs) > max_pages:
        log.append(f"Limiting to first {max_pages} pages")
        imgs = imgs[:max_pages]
        if book is not None and len(imgs) < len(book):
            log.append(
                "Note: searchable-epub with --max_pages truncates "
                "the eligible-page list and may produce unexpected output."
            )

    docs = []
    n = max(len(imgs), 1)
    for i, img in enumerate(imgs):
        if progress is not None:
            progress(0.1 + 0.8 * (i / n), desc=f"OCR page {i + 1}/{n}")
        result, _, _ = analyzer(img)
        docs.append(result)

    if progress is not None:
        progress(0.92, desc="Writing output")

    if output_format == "searchable-pdf":
        out = tmpdir / f"{src.stem}.pdf"
        pil_images = [Image.fromarray(img[:, :, ::-1]) for img in imgs]
        create_searchable_pdf(pil_images, docs, output_path=str(out))
    elif output_format == "searchable-epub":
        if book is None:
            return None, "searchable-epub requires .epub input."
        out = tmpdir / f"{src.stem}.epub"
        create_searchable_epub(book, docs, output_path=str(out))
    elif output_format == "json":
        out = tmpdir / f"{src.stem}.json"
        docs[0].to_json(str(out), encoding="utf-8")
    elif output_format == "html":
        out = tmpdir / f"{src.stem}.html"
        docs[0].to_html(str(out), encoding="utf-8")
    elif output_format == "markdown":
        out = tmpdir / f"{src.stem}.md"
        docs[0].to_markdown(str(out), encoding="utf-8")
    else:
        return None, f"Unknown output format: {output_format}"

    log.append(f"Wrote: {out.name}")
    if progress is not None:
        progress(1.0, desc="Done")
    return str(out), "\n".join(log)


def build_demo() -> "gr.Blocks":
    with gr.Blocks(title="YomiToku GUI") as demo:
        gr.Markdown(
            "# YomiToku\nJapanese Document AI — OCR + layout + searchable output."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Input (image / PDF / EPUB)",
                    file_types=[
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tif",
                        ".tiff",
                        ".pdf",
                        ".epub",
                    ],
                )
                fmt = gr.Radio(
                    choices=[
                        "json",
                        "html",
                        "markdown",
                        "searchable-pdf",
                        "searchable-epub",
                    ],
                    value="json",
                    label="Output format",
                )
                device = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    label="Device",
                )
                lite = gr.Checkbox(value=False, label="Lite mode (CPU-friendly)")
                max_pages = gr.Number(
                    value=0,
                    precision=0,
                    label="Max pages (0 = all)",
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=1):
                file_out = gr.File(label="Result")
                log_out = gr.Textbox(label="Log", lines=12)

        def _run(file_obj, fmt, device, lite, max_pages, progress=gr.Progress()):
            return _process(
                file_obj,
                output_format=fmt,
                device=device,
                lite=lite,
                max_pages=int(max_pages or 0),
                progress=progress,
            )

        run_btn.click(
            _run,
            inputs=[file_in, fmt, device, lite, max_pages],
            outputs=[file_out, log_out],
            concurrency_limit=1,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the yomitoku Gradio GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="create a public Gradio link"
    )
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
