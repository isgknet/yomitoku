import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from PIL import Image
import numpy as np
import torch
import pypdfium2
from lxml import etree

from ..constants import (
    MIN_IMAGE_SIZE,
    SUPPORT_INPUT_FORMAT,
    WARNING_IMAGE_SIZE,
)
from ..utils.logger import set_logger

logger = set_logger(__name__)


def validate_image(img: np.ndarray):
    h, w = img.shape[:2]
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        raise ValueError("Image size is too small.")

    if min(h, w) < WARNING_IMAGE_SIZE:
        logger.warning(
            """
            The image size is small, which may result in reduced OCR accuracy. 
            The process will continue, but it is recommended to input images with a minimum size of 720 pixels on the shorter side.
            """
        )


def load_image(image_path: str) -> np.ndarray:
    """
    Open an image file.

    Args:
        image_path (str): path to the image file

    Returns:
        np.ndarray: image data(BGR)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    ext = image_path.suffix[1:].lower()
    if ext not in SUPPORT_INPUT_FORMAT:
        raise ValueError(
            f"Unsupported image format. Supported formats are {SUPPORT_INPUT_FORMAT}"
        )

    if ext == "pdf":
        raise ValueError(
            "PDF file is not supported by load_image(). Use load_pdf() instead."
        )

    if ext == "epub":
        raise ValueError(
            "EPUB file is not supported by load_image(). Use load_epub() instead."
        )

    try:
        img = Image.open(image_path)
    except Exception:
        raise ValueError("Invalid image data.")

    pages = []
    if ext in ["tif", "tiff"]:
        try:
            while True:
                img_arr = np.array(img.copy().convert("RGB"))
                validate_image(img_arr)
                pages.append(img_arr[:, :, ::-1])
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    else:
        img_arr = np.array(img.convert("RGB"))
        validate_image(img_arr)
        pages.append(img_arr[:, :, ::-1])

    return pages


class PdfPageIterator:
    """PDF ページを1ページずつ遅延レンダリングするイテレータ。

    全ページを一括でメモリに展開せず、1 ページずつレンダリングして yield
    することで、数百ページ超の PDF でも OOM を回避する。

    Attributes:
        total_pages (int): PDF の総ページ数。
    """

    def __init__(self, pdf_path: Path, dpi: int = 200):
        self._pdf_path = pdf_path
        self._dpi = dpi

        try:
            doc = pypdfium2.PdfDocument(pdf_path)
            self.total_pages = len(doc)
            doc.close()
        except Exception as e:
            raise ValueError(f"Failed to open the PDF file: {pdf_path}") from e

    def __len__(self):
        return self.total_pages

    def _render_page(self, doc, index: int) -> np.ndarray:
        page = doc[index]
        bitmap = page.render(scale=self._dpi / 72)
        pil_image = bitmap.to_pil()
        return np.array(pil_image.convert("RGB"))[:, :, ::-1]

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(self.total_pages))
            try:
                doc = pypdfium2.PdfDocument(self._pdf_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to open the PDF file: {self._pdf_path}"
                ) from e
            try:
                return [self._render_page(doc, i) for i in indices]
            finally:
                doc.close()

        if isinstance(index, int):
            if index < 0:
                index += self.total_pages
            if not (0 <= index < self.total_pages):
                raise IndexError(f"page index {index} out of range")
            try:
                doc = pypdfium2.PdfDocument(self._pdf_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to open the PDF file: {self._pdf_path}"
                ) from e
            try:
                return self._render_page(doc, index)
            finally:
                doc.close()

        raise TypeError(
            f"indices must be integers or slices, not {type(index).__name__}"
        )

    def __iter__(self):
        try:
            doc = pypdfium2.PdfDocument(self._pdf_path)
        except Exception as e:
            raise ValueError(f"Failed to open the PDF file: {self._pdf_path}") from e

        try:
            for i in range(self.total_pages):
                yield self._render_page(doc, i)
        finally:
            doc.close()


def load_pdf(pdf_path: str, dpi=200) -> PdfPageIterator:
    """
    Load a PDF file and return an iterator that yields page images in BGR format.

    Pages are rendered lazily one at a time to avoid loading all pages into
    memory at once, preventing OOM errors for large PDFs with hundreds of pages.

    Args:
        pdf_path (str): The path to the PDF file to be loaded.
        dpi (int, optional): The resolution for rendering. Defaults to 200.

    Returns:
        PdfPageIterator: An iterator yielding NumPy arrays (BGR format) for each page.
            Has a `total_pages` attribute and supports `len()`.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the file format is not supported or not a valid PDF.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    ext = pdf_path.suffix[1:].lower()
    if ext not in SUPPORT_INPUT_FORMAT:
        raise ValueError(
            f"Unsupported image format. Supported formats are {SUPPORT_INPUT_FORMAT}"
        )

    if ext != "pdf":
        raise ValueError(
            "image file is not supported by load_pdf(). Use load_image() instead."
        )

    return PdfPageIterator(pdf_path, dpi=dpi)


# ---- EPUB ----------------------------------------------------------------

_EPUB_RASTER_EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "gif", "webp"}


def _normalize_zip_path(base_dir: str, href: str) -> str:
    """zip 内部での `href` を `base_dir` 起点に解決する (区切りは "/")。"""
    href = href.split("#", 1)[0]
    if href.startswith("/"):
        parts: List[str] = []
        for p in href.lstrip("/").split("/"):
            if p in ("", "."):
                continue
            if p == ".." and parts:
                parts.pop()
            else:
                parts.append(p)
        return "/".join(parts)

    parts = [p for p in base_dir.split("/") if p] if base_dir else []
    for p in href.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
        else:
            parts.append(p)
    return "/".join(parts)


def _local(name: str) -> str:
    return f"*[local-name()='{name}']"


@dataclass
class EpubPageRef:
    """`load_epub` が記録する spine エントリ単位のメタデータ。

    OCR 対象ページは `skip_reason is None` かつ `image_zip_path` を保持。
    非対象ページも `skip_reason` 付きで記録し、EPUB をそのままラウンドトリップ
    できるようにする。
    """

    xhtml_path: str
    image_zip_path: Optional[str] = None
    image_src_in_xhtml: Optional[str] = None
    image_pixel_size: Optional[Tuple[int, int]] = None
    skip_reason: Optional[str] = None
    eligible_index: Optional[int] = None


@dataclass
class EpubBook:
    """EPUB ファイルへの遅延ビュー。OCR 用の画像ソースと、
    `create_searchable_epub` が再パッケージする際の構造化コンテナを兼ねる。

    イテレーション / `len` / `__getitem__` は OCR 対象ページのみを BGR
    `np.ndarray` として返す。`load_image` / `load_pdf` を呼んでいた既存コードに
    そのまま差し込める。
    """

    src_path: Path
    opf_path: str
    opf_dir: str
    page_refs: List[EpubPageRef] = field(default_factory=list)
    _eligible_refs: List[EpubPageRef] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._eligible_refs = [r for r in self.page_refs if r.skip_reason is None]
        for i, r in enumerate(self._eligible_refs):
            r.eligible_index = i

    def __len__(self) -> int:
        return len(self._eligible_refs)

    def __iter__(self):
        with zipfile.ZipFile(self.src_path, "r") as zf:
            for ref in self._eligible_refs:
                yield self._decode(zf, ref)

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self._eligible_refs)))
            with zipfile.ZipFile(self.src_path, "r") as zf:
                return [self._decode(zf, self._eligible_refs[i]) for i in indices]

        if isinstance(index, int):
            n = len(self._eligible_refs)
            if index < 0:
                index += n
            if not (0 <= index < n):
                raise IndexError(f"page index {index} out of range")
            with zipfile.ZipFile(self.src_path, "r") as zf:
                return self._decode(zf, self._eligible_refs[index])

        raise TypeError(
            f"indices must be integers or slices, not {type(index).__name__}"
        )

    @staticmethod
    def _decode(zf: zipfile.ZipFile, ref: EpubPageRef) -> np.ndarray:
        with zf.open(ref.image_zip_path) as fp:
            data = fp.read()
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Failed to decode image '{ref.image_zip_path}' inside EPUB"
            ) from e
        arr = np.array(pil)[:, :, ::-1]
        validate_image(arr)
        return arr


def _xhtml_image_eligibility(
    xhtml_bytes: bytes,
    xhtml_zip_path: str,
    zf: zipfile.ZipFile,
):
    """spine の XHTML 1 ページを検査し、含まれる画像 (あれば) を OCR 対象に
    すべきか判定する。

    戻り値: (image_zip_path, src_attr, (W, H), skip_reason)。OCR 対象なら
    `skip_reason` が None で他 3 値が埋まる。非対象なら `skip_reason` のみ。
    """
    try:
        root = etree.fromstring(
            xhtml_bytes, parser=etree.XMLParser(recover=True, resolve_entities=False)
        )
    except Exception:
        try:
            root = etree.HTML(xhtml_bytes)
        except Exception:
            return None, None, None, "parse_error"

    if root is None:
        return None, None, None, "parse_error"

    # SVG <image> はベクター埋め込みのため OCR 対象外
    svg_images = root.xpath(f".//{_local('svg')}//{_local('image')}")
    if svg_images:
        return None, None, None, "svg_image"

    imgs = root.xpath(f".//{_local('img')}")
    if len(imgs) == 0:
        return None, None, None, "no_image"
    if len(imgs) > 1:
        return None, None, None, "multiple_images"

    img_el = imgs[0]
    src = img_el.get("src") or img_el.get("{http://www.w3.org/1999/xlink}href")
    if not src:
        return None, None, None, "no_image"

    # body 内テキスト判定: 画像を除外した残りに有意なテキストがあるかを確認
    bodies = root.xpath(f".//{_local('body')}")
    if bodies:
        body_copy = etree.fromstring(etree.tostring(bodies[0]))
        for el in body_copy.xpath(f".//{_local('img')}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
        text = "".join(body_copy.itertext()) or ""
        if text.strip():
            return None, None, None, "text_and_image"

    ext = src.rsplit(".", 1)[-1].lower().split("?")[0]
    if ext not in _EPUB_RASTER_EXTS:
        return None, None, None, "non_raster_image"

    xhtml_dir = "/".join(xhtml_zip_path.split("/")[:-1])
    img_zip_path = _normalize_zip_path(xhtml_dir, src)

    if img_zip_path not in zf.namelist():
        return None, None, None, "image_not_found"

    try:
        with zf.open(img_zip_path) as fp:
            pil = Image.open(io.BytesIO(fp.read()))
            w, h = pil.size
    except Exception:
        return None, None, None, "image_decode_error"

    return img_zip_path, src, (w, h), None


def load_epub(epub_path: str) -> EpubBook:
    """画像ベース EPUB を開いて `EpubBook` ビューを生成する。

    body に単一のラスタ `<img>` のみを持ち、他に有意なテキストが無い spine
    の XHTML だけを OCR 対象として公開する。それ以外のページは `skip_reason`
    付きで `book.page_refs` に残し、`create_searchable_epub` が原本のまま
    コピーできるようにする。

    Args:
        epub_path: .epub ファイルのパス。

    Returns:
        OCR 対象ページごとに BGR の `np.ndarray` を返すイテレータを持つ
        `EpubBook`。
    """
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"File not found: {epub_path}")

    ext = epub_path.suffix[1:].lower()
    if ext not in SUPPORT_INPUT_FORMAT:
        raise ValueError(
            f"Unsupported image format. Supported formats are {SUPPORT_INPUT_FORMAT}"
        )
    if ext != "epub":
        raise ValueError(
            "non-EPUB file is not supported by load_epub(). "
            "Use load_image() or load_pdf() instead."
        )

    try:
        zf = zipfile.ZipFile(epub_path, "r")
    except zipfile.BadZipFile as e:
        raise ValueError(f"Invalid EPUB (not a valid zip): {epub_path}") from e

    try:
        names = set(zf.namelist())
        if "META-INF/container.xml" not in names:
            raise ValueError(
                f"Invalid EPUB (missing META-INF/container.xml): {epub_path}"
            )

        container_bytes = zf.read("META-INF/container.xml")
        container = etree.fromstring(container_bytes)
        rootfiles = container.xpath(f".//{_local('rootfile')}")
        if not rootfiles:
            raise ValueError(f"Invalid EPUB (no rootfile entry): {epub_path}")

        opf_path = rootfiles[0].get("full-path")
        if not opf_path:
            raise ValueError(f"Invalid EPUB (rootfile has no full-path): {epub_path}")

        opf_dir = "/".join(opf_path.split("/")[:-1])
        opf_bytes = zf.read(opf_path)
        opf = etree.fromstring(opf_bytes)

        manifest_items = opf.xpath(f".//{_local('manifest')}/{_local('item')}")
        manifest = {}
        for item in manifest_items:
            iid = item.get("id")
            href = item.get("href")
            if iid and href:
                manifest[iid] = _normalize_zip_path(opf_dir, href)

        spine_refs = opf.xpath(f".//{_local('spine')}/{_local('itemref')}")
        if not spine_refs:
            raise ValueError(f"Invalid EPUB (empty spine): {epub_path}")

        page_refs: List[EpubPageRef] = []
        for itemref in spine_refs:
            idref = itemref.get("idref")
            if not idref or idref not in manifest:
                continue

            xhtml_path = manifest[idref]
            if xhtml_path not in names:
                page_refs.append(
                    EpubPageRef(xhtml_path=xhtml_path, skip_reason="missing_in_zip")
                )
                continue

            xhtml_bytes = zf.read(xhtml_path)
            img_zip_path, src, size, reason = _xhtml_image_eligibility(
                xhtml_bytes, xhtml_path, zf
            )
            if reason is not None:
                page_refs.append(EpubPageRef(xhtml_path=xhtml_path, skip_reason=reason))
                continue

            page_refs.append(
                EpubPageRef(
                    xhtml_path=xhtml_path,
                    image_zip_path=img_zip_path,
                    image_src_in_xhtml=src,
                    image_pixel_size=size,
                )
            )
    finally:
        zf.close()

    book = EpubBook(
        src_path=epub_path,
        opf_path=opf_path,
        opf_dir=opf_dir,
        page_refs=page_refs,
    )

    skipped = [r for r in page_refs if r.skip_reason is not None]
    if skipped:
        logger.info(
            f"EPUB load: {len(book)} eligible page(s), {len(skipped)} skipped "
            f"({', '.join(sorted({r.skip_reason for r in skipped}))})"
        )
    return book


def resize_shortest_edge(
    img: np.ndarray, shortest_edge_length: int, max_length: int
) -> np.ndarray:
    """
    Resize the shortest edge of the image to `shortest_edge_length` while keeping the aspect ratio.
    if the longest edge is longer than `max_length`, resize the longest edge to `max_length` while keeping the aspect ratio.

    Args:
        img (np.ndarray): target image
        shortest_edge_length (int): pixel length of the shortest edge after resizing
        max_length (int): pixel length of maximum edge after resizing

    Returns:
        np.ndarray: resized image
    """

    h, w = img.shape[:2]
    scale = shortest_edge_length / min(h, w)
    if h < w:
        new_h, new_w = shortest_edge_length, int(w * scale)
    else:
        new_h, new_w = int(h * scale), shortest_edge_length

    if max(new_h, new_w) > max_length:
        scale = float(max_length) / max(new_h, new_w)
        new_h, new_w = int(new_h * scale), int(new_w * scale)

    neww = max(int(new_w / 32) * 32, 32)
    newh = max(int(new_h / 32) * 32, 32)

    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    return img


def standardization_image(
    img: np.ndarray, rgb=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize the image data.

    Args:
        img (np.ndarray): target image

    Returns:
        np.ndarray: normalized image
    """
    img = img[:, :, ::-1]
    img = img / 255.0
    img = (img - np.array(rgb)) / np.array(std)
    img = img.astype(np.float32)

    return img


def array_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert the image data to tensor.
    (H, W, C) -> (N, C, H, W)

    Args:
        img (np.ndarray): target image(H, W, C)

    Returns:
        torch.Tensor: (N, C, H, W) tensor
    """
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.as_tensor(img, dtype=torch.float)
    tensor = tensor[None, :, :, :]
    return tensor


def validate_quads(img: np.ndarray, quad: list[list[list[int]]]):
    """
    Validate the vertices of the quadrilateral.

    Args:
        img (np.ndarray): target image
        quads (list[list[list[int]]]): list of quadrilateral

    Raises:
        ValueError: if the vertices are invalid
    """

    h, w = img.shape[:2]
    if len(quad) != 4:
        # raise ValueError("The number of vertices must be 4.")
        return None

    for point in quad:
        if len(point) != 2:
            return None

    quad = np.array(quad, dtype=int)
    x1 = np.min(quad[:, 0])
    x2 = np.max(quad[:, 0])
    y1 = np.min(quad[:, 1])
    y2 = np.max(quad[:, 1])
    h, w = img.shape[:2]

    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
        return None

    return True


def extract_roi_with_perspective(img, quad):
    """
    Extract the word image from the image with perspective transformation.

    Args:
        img (np.ndarray): target image
        polygon (np.ndarray): polygon vertices

    Returns:
        np.ndarray: extracted image
    """
    quad = np.array(quad, dtype=np.int64)

    roi_img = img[
        int(min(quad[:, 1])) : int(max(quad[:, 1])),
        int(min(quad[:, 0])) : int(max(quad[:, 0])),
        :,
    ]

    quad[:, 0] -= int(min(quad[:, 0]))
    quad[:, 1] -= int(min(quad[:, 1]))

    width = np.linalg.norm(quad[0] - quad[1])
    height = np.linalg.norm(quad[1] - quad[2])

    width = int(width)
    height = int(height)
    pts1 = np.float32(quad)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(roi_img, M, (width, height))
    return dst


def rotate_text_image(img, thresh_aspect=2):
    """
    Rotate the image if the aspect ratio is too high.

    Args:
        img (np.ndarray): target image
        thresh_aspect (int): threshold of aspect ratio

    Returns:
        np.ndarray: rotated image
    """
    h, w = img.shape[:2]
    if h > thresh_aspect * w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def resize_with_padding(img, target_size, background_color=(0, 0, 0)):
    """
    Resize the image with padding.

    Args:
        img (np.ndarray): target image
        target_size (int, int): target size
        background_color (Tuple[int, int, int]): background color

    Returns:
        np.ndarray: resized image
    """
    h, w = img.shape[:2]
    scale_w = 1.0
    scale_h = 1.0
    if w > target_size[1]:
        scale_w = target_size[1] / w
    if h > target_size[0]:
        scale_h = target_size[0] / h

    new_w = int(w * min(scale_w, scale_h))
    new_h = int(h * min(scale_w, scale_h))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    canvas[:, :] = background_color

    resized_size = resized.shape[:2]
    canvas[: resized_size[0], : resized_size[1], :] = resized

    return canvas
