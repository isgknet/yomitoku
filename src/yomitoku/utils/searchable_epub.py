"""検索可能 EPUB の生成: 画像ベース EPUB に yomitoku の OCR 結果から作った
透明テキストレイヤを重ねて出力する。

元 EPUB のコンテナ (CSS, TOC, spine, 画像のみ以外のページ) はそのまま保ち、
`load_epub` が OCR 対象と判定したページの body だけを書き換える。
`searchable_pdf.py` と同じ設計で、共通ヘルパ (`_poly2rect`,
`_calc_font_size`, `to_full_width`, `IMAGE_QUALITY_PRESETS`) を再利用する。
"""

import hashlib
import html
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from lxml import etree
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from ..data.functions import EpubBook, EpubPageRef
from ..schemas import DocumentAnalyzerSchema
from .misc import is_contained
from .searchable_pdf import (
    FONT_PATH,
    IMAGE_QUALITY_PRESETS,
    _calc_font_size,
    _poly2rect,
    to_full_width,
)

DEFAULT_FONT_PATH = FONT_PATH

_OPF_NS = "http://www.idpf.org/2007/opf"
_XHTML_NS = "http://www.w3.org/1999/xhtml"

_FONT_FILE_NAME = "MPLUS1p-Medium.ttf"
_CSS_FILE_NAME = "yomitoku-text.css"

_FONT_FACE_CSS = """\
@font-face {
  font-family: "MPLUS1p-Medium";
  src: url("__FONT_HREF__") format("truetype");
}
"""

_TEXT_LAYER_CSS = """\
.yomitoku-page {
  position: relative;
  margin: 0 auto;
  container-type: inline-size;
  container-name: yomitoku-page;
  max-width: 100%;
  height: auto;
}
.yomitoku-page > img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.yomitoku-page .word {
  position: absolute;
  color: transparent;
  font-family: "MPLUS1p-Medium", sans-serif;
  white-space: pre;
  line-height: 1;
  margin: 0;
  padding: 0;
  pointer-events: auto;
  user-select: text;
}
"""


def _rel_path(from_zip_path: str, to_zip_path: str) -> str:
    """zip 内のファイルパス間の相対パス (区切りは "/") を返す。"""
    src_dir = from_zip_path.split("/")[:-1]
    dst_parts = to_zip_path.split("/")
    common = 0
    for a, b in zip(src_dir, dst_parts[:-1]):
        if a != b:
            break
        common += 1
    up = [".."] * (len(src_dir) - common)
    down = dst_parts[common:]
    parts = up + down
    return "/".join(parts) if parts else "."


def _short_hash(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:8]


def _build_words_in_reading_order(
    doc: DocumentAnalyzerSchema,
) -> List:
    """単語をコンテナ (段落 / 表セル / 図中段落) にまとめ、コンテナを
    読み順でソートし、単語のフラットリストを返す。`searchable_pdf.py:117-180`
    と同じロジック。
    """
    containers = []
    for p in doc.paragraphs:
        containers.append(
            {
                "box": p.box,
                "order": p.order if p.order is not None else 1 << 30,
                "sub_order": 0,
                "direction": p.direction or "horizontal",
            }
        )
    for t in doc.tables:
        for cell in t.cells:
            containers.append(
                {
                    "box": cell.box,
                    "order": t.order if t.order is not None else 1 << 30,
                    "sub_order": (cell.row, cell.col),
                    "direction": "horizontal",
                }
            )
    for f in doc.figures:
        for para_idx, p in enumerate(f.paragraphs):
            containers.append(
                {
                    "box": p.box,
                    "order": f.order if f.order is not None else 1 << 30,
                    "sub_order": para_idx,
                    "direction": p.direction or "horizontal",
                }
            )

    containers.sort(key=lambda c: (c["order"], c["sub_order"]))

    seen = set()
    ordered: List = []
    for c in containers:
        bucket = []
        for word in doc.words:
            wid = id(word)
            if wid in seen:
                continue
            wbox = _poly2rect(word.points)
            if is_contained(c["box"], wbox, 0.7):
                bucket.append(word)
        if c["direction"] == "vertical":
            bucket.sort(
                key=lambda w: (-_poly2rect(w.points)[0], _poly2rect(w.points)[1])
            )
        else:
            bucket.sort(
                key=lambda w: (_poly2rect(w.points)[1], _poly2rect(w.points)[0])
            )
        for w in bucket:
            seen.add(id(w))
            ordered.append(w)

    # どのコンテナにも入らなかった単語は元順で末尾追加。読み順は近似でも
    # 選択は可能にしておく。
    for word in doc.words:
        if id(word) not in seen:
            ordered.append(word)

    return ordered


def _emit_word_spans(
    doc: DocumentAnalyzerSchema,
    page_w: int,
    page_h: int,
) -> List[str]:
    """% 座標と `cqw` フォントサイズで配置した
    `<span class="word">` の XHTML 文字列リストを返す。
    """
    if page_w <= 0 or page_h <= 0:
        return []

    spans: List[str] = []
    words = _build_words_in_reading_order(doc)

    for word in words:
        text = word.content or ""
        if not text:
            continue
        bbox = _poly2rect(word.points)
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        direction = getattr(word, "direction", "horizontal") or "horizontal"

        if direction == "vertical":
            text = to_full_width(text)
            font_size_px = _calc_font_size(text, bw, bh)
        else:
            font_size_px = _calc_font_size(text, bh, bw)

        if not font_size_px:
            continue

        font_size_cqw = font_size_px / page_w * 100.0

        if direction == "vertical":
            char_h = bh / max(len(text), 1)
            for j, ch in enumerate(text):
                if not ch.strip():
                    continue
                cx1 = x1 + (bw - font_size_px) / 2
                cy1 = y1 + j * char_h
                left_p = cx1 / page_w * 100.0
                top_p = cy1 / page_h * 100.0
                width_p = font_size_px / page_w * 100.0
                height_p = char_h / page_h * 100.0
                spans.append(
                    f'<span class="word" style="left:{left_p:.4f}%;'
                    f"top:{top_p:.4f}%;"
                    f"width:{width_p:.4f}%;"
                    f"height:{height_p:.4f}%;"
                    f'font-size:{font_size_cqw:.4f}cqw;">'
                    f"{html.escape(ch, quote=True)}</span>"
                )
        else:
            left_p = x1 / page_w * 100.0
            top_p = y1 / page_h * 100.0
            width_p = bw / page_w * 100.0
            height_p = bh / page_h * 100.0
            spans.append(
                f'<span class="word" style="left:{left_p:.4f}%;'
                f"top:{top_p:.4f}%;"
                f"width:{width_p:.4f}%;"
                f"height:{height_p:.4f}%;"
                f'font-size:{font_size_cqw:.4f}cqw;">'
                f"{html.escape(text, quote=True)}</span>"
            )

    return spans


def _rewrite_xhtml(
    xhtml_bytes: bytes,
    ref: EpubPageRef,
    doc: DocumentAnalyzerSchema,
    css_zip_path: str,
) -> bytes:
    """OCR 対象の XHTML 1 ページを書き換え、body を
    `<div class="yomitoku-page">` + 画像 + 透明テキスト span 群に置換する。
    """
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    try:
        root = etree.fromstring(xhtml_bytes, parser=parser)
    except Exception:
        return xhtml_bytes  # parse 失敗時はページ無変更で戻す
    if root is None:
        return xhtml_bytes

    nsmap_ns = root.nsmap.get(None)
    use_xhtml_ns = nsmap_ns == _XHTML_NS

    def _q(name: str) -> str:
        return f"{{{_XHTML_NS}}}{name}" if use_xhtml_ns else name

    head_list = root.xpath(".//*[local-name()='head']")
    body_list = root.xpath(".//*[local-name()='body']")
    if not body_list:
        return xhtml_bytes

    page_w, page_h = ref.image_pixel_size or (0, 0)
    spans = _emit_word_spans(doc, page_w, page_h)

    body = body_list[0]
    body_attrs = dict(body.attrib)
    for child in list(body):
        body.remove(child)
    if body.text:
        body.text = None

    page_div = etree.SubElement(body, _q("div"))
    page_div.set("class", "yomitoku-page")
    page_div.set(
        "style",
        f"width:{page_w}px;aspect-ratio:{page_w}/{page_h};",
    )

    img_el = etree.SubElement(page_div, _q("img"))
    img_el.set("src", ref.image_src_in_xhtml or "")
    img_el.set("alt", "")

    if spans:
        spans_xml = "<g xmlns='" + (_XHTML_NS if use_xhtml_ns else "") + "'>"
        spans_xml += "".join(spans)
        spans_xml += "</g>"
        try:
            wrapper = etree.fromstring(spans_xml)
            for span in list(wrapper):
                page_div.append(span)
        except Exception:
            # OCR 出力に起因する不正 XML が混じった場合、本ページの text レイヤ
            # のみ落とし、book 全体の処理は継続する。
            pass

    if head_list:
        head = head_list[0]
        css_href = _rel_path(ref.xhtml_path, css_zip_path)
        link = etree.SubElement(head, _q("link"))
        link.set("rel", "stylesheet")
        link.set("type", "text/css")
        link.set("href", css_href)

    body.attrib.update(body_attrs)

    out = etree.tostring(
        root,
        xml_declaration=True,
        encoding="utf-8",
        doctype="<!DOCTYPE html>",
    )
    return out


def _patch_opf(
    opf_bytes: bytes, opf_dir: str, css_zip_path: str, font_zip_path: Optional[str]
) -> bytes:
    """新規 CSS (および埋め込みフォント) の manifest item を追記する。
    spine / metadata は変更しない。
    """
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    root = etree.fromstring(opf_bytes, parser=parser)
    if root is None:
        return opf_bytes

    manifest_list = root.xpath(".//*[local-name()='manifest']")
    if not manifest_list:
        return opf_bytes
    manifest = manifest_list[0]

    existing_ids = set()
    existing_hrefs = set()
    for item in manifest.xpath("./*[local-name()='item']"):
        if item.get("id"):
            existing_ids.add(item.get("id"))
        if item.get("href"):
            existing_hrefs.add(item.get("href"))

    nsmap_ns = root.nsmap.get(None)
    use_opf_ns = nsmap_ns == _OPF_NS
    item_qn = f"{{{_OPF_NS}}}item" if use_opf_ns else "item"

    def _unique_id(base: str) -> str:
        if base not in existing_ids:
            return base
        suffix = _short_hash(base, css_zip_path)
        return f"{base}-{suffix}"

    css_href_in_opf = _rel_path(_opf_self_path(opf_dir), css_zip_path)
    if css_href_in_opf not in existing_hrefs:
        css_item = etree.SubElement(manifest, item_qn)
        css_item.set("id", _unique_id("yomitoku-text-css"))
        css_item.set("href", css_href_in_opf)
        css_item.set("media-type", "text/css")

    if font_zip_path is not None:
        font_href = _rel_path(_opf_self_path(opf_dir), font_zip_path)
        if font_href not in existing_hrefs:
            font_item = etree.SubElement(manifest, item_qn)
            font_item.set("id", _unique_id("yomitoku-mplus1p-font"))
            font_item.set("href", font_href)
            font_item.set("media-type", "application/font-sfnt")

    return etree.tostring(root, xml_declaration=True, encoding="utf-8")


def _opf_self_path(opf_dir: str) -> str:
    """`_rel_path` で href を相対化する際の基準として使う OPF の擬似 zip パス。
    ファイル名は意味を持たず、ディレクトリ部だけが効く。"""
    return f"{opf_dir}/_.opf" if opf_dir else "_.opf"


def create_searchable_epub(
    book: EpubBook,
    docs: List[DocumentAnalyzerSchema],
    output_path: str,
    font_path: Optional[str] = None,
    image_quality: str = "high",
    embed_font: bool = True,
):
    """`book` の OCR 対象ページに透明テキストレイヤを追加し、それ以外の zip
    エントリを原本のままコピーして検索可能 EPUB を生成する。

    Args:
        book: `load_epub` の戻り値。
        docs: OCR 対象ページごとの解析結果 (`len(docs) == len(book)`)。
        output_path: 出力先 .epub のパス。
        font_path: フォント幅計算 (reportlab) と、`embed_font=True` 時の EPUB
            への埋め込みに使う TTF。既定はバンドル済みの M+ フォント。
        image_quality: `create_searchable_pdf` との API 互換のため残しているが
            現状は未使用。ページ画像はバイト単位でそのままラウンドトリップする。
        embed_font: True (既定) なら TTF を EPUB に同梱し、`@font-face` で
            参照する。テキスト選択の精度を保つために推奨。
    """
    if image_quality not in IMAGE_QUALITY_PRESETS:
        image_quality = "high"

    if len(docs) != len(book):
        raise ValueError(
            f"docs length ({len(docs)}) does not match number of eligible "
            f"pages in EPUB ({len(book)})"
        )
    docs_by_eligible_idx: Dict[int, DocumentAnalyzerSchema] = {
        i: d for i, d in enumerate(docs)
    }

    font_path = font_path or DEFAULT_FONT_PATH
    pdfmetrics.registerFont(TTFont("MPLUS1p-Medium", font_path))

    opf_dir = book.opf_dir
    css_zip_path = (
        f"{opf_dir}/styles/{_CSS_FILE_NAME}" if opf_dir else f"styles/{_CSS_FILE_NAME}"
    )
    font_zip_path = None
    if embed_font:
        font_zip_path = (
            f"{opf_dir}/fonts/{_FONT_FILE_NAME}"
            if opf_dir
            else f"fonts/{_FONT_FILE_NAME}"
        )

    eligible_by_xhtml = {
        ref.xhtml_path: ref for ref in book.page_refs if ref.skip_reason is None
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with (
        zipfile.ZipFile(book.src_path, "r") as src_zip,
        zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as out_zip,
    ):
        # mimetype は zip 先頭・STORED・改行なし固定バイトで書く必要あり
        mimetype_info = zipfile.ZipInfo("mimetype")
        mimetype_info.compress_type = zipfile.ZIP_STORED
        out_zip.writestr(mimetype_info, b"application/epub+zip")

        if embed_font and font_zip_path:
            font_face = _FONT_FACE_CSS.replace(
                "__FONT_HREF__",
                _rel_path(css_zip_path, font_zip_path),
            )
            css_text = font_face + _TEXT_LAYER_CSS
        else:
            css_text = _TEXT_LAYER_CSS

        written: set = {"mimetype"}

        for info in src_zip.infolist():
            name = info.filename
            if name == "mimetype" or name in written:
                continue

            data = src_zip.read(name)

            if name == book.opf_path:
                data = _patch_opf(data, opf_dir, css_zip_path, font_zip_path)
            elif name in eligible_by_xhtml:
                ref = eligible_by_xhtml[name]
                doc = docs_by_eligible_idx.get(ref.eligible_index)
                if doc is not None:
                    data = _rewrite_xhtml(data, ref, doc, css_zip_path)

            out_info = zipfile.ZipInfo(name)
            out_info.date_time = info.date_time
            out_info.compress_type = (
                zipfile.ZIP_DEFLATED
                if info.compress_type != zipfile.ZIP_STORED
                else zipfile.ZIP_STORED
            )
            out_info.external_attr = info.external_attr
            out_zip.writestr(out_info, data)
            written.add(name)

        if css_zip_path not in written:
            out_zip.writestr(css_zip_path, css_text.encode("utf-8"))
            written.add(css_zip_path)

        if embed_font and font_zip_path and font_zip_path not in written:
            with open(font_path, "rb") as fp:
                font_bytes = fp.read()
            font_info = zipfile.ZipInfo(font_zip_path)
            font_info.compress_type = zipfile.ZIP_DEFLATED
            out_zip.writestr(font_info, font_bytes)
            written.add(font_zip_path)
