from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "docs" / "byte_based_byte_to_int_operators.pptx"

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def color(hex_color: str) -> str:
    return hex_color.strip().lstrip("#").upper()


def shape_text(
    shape_id: int,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    *,
    font_size: int = 22,
    bold: bool = False,
    fill: str | None = None,
    line: str | None = None,
    font_color: str = "1F2937",
    radius: bool = False,
) -> str:
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
        if fill
        else "<a:noFill/>"
    )
    line_xml = (
        f'<a:ln w="9000"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>'
        if line
        else '<a:ln><a:noFill/></a:ln>'
    )
    bold_xml = ' b="1"' if bold else ""
    preset = "roundRect" if radius else "rect"
    paras = "".join(
        f'<a:p><a:r><a:rPr lang="zh-CN" sz="{font_size * 100}"{bold_xml}>'
        f'<a:solidFill><a:srgbClr val="{color(font_color)}"/></a:solidFill>'
        f'<a:latin typeface="Aptos"/><a:ea typeface="PingFang SC"/></a:rPr>'
        f"<a:t>{esc(line_text)}</a:t></a:r></a:p>"
        for line_text in text.split("\n")
    )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="Text {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="{preset}"><a:avLst/></a:prstGeom>
        {fill_xml}{line_xml}
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="100000" tIns="70000" rIns="100000" bIns="70000"/>
        <a:lstStyle/>
        {paras}
      </p:txBody>
    </p:sp>
    """


def title(shape_id: int, text: str, subtitle: str | None = None) -> str:
    xml = shape_text(
        shape_id,
        520_000,
        300_000,
        12_250_000,
        600_000,
        text,
        font_size=30,
        bold=True,
        font_color="0F172A",
    )
    if subtitle:
        xml += shape_text(
            shape_id + 1,
            540_000,
            900_000,
            12_000_000,
            420_000,
            subtitle,
            font_size=15,
            font_color="475569",
        )
    return xml


def arrow(shape_id: int, x1: int, y1: int, x2: int, y2: int, text: str = "") -> str:
    label = ""
    if text:
        label = shape_text(
            shape_id + 1000,
            min(x1, x2) + abs(x2 - x1) // 2 - 450_000,
            min(y1, y2) - 220_000,
            900_000,
            300_000,
            text,
            font_size=12,
            fill="FFFFFF",
            line="CBD5E1",
            font_color="334155",
            radius=True,
        )
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr>
        <p:cNvPr id="{shape_id}" name="Arrow {shape_id}"/>
        <p:cNvCxnSpPr/><p:nvPr/>
      </p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{min(x1, x2)}" y="{min(y1, y2)}"/><a:ext cx="{abs(x2 - x1)}" cy="{abs(y2 - y1) or 1}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="26000">
          <a:solidFill><a:srgbClr val="2563EB"/></a:solidFill>
          <a:tailEnd type="triangle"/>
        </a:ln>
      </p:spPr>
    </p:cxnSp>
    {label}
    """


def table(shape_id: int, x: int, y: int, col_widths: list[int], row_h: int, rows: list[list[str]]) -> str:
    xml = []
    sid = shape_id
    for r, row in enumerate(rows):
        xx = x
        for c, cell in enumerate(row):
            is_head = r == 0
            xml.append(
                shape_text(
                    sid,
                    xx,
                    y + r * row_h,
                    col_widths[c],
                    row_h,
                    cell,
                    font_size=12 if not is_head else 13,
                    bold=is_head,
                    fill="E0F2FE" if is_head else ("F8FAFC" if r % 2 else "FFFFFF"),
                    line="CBD5E1",
                    font_color="0F172A",
                )
            )
            sid += 1
            xx += col_widths[c]
    return "".join(xml)


def slide_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {body}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def build_slides() -> list[str]:
    slides: list[str] = []

    body = title(
        2,
        "Byte-based 算子中哪些会把 byte 处理成 int？",
        "结论：外部自动机仍是 byte -> byte；内部会把 byte 解析为长度、偏移、计数、rank、频率、状态号等整数符号。",
    )
    body += shape_text(4, 780_000, 1_650_000, 11_800_000, 900_000, "可认为“byte 被处理为 int”的算子", font_size=24, bold=True, fill="DBEAFE", line="93C5FD", radius=True)
    body += table(
        10,
        780_000,
        2_850_000,
        [2_350_000, 3_450_000, 5_100_000],
        520_000,
        [
            ["算子", "内部整数含义", "典型 codec"],
            ["integer code", "varint 长度、offset、block size", "Snappy, LZ4-family"],
            ["match tokens", "literal length / match length / offset", "LZ4, Snappy, LZF"],
            ["lz match / window", "匹配位置、距离、长度", "gzip, LZ4, zstd, LZMA"],
            ["symbol recode / run length", "rank、run count", "Bzip2, Bzip3"],
            ["entropy coding", "频率、码长、状态号", "gzip, zstd, Brotli"],
        ],
    )
    slides.append(slide_xml(body))

    body = title(2, "1. match tokens：把 token byte 解析成 length / offset 整数")
    body += shape_text(4, 700_000, 1_350_000, 3_100_000, 800_000, "输入 byte 数据\n\"abcabcabc\"", font_size=20, bold=True, fill="EEF2FF", line="A5B4FC", radius=True)
    body += arrow(5, 3_900_000, 1_750_000, 5_000_000, 1_750_000, "lz match")
    body += shape_text(6, 5_050_000, 1_250_000, 3_400_000, 1_100_000, "内部整数\nliteral_len = 3\noffset = 3\nmatch_len = 6", font_size=19, bold=True, fill="ECFEFF", line="67E8F9", radius=True)
    body += arrow(7, 8_550_000, 1_750_000, 9_600_000, 1_750_000, "pack")
    body += shape_text(8, 9_650_000, 1_350_000, 2_750_000, 850_000, "输出仍是 byte stream\n[token][offset][len]", font_size=18, fill="F0FDF4", line="86EFAC", radius=True)
    body += shape_text(9, 760_000, 3_150_000, 11_800_000, 1_050_000, "例子：LZ4/Snappy 这类格式不是把数据类型永久变成 int，而是在压缩阶段从 byte 序列中发现重复片段，并把重复片段表示为整数 offset 和 length。自动机上仍然画 byte -> byte。", font_size=18, fill="FFFFFF", line="CBD5E1", radius=True)
    slides.append(slide_xml(body))

    body = title(2, "2. integer code：把若干 byte 解码为整数元数据")
    body += shape_text(4, 760_000, 1_320_000, 4_050_000, 1_150_000, "长度 300\nvarint bytes: AC 02", font_size=23, bold=True, fill="FEF3C7", line="FCD34D", radius=True)
    body += arrow(5, 4_950_000, 1_850_000, 6_200_000, 1_850_000, "varint")
    body += shape_text(6, 6_300_000, 1_320_000, 3_050_000, 1_150_000, "整数值\n300", font_size=30, bold=True, fill="DCFCE7", line="86EFAC", radius=True)
    body += arrow(7, 9_500_000, 1_850_000, 10_450_000, 1_850_000, "write")
    body += shape_text(8, 10_500_000, 1_320_000, 2_150_000, 1_150_000, "再编码为\nbyte", font_size=22, bold=True, fill="E0F2FE", line="7DD3FC", radius=True)
    body += table(
        20,
        900_000,
        3_230_000,
        [2_400_000, 4_100_000, 4_900_000],
        560_000,
        [
            ["具体数据", "内部整数", "作用"],
            ["literal 长度 300", "300", "决定后面复制多少 literal bytes"],
            ["match offset 1024", "1024", "从窗口中回看 1024 bytes"],
            ["block size 65536", "65536", "决定 block 边界和状态重置"],
        ],
    )
    slides.append(slide_xml(body))

    body = title(2, "3. symbol recode / run length：byte 符号变成 rank 和 count")
    body += shape_text(4, 700_000, 1_250_000, 3_050_000, 900_000, "BWT/MTF 后的 byte/rank\n[0, 0, 0, 0, 2]", font_size=20, bold=True, fill="FCE7F3", line="F9A8D4", radius=True)
    body += arrow(5, 3_850_000, 1_700_000, 5_050_000, 1_700_000, "RLE")
    body += shape_text(6, 5_150_000, 1_250_000, 3_500_000, 900_000, "内部整数\nsymbol = 0\ncount = 4\nnext_rank = 2", font_size=20, bold=True, fill="FAE8FF", line="D8B4FE", radius=True)
    body += arrow(7, 8_750_000, 1_700_000, 9_850_000, 1_700_000, "entropy")
    body += shape_text(8, 9_950_000, 1_250_000, 2_500_000, 900_000, "输出 byte\n压缩码流", font_size=20, fill="E0F2FE", line="7DD3FC", radius=True)
    body += shape_text(9, 760_000, 3_100_000, 11_800_000, 1_200_000, "Bzip2/Bzip3 的 block transform 后，byte 符号常被重编码成 rank；连续 rank 再变成整数 count。这里 byte 数据被处理成 int-like 符号，但这是 codec 内部 token，不是用户数据类型状态。", font_size=18, fill="FFFFFF", line="CBD5E1", radius=True)
    slides.append(slide_xml(body))

    body = title(2, "4. entropy coding：byte 频率变成 code length / state 整数")
    body += shape_text(4, 680_000, 1_180_000, 3_100_000, 900_000, "输入 bytes\nAAAAABBC", font_size=22, bold=True, fill="ECFCCB", line="BEF264", radius=True)
    body += arrow(5, 3_900_000, 1_620_000, 5_050_000, 1_620_000, "count")
    body += shape_text(6, 5_120_000, 1_050_000, 3_200_000, 1_250_000, "整数统计\nA:5\nB:2\nC:1", font_size=21, bold=True, fill="FEF9C3", line="FDE047", radius=True)
    body += arrow(7, 8_430_000, 1_620_000, 9_600_000, 1_620_000, "table")
    body += shape_text(8, 9_700_000, 1_050_000, 2_750_000, 1_250_000, "整数码长/状态\nA:1 bit\nB:2 bits\nC:3 bits", font_size=19, bold=True, fill="DBEAFE", line="93C5FD", radius=True)
    body += shape_text(9, 760_000, 3_000_000, 11_800_000, 1_280_000, "Huffman/FSE/ANS/range coding 都会把 byte 符号映射到整数频率、码长、累积分布或状态表。压缩输出仍是 byte stream，但内部状态不是“纯 byte 自循环”。", font_size=18, fill="FFFFFF", line="CBD5E1", radius=True)
    slides.append(slide_xml(body))

    body = title(2, "5. byte delta / dictionary filter：byte 到整数 residual 或 id")
    body += table(
        4,
        780_000,
        1_300_000,
        [2_600_000, 3_700_000, 5_300_000],
        650_000,
        [
            ["算子", "具体数据", "内部 int-like 表示"],
            ["byte delta", "[10, 12, 14, 9]", "residual: [10, +2, +2, -5]"],
            ["dictionary/filter", "bytes for \"temperature\"", "dict_id = 17, length = 11"],
            ["container/frame", "header bytes", "magic, block_size, checksum 等整数字段"],
            ["block/state", "64KB chunk boundary", "block index, reset flag, level id"],
        ],
    )
    body += shape_text(20, 840_000, 4_350_000, 11_650_000, 950_000, "这些算子更像“解释 byte”：它们从 byte 中推导出整数 residual、字典 ID 或结构化 header 字段，然后再把结果编码回 byte。", font_size=19, fill="FFFFFF", line="CBD5E1", radius=True)
    slides.append(slide_xml(body))

    body = title(2, "对 web_compression 自动机的建议")
    body += shape_text(4, 820_000, 1_200_000, 11_700_000, 850_000, "主自动机仍保持：int / float / byte 输入 -> byte", font_size=24, bold=True, fill="DBEAFE", line="93C5FD", radius=True)
    body += shape_text(5, 820_000, 2_350_000, 11_700_000, 1_050_000, "不要把 byte-based 主状态直接画成 byte -> int -> byte，因为这会暗示外部 pipeline 的数据表示变成了 int。更准确的画法是在 tooltip 或辅助图中标注“internal integer tokens”。", font_size=19, fill="FFFFFF", line="CBD5E1", radius=True)
    body += table(
        10,
        960_000,
        4_000_000,
        [3_350_000, 3_700_000, 4_150_000],
        520_000,
        [
            ["主图状态", "内部注释", "适用算子"],
            ["byte -> byte", "byte -> int tokens -> byte", "match tokens, integer code"],
            ["byte -> byte", "byte -> ranks/counts -> byte", "symbol recode, run length"],
            ["byte -> byte", "byte -> frequencies/states -> byte", "Huffman, FSE/ANS, range"],
        ],
    )
    slides.append(slide_xml(body))

    body = title(2, "一句话结论")
    body += shape_text(
        4,
        900_000,
        1_450_000,
        11_500_000,
        1_250_000,
        "可以将 byte 处理为 int 的 byte-based 算子，主要是：\ninteger_code、match_token_format、lz_match/window、symbol_recode、run_length、entropy_*、byte_delta_filter、dictionary_filter、container_frame。",
        font_size=24,
        bold=True,
        fill="ECFEFF",
        line="67E8F9",
        radius=True,
    )
    body += shape_text(
        5,
        900_000,
        3_250_000,
        11_500_000,
        1_250_000,
        "但它们处理出来的是 codec 内部整数 token：长度、偏移、rank、count、频率、码长、状态号、字典 ID、block size。自动机外部状态仍建议保持 byte。",
        font_size=22,
        fill="FFFFFF",
        line="CBD5E1",
        radius=True,
    )
    slides.append(slide_xml(body))

    return slides


def write_pptx(slides: list[str]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
"""
            + "".join(
                f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>\n'
                for i in range(1, len(slides) + 1)
            )
            + "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>""",
        )
        slide_ids = "".join(
            f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1)
        )
        z.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""",
        )
        rels = "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
            for i in range(1, len(slides) + 1)
        )
        z.writestr(
            "ppt/_rels/presentation.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>""",
        )
        for i, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide)
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>""",
            )


if __name__ == "__main__":
    write_pptx(build_slides())
    print(OUT)
