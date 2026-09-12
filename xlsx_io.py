"""仅依赖标准库的简易 XLSX 读写器。

数据附件只有普通单元格，使用 OOXML 直接读写可以避免对 Excel/Gurobi 的强依赖。
"""

from __future__ import annotations

import math
import os
import re
import zipfile
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return value


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(name))
    result = []
    for item in root.findall("main:si", NS):
        result.append("".join(node.text or "" for node in item.iter(f"{{{NS['main']}}}t")))
    return result


def _sheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rels = {node.attrib["Id"]: node.attrib["Target"] for node in rel_root}
    result = []
    for sheet in workbook.findall("main:sheets/main:sheet", NS):
        rid = sheet.attrib[f"{{{NS['rel']}}}id"]
        target = rels[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        result.append((sheet.attrib["name"], target))
    return result


def read_sheet(path: str, sheet_index: int = 0) -> list[list[object]]:
    """返回一个工作表的矩阵，空单元格为 None。"""
    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        sheets = _sheet_paths(archive)
        if not 0 <= sheet_index < len(sheets):
            raise IndexError(f"工作表索引越界: {sheet_index}, 共 {len(sheets)} 个工作表")
        root = ET.fromstring(archive.read(sheets[sheet_index][1]))
        rows = []
        max_col = 0
        for row_node in root.findall("main:sheetData/main:row", NS):
            values = {}
            for cell in row_node.findall("main:c", NS):
                ref = cell.attrib.get("r", "A1")
                col = _column_number(ref)
                max_col = max(max_col, col)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("main:v", NS)
                inline_node = cell.find("main:is", NS)
                if cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(node.text or "" for node in inline_node.iter(f"{{{NS['main']}}}t"))
                elif value_node is None:
                    value = None
                else:
                    raw = value_node.text or ""
                    if cell_type == "s":
                        value = strings[int(raw)]
                    elif cell_type == "b":
                        value = raw == "1"
                    else:
                        try:
                            value = float(raw)
                        except ValueError:
                            value = raw
                values[col] = value
            rows.append(values)
        matrix = []
        for values in rows:
            matrix.append([values.get(col) for col in range(1, max_col + 1)])
        return matrix


def excel_serial_to_date(value: object) -> datetime | None:
    """将 Excel 日期序列转换为日期；非日期输入返回 None。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return datetime(1899, 12, 30) + timedelta(days=float(value))
    return None


def _xml_escape(value: object) -> str:
    text = str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _cell_xml(ref: str, value: object, shared: dict[str, int]) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        if value is None:
            return ""
        return f'<c r="{ref}"><v>{value:.12g}</v></c>'
    index = shared.setdefault(str(value), len(shared))
    return f'<c r="{ref}" t="s"><v>{index}</v></c>'


def _column_name(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_xlsx(path: str, sheets: dict[str, list[list[object]]]) -> None:
    """写出普通值工作簿，足够覆盖题目要求的结果表。"""
    shared: dict[str, int] = {}
    sheet_xml = []
    for sheet_index, (name, rows) in enumerate(sheets.items(), 1):
        row_xml = []
        for row_index, row in enumerate(rows, 1):
            cells = []
            for col_index, value in enumerate(row, 1):
                cell = _cell_xml(f"{_column_name(col_index)}{row_index}", value, shared)
                if cell:
                    cells.append(cell)
            row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
        sheet_xml.append((name, "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>"
                          + "".join(row_xml) + "</sheetData></worksheet>"))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        content_types = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>',
        ]
        for index in range(1, len(sheet_xml) + 1):
            content_types.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        content_types.append("</Types>")
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        rels = [f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheet_xml) + 1)]
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{NS["pkgrel"]}">{"".join(rels)}</Relationships>')
        sheet_nodes = "".join(f'<sheet name="{_xml_escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheet_xml, 1))
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{NS["main"]}" xmlns:r="{NS["rel"]}"><sheets>{sheet_nodes}</sheets></workbook>')
        sst = "".join(f"<si><t>{_xml_escape(value)}</t></si>" for value, _ in sorted(shared.items(), key=lambda item: item[1]))
        archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{NS["main"]}" count="{len(shared)}" uniqueCount="{len(shared)}">{sst}</sst>')
        for index, (_, xml) in enumerate(sheet_xml, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", xml)
