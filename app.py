#workflow:

#accept test fp, true fp
#test file (pdf of scanned image) is uploaded in the test folder
#true file (native pdf) is uploaded in the true folder

#test file -> converted to image -> deskewed -> deskewed file is converted to pdf -> put through DOCUMENTAI API -> line item table is extracted -> final price of line items is extracted as a list -> set is returned

#true file -> table is extracted, cleaned -> price of line items is extracted -> prices are compared and flagged

import re

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1

import os
import google.cloud

import img2pdf
import pdf2image

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pdf2image import convert_from_path
from PIL import Image
from deskew import determine_skew
from collections import Counter

from dotenv import load_dotenv
from img2table.document import PDF
from img2table.ocr import TesseractOCR

load_dotenv()
assert os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), "GOOGLE_APPLICATION_CREDENTIALS not set"

def check_fp(fp):
    if os.path.exists(fp):
        if str(fp).endswith(".pdf"):
            return True
    else:
        return False
    
def accept_ip_op_fp():
    test_fp = input("enter test image fp: ")
    while check_fp(test_fp) != True:
        test_fp = input("sorry, that path was incorrect. try again: ")
    print("that worked! test image found")

    true_fp = input("enter true image fp: ")
    while check_fp(true_fp) != True:
        true_fp = input("sorry, that path was incorrect. try again: ")
    print("that worked! true image found")

    return(test_fp, true_fp)

def pdf_to_img(pdf_fp):

    img = convert_from_path(pdf_fp)[0]
    img_fp = str(pdf_fp).split(".")[0]+"_png.png"
    img.save(f"{img_fp}", "PNG")

    print(img_fp)
    return(img_fp)

def img_to_pdf(img_fp):
    img = Image.open(img_fp)
    
    pdf_bytes = img2pdf.convert(img.filename)
    pdf_fp = str(img_fp).split(".")[0].replace("_png", "")+".pdf"

    pdf = open(pdf_fp, "wb")
    pdf.write(pdf_bytes)

    img.close()
    pdf.close()

    print(pdf_fp)
    return(pdf_fp)

class Deskewer:
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._load_image()
        self.angle = None
    
    def _load_image(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("The image could not be loaded. Please provide a correct file path.")
        return image
    
    def calculate_skew_angle(self):
        self.angle = determine_skew(self.image)
        return self.angle
    
    def rotate_image(self):
        if self.angle is None:
            raise ValueError("The image cannot be rotated without calculating the tilt angle.")
        
        (h, w) = self.image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def save_corrected_image(self, output_path):
        corrected_image = self.rotate_image()
        cv2.imwrite(output_path, corrected_image)
        print(f"Corrected image saved: {output_path}")
    
    def display_images(self):
        corrected_image = self.rotate_image()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(corrected_image, cmap='gray')
        axes[1].set_title('Corrected Image')
        plt.show()

def process_with_docai(file_path, project_id, processor_id, location="us"):
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai_v1.DocumentProcessorServiceClient(client_options=opts)

    full_processor_name = client.processor_path(project_id, location, processor_id)
    request = documentai_v1.GetProcessorRequest(name=full_processor_name)
    processor = client.get_processor(request=request)

    #print(f"Processor Name: {processor.name}")

    with open(file_path, "rb") as image:
        image_content = image.read()

    raw_document = documentai_v1.RawDocument(
        content=image_content,
        mime_type="application/pdf",
    )

    request = documentai_v1.ProcessRequest(name=processor.name, raw_document=raw_document)
    result = client.process_document(request=request)
    document = result.document

    #print("The document contains the following text:")
    #print(document.text)

    return document

class testPOParser:
    
    NUM = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
    NUM_OR_PCT = re.compile(rf"^{NUM}(?:\s*%?)$")

    def __init__(self, document):
        self.document = document
        self.items_table_ids = []

        # outputs
        self.df = None
        self.final_list = None

    def to_float(self, s):
        strs = str(s).replace(",", "").strip()
        return float(strs) if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", strs) else s

    def read_anchor(self, doc, anchor):
        if anchor is None or not getattr(anchor, "text_segments", None):
            return ""
        out = []
        for seg in anchor.text_segments:
            start = seg.start_index or 0
            end = seg.end_index or 0
            out.append(doc.text[start:end])
        return " ".join("".join(out).split())

    def dedupe_header(self, headers):
        seen = Counter()
        out = []
        for h in headers:
            seen[h] += 1
            out.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
        return out

    def bbox_norm_area(self, bpoly):
        xs = [v.x for v in bpoly.normalized_vertices]
        ys = [v.y for v in bpoly.normalized_vertices]
        return (max(xs) - min(xs)) * (max(ys) - min(ys)), min(ys)

    def list_tables_with_headers(self):
        self.items_table_ids = []
        for i, page in enumerate(self.document.pages):
            for j, table in enumerate(page.tables):
                headers_text = []
                for hr in table.header_rows:
                    for cell in hr.cells:
                        headers_text.append(self.read_anchor(self.document, cell.layout.text_anchor))
                header_str = " ".join(headers_text)
                print(f"[page {i} table {j}] header:", header_str)
                self.items_table_ids.append((i, j))
        print("Candidate line-item tables:", self.items_table_ids)

    def select_items_table_via_geometry(self):
        fallback = []
        for i, page in enumerate(self.document.pages):
            for j, table in enumerate(page.tables):
                if not table.body_rows:
                    continue
                area, top_y = self.bbox_norm_area(table.layout.bounding_poly)
                if top_y > 0.25:
                    fallback.append((area, i, j))
        fallback.sort(reverse=True)
        if not fallback:
            print("Candidate (geometry) line-item tables: []")
            self.items_table_ids = []
            return
        h, i, j = fallback[0]
        self.items_table_ids = [(i, j)]
        print("Candidate (geometry) line-item tables:", self.items_table_ids)

    def preview_first_rows(self, n=5):
        for (i, j) in self.items_table_ids[:1]:
            table = self.document.pages[i].tables[j]
            print("Header cell count:", sum(len(r.cells) for r in table.header_rows))
            for idx, br in enumerate(table.body_rows[:n]):
                print(f"body row {idx} cell count:", len(br.cells))
                print([self.read_anchor(self.document, c.layout.text_anchor) for c in br.cells])

    def build_naive_rows(self):
        rows = []
        for (p_i, t_i) in self.items_table_ids:
            table = self.document.pages[p_i].tables[t_i]

            headers = []
            for hr in table.header_rows:
                for cell in hr.cells:
                    headers.append(self.read_anchor(self.document, cell.layout.text_anchor))

            for br in table.body_rows:
                cells = [self.read_anchor(self.document, c.layout.text_anchor) for c in br.cells]
                try:
                    row = dict(zip(headers, cells))
                    rows.append(row)
                except Exception:
                    print("mismatch in rows/cols")
                    if len(cells) != len(headers):
                        continue
                    row = dict(zip(headers, cells))
                    rows.append(row)
        return rows

    def fix_header(self, raw_header_cells):
        clean_headers = []
        col_extras = []

        for cell in raw_header_cells:
            tokens = str(cell).split()
            extras = []
            while tokens:
                final = tokens[-1]
                if final == "%" and len(tokens) >= 2 and self.NUM_OR_PCT.match(tokens[-2]):
                    extras.append(tokens[-2] + " %")
                    tokens = tokens[:-2]
                    continue
                if self.NUM_OR_PCT.match(final):
                    extras.append(final)
                    tokens.pop()
                    continue
                break

            base_label = " ".join(tokens).strip() or "Col"
            label = base_label
            clean_headers.append(label)
            col_extras.append(list(reversed(extras)))

            clean_headers = self.dedupe_header(clean_headers)

        max_len = max((len(x) for x in col_extras), default=0)
        extra_rows = []
        for i in range(max_len):
            row = {}
            for label, extras in zip(clean_headers, col_extras):
                row[label] = extras[i] if i < len(extras) else None
            extra_rows.append(row)

        return clean_headers, extra_rows

    def build_dataframe_from_fix_header(self):
        if not self.items_table_ids:
            return None

        p_i, t_i = self.items_table_ids[0]
        table = self.document.pages[p_i].tables[t_i]

        raw_headers = [self.read_anchor(self.document, cell.layout.text_anchor)
                       for cell in table.header_rows[0].cells]

        headers, extra_rows = self.fix_header(raw_headers)

        body = []
        for br in table.body_rows:
            cells = [self.read_anchor(self.document, c.layout.text_anchor) for c in br.cells]
            if len(cells) != len(headers):
                continue
            body.append(dict(zip(headers, cells)))

        self.df = pd.DataFrame((extra_rows or []) + body)

        return self.df

    def pick_numeric_col(self, df, threshold=0.6):
        last_series = None
        for col in reversed(df.columns):
            s = df[col].dropna()
            last_series = s
            ok = 0
            for v in s:
                t = str(v).strip()
                if t.startswith('(') or t.endswith(')'):
                    t = t[1:-1].strip()
                if len(t.split()) > 1:
                    t = (t.split()[0])
                if self.NUM_OR_PCT.search(t):
                    ok += 1
            if len(s) > 0 and ok / len(s) >= threshold:
                return s
        return last_series

    def extract_final_list_from_df(self):
        if self.df is None or self.df.empty:
            self.final_list = []
            return self.final_list

        series = self.pick_numeric_col(self.df, threshold=0.7)
        if series is None or series.empty:
            self.final_list = []
            return self.final_list

        out = []
        for v in series.tolist():
            if v is None:
                out.append(None)
                continue
            s = str(v).strip()
            if not s:
                out.append(None)
                continue
            tokens = s.split()
            if not tokens:
                out.append(None)
                continue
            print(f"tokens={tokens}")
            token = tokens[-1]
            f = self.to_float(token)
            out.append(f)

        out = list(set(out))
        self.final_list = out
        return self.final_list


    def run_all(self):
        self.list_tables_with_headers()
        self.select_items_table_via_geometry()
        self.preview_first_rows(n=5)
        rows = self.build_naive_rows()
        self.build_dataframe_from_fix_header()
        self.extract_final_list_from_df()
        return self

class truePOParser:
    NUM = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
    NUM_OR_PCT = re.compile(rf"^{NUM}(?:\s*%?)$")

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.final_list: list[float] | None = None

    def join_spaced_letters(self, s: str) -> str:
        parts = s.split()
        if len(parts) >= 1:
            single = sum(1 for p in parts if len(p) == 1)
            if single/len(parts) >= 0.7:
                return "".join(parts)
        return s

    def clean_cell(self, x):
        if pd.isna(x):
            return x
        s = str(x)
        s = s.replace("\\n", " ").replace("\n", " ")
        s = " ".join(s.split())

        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            s = s[1:-1].strip()

        s = self.join_spaced_letters(s)

        return s

    def to_float_if_numeric(self, s):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return s
        
        t = str(s).strip()

        neg = False

        if (t.startswith("(") and t.endswith(")")) or (t.startswith(" (") and t.endswith(") ")):
            neg = True
            t = t[1:-1].strip()

        t = t.replace(" ", "")

        if self.NUM_OR_PCT.match(t):
            t = t.replace("%", "")
            t = t.replace(",", "")
            try:
                val = np.float64(t)
                return -val if neg else val
            except ValueError:
                return s
        return s

    def html_table_to_clean_df(self, table):
        html = table.html_repr()
        dfs = pd.read_html(html)
        if not dfs:
            return pd.DataFrame()

        df = dfs[0]

        df = df.map(self.clean_cell)

        for col in df.columns:
            df[col] = df[col].apply(self.to_float_if_numeric)
            if df[col].map(lambda v: (isinstance(v, (int, float, np.floating)) or pd.isna(v))).all():
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def last_numeric_col_values(self, df: pd.DataFrame):
        if df is None or df.empty:
            return []

        col = df.columns[-1]
        out: list[float] = []

        for v in df[col]:
            if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                out.append(float(v))
                continue

            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue

            t = str(v).strip()

            neg = False

            if t.startswith("(") and t.endswith(")"):
                neg = True
                t = t[1:-1].strip()

            t = t.split()[-1]

            t = t.replace(",", "")
            if t.endswith("%"):
                t = t[:-1].strip()

            if self.NUM_OR_PCT.match(t):
                try:
                    val = float(t)
                    out.append(-val if neg else val)
                except ValueError:
                    pass

        return out

    def set_from_table(self, table):
        self.df = self.html_table_to_clean_df(table)
        self.final_list = self.last_numeric_col_values(self.df)
        return self

    def set_from_extracted_tables(self, extracted_tables):
        for i, tables in extracted_tables.items():
            if tables:
                return self.set_from_table(tables[0])
        self.df = pd.DataFrame()
        self.final_list = []
        return self

def compare_items(test_list, true_list):
    line_items_to_check = []
    for i in range(len(true_list)):
        if true_list[i] not in test_list:
            line_items_to_check.append(i)
    return line_items_to_check

if __name__ == "__main__":
    
    test_pdf_fp, true_pdf_fp = accept_ip_op_fp()
    test_img_fp = pdf_to_img(test_pdf_fp)

    test_img_deskewed_fp = str(test_img_fp.split(".")[0].replace("_png", "_deskewed_png"))+".png"
    
    deskewer = Deskewer(test_img_fp)
    angle = deskewer.calculate_skew_angle()
    print(f"Detected tilt angle: {angle:.2f} degrees")
    deskewer.save_corrected_image(test_img_deskewed_fp)
    deskewer.display_images()

    test_pdf_deskewed_fp = img_to_pdf(test_img_deskewed_fp)

    gcp_fp = test_pdf_deskewed_fp
    gcp_project_id = "seraphic-result-468108-a8"
    gcp_processor_id = "a2eb57397c83723c"
    gcp_location = "us"

    document = process_with_docai(file_path=gcp_fp,
                                  project_id=gcp_project_id,
                                  processor_id=gcp_processor_id,
                                  location=gcp_location)
    
    print("processing test image...")

    parser = testPOParser(document)
    parser.run_all()

    test_df = parser.df
    test_final_list = parser.final_list

    print("done processing test image!")

    if test_df is not None:
        print("Final Test DataFrame shape:", test_df.shape)
    print("Final List:", test_final_list)

    pdf = PDF(
        true_pdf_fp,
        pages=[0],
        detect_rotation=True,
        pdf_text_extraction=True
    )
    tessocr = TesseractOCR(n_threads=1, lang="eng")

    print("processing true pdf...")

    extracted_tables = pdf.extract_tables(
        ocr=tessocr,
        implicit_rows=False,
        borderless_tables=False,
        min_confidence=0
    )

    parser = truePOParser().set_from_extracted_tables(extracted_tables)

    true_df = parser.df
    true_final_list = parser.final_list

    print("done processing true image!")

    print("DF shape:", true_df.shape if true_df is not None else None)
    print("First 5 values:", true_final_list[:5] if true_final_list else true_final_list)
    
    line_items_to_check = compare_items(test_final_list, true_final_list)

    if len(line_items_to_check) > 0:
        for i in line_items_to_check:
            print(f"check line item {i+1} in true file")
    else:
        print(f"no items to check! feel free to run this again with different PO's!")