from html_to_markdown import convert, convert_with_inline_images, InlineImageConfig




from bs4 import BeautifulSoup, Tag
import copy

def expand_table(table: Tag) -> Tag:
    """
    Trả về một bản sao của `table` với tất cả rowspan/colspan đã được "fill" bằng giá trị
    tương đương và không còn rowspan/colspan nữa.
    """
    # Lưu thông tin hàng <tr> ban đầu để xác định vị trí bắt đầu các ô
    rows = table.find_all("tr")
    # occupancy[(r, c)] = {"text_html": str(inner_html), "is_th": bool, "attrs": dict}
    occupancy = {}
    max_row = 0
    max_col = 0

    # r_index là chỉ số hàng logic (bắt đầu ô) — dùng theo thứ tự các <tr> xuất hiện
    for r_index, tr in enumerate(rows):
        c = 0
        # tìm c bắt đầu (bỏ qua các ô đã được chiếm bởi rowspan từ hàng trước)
        while (r_index, c) in occupancy:
            c += 1

        # lặp qua từng ô hiện tại
        for cell in tr.find_all(["td", "th"], recursive=False):
            # lấy text/html bên trong ô (bảo toàn HTML con)
            inner_html = ''.join(str(x) for x in cell.contents).strip()
            is_th = (cell.name.lower() == "th")
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # nếu ô rỗng text thì dùng "" (vẫn copy)
            for dr in range(rowspan):
                for dc in range(colspan):
                    rr = r_index + dr
                    cc = c + dc
                    occupancy[(rr, cc)] = {
                        "inner_html": inner_html,
                        "is_th": is_th,
                        # lưu attributes của ô gốc (chỉ cho ô gốc vị trí dr==0 and dc==0)
                        "attrs": dict(cell.attrs) if (dr == 0 and dc == 0) else {}
                    }
                    if rr + 1 > max_row:
                        max_row = rr + 1
                    if cc + 1 > max_col:
                        max_col = cc + 1
            # tiến c tới sau colspan và skip các vị trí đã được chiếm
            c += colspan
            while (r_index, c) in occupancy:
                c += 1

    # Tạo table mới (sao chép các attribute của table gốc, nhưng loại bỏ các ô con)
    new_table = Tag(name="table")
    for k, v in table.attrs.items():
        new_table.attrs[k] = v

    # xây các hàng mới từ occupancy
    for r in range(max_row):
        new_tr = Tag(name="tr")
        for c in range(max_col):
            info = occupancy.get((r, c))
            if info is None:
                # nếu không tồn tại cell (vị trí trống), tạo ô rỗng td
                new_cell = Tag(name="td")
                new_cell.string = ""
            else:
                tagname = "th" if info["is_th"] else "td"
                new_cell = Tag(name=tagname)
                # copy attributes only from the original "top-left" cell stored in attrs
                for a_k, a_v in info.get("attrs", {}).items():
                    # loại bỏ rowspan/colspan nếu còn
                    if a_k.lower() in ("rowspan", "colspan"):
                        continue
                    new_cell.attrs[a_k] = a_v
                # set inner HTML (cẩn thận với parsing)
                if info["inner_html"] == "":
                    new_cell.string = ""
                else:
                    # parse inner html fragment và chèn
                    frag = BeautifulSoup(info["inner_html"], "html.parser")
                    for node in frag.contents:
                        new_cell.append(copy.copy(node))
            new_tr.append(new_cell)
        new_table.append(new_tr)

    return new_table

def expand_all_tables(html: str) -> str:
    """
    Nhận HTML (chuỗi), xử lý tất cả <table> bên trong để expand rowspan/colspan,
    và trả về HTML mới (dưới dạng string).
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    for tbl in tables:
        new_tbl = expand_table(tbl)
        tbl.replace_with(new_tbl)
    # trả về toàn bộ document (không prettify để giữ format)
    return str(soup)



if __name__ == "__main__":

    input_file = "/data/quanglhm/OD.v5i.yolov11/airace/merged_output.md"
    output_file_html = "/data/AIRACE/RAG/output/answer_passer.html"
    output_file_md = "/data/AIRACE/RAG/output/answer_passer.md"


    with open(input_file, "r", encoding="utf-8") as f:
        html_string = f.read()

    html_file= expand_all_tables(html_string)

    with open("/data/AIRACE/RAG/output/json_file.html", "w") as f:
        f.write(html_file)
    markdown = convert(html_file)

    with open(output_file_md, "w", encoding="utf-8") as f:
        f.write(markdown)



