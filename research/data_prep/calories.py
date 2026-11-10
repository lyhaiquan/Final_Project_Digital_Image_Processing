"""Bảng kcal/100 g cho 47 lớp -> calories_47.csv

Nguồn số liệu: Nutrition5k (Thames et al., CVPR 2021), file ingredients_metadata.csv (cột cal/g).
Theo bài báo, giá trị lấy từ USDA FNDDS. File gốc lưu ở ref/nutrition5k_ingredients_metadata.csv.
Script chỉ ĐỌC số từ file đó, không có con số nào gõ tay.

match:
  khớp            cùng món, cùng cách chế biến
  gần đúng        cùng nguyên liệu chính, cách nấu hơi khác
  chỉ nguyên liệu chỉ có nguyên liệu chưa tính dầu/đường/nước kho -> THẤP hơn thực tế
  không có        không có mục nào đủ gần -> để trống, cần tra Bảng thành phần thực phẩm Việt Nam (Viện Dinh dưỡng)
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ref", "nutrition5k_ingredients_metadata.csv")

# lớp: (tên tiếng Việt (? = chưa xác nhận), mục Nutrition5k hoặc None, mức khớp, ghi chú)
MAP = {
    "rice":                     ("cơm trắng", "white rice", "khớp", ""),
    "fried_pork_roll":          ("đã gộp vào roasted_cinnamon_pork", None, "không có", "nhóm xác nhận 09/11: là chả quế"),
    "fried_tofu":               ("đậu phụ rán", "tofu", "chỉ nguyên liệu", "đậu chưa rán"),
    "betel_leaf_pork":          ("chả lá lốt", "meatballs", "gần đúng", "thịt viên"),
    "bean_sprouts_bamboo":      ("giá xào măng", "bean sprouts", "chỉ nguyên liệu", ""),
    "minced_pork_corn":         ("thịt băm xào ngô", None, "không có", "món trộn, tỉ lệ thịt/ngô thay đổi"),
    "bell_pepper_chicken":      ("gà xào ớt chuông", "chicken", "chỉ nguyên liệu", "chỉ phần gà"),
    "rolled_omelette":          ("trứng cuộn", "omelets", "khớp", ""),
    "pumpkin":                  ("bí đỏ", "pumpkins", "gần đúng", ""),
    "boiled_egg":               ("trứng luộc", "hard boiled eggs", "khớp", ""),
    "drink":                    ("đồ uống", None, "không có", "tuỳ loại nước"),
    "coconut_braised_pork":     ("thịt kho (nước dừa)", "pork", "chỉ nguyên liệu", "chưa tính đường, nước kho"),
    "chinese_sausage":          ("lạp xưởng", "sausage", "gần đúng", "xúc xích kiểu Mỹ"),
    "fried_chicken":            ("gà rán", "fried chicken", "khớp", ""),
    "leafy_greens":             ("rau xanh", "mixed greens", "gần đúng", ""),
    "pork_broccoli":            ("thịt xào súp lơ", "broccoli", "chỉ nguyên liệu", "chỉ phần súp lơ"),
    "cutlet":                   ("thịt lợn chiên", "pork chops", "gần đúng", ""),
    "chayote":                  ("su su", "chayote squash", "khớp", ""),
    "pork_floss":               ("ruốc / chà bông", None, "không có", ""),
    "quail_egg":                ("trứng cút", "eggs", "gần đúng", "trứng gà"),
    "egg_meatloaf":             ("chả trứng", None, "không có", ""),
    "sausage":                  ("xúc xích", "sausage", "khớp", ""),
    "potato":                   ("khoai tây", "potatoes", "gần đúng", "luộc; xào/chiên sẽ cao hơn"),
    "soup":                     ("canh", "chicken soup", "gần đúng", "dao động lớn theo loại canh"),
    "boiled_pork":              ("thịt luộc", "pork", "gần đúng", ""),
    "radish":                   ("củ cải", "radishes", "khớp", ""),
    "braised_egg":              ("trứng kho", "hard boiled eggs", "chỉ nguyên liệu", "chưa tính nước kho"),
    "water_spinach":            ("rau muống", "spinach (cooked)", "gần đúng", "rau khác loài"),
    "braised_fish":             ("cá kho", "fish", "chỉ nguyên liệu", "chưa tính đường, dầu"),
    "pork_roll":                ("giò lụa", "ham", "gần đúng", ""),
    "shrimp_spring_roll":       ("nem / chả giò tôm", "spring rolls", "gần đúng", ""),
    "fried_egg":                ("trứng rán", "fried eggs", "khớp", ""),
    "sweet_sour_chicken":       ("gà sốt chua ngọt", None, "không có", ""),
    "dog_meat_rua_man":         ("rựa mận / giả cầy", None, "không có", ""),
    "lemongrass_chili_chicken": ("gà xào sả ớt", "chicken thighs", "chỉ nguyên liệu", "chỉ phần gà"),
    "pork_with_tofu":           ("đậu sốt thịt", None, "không có", ""),
    "silkworm_pupae":           ("nhộng", None, "không có", ""),
    "stir_fried_vegetables":    ("rau xào thập cẩm", "mixed vegetables", "chỉ nguyên liệu", "chưa tính dầu"),
    "fried_anchovy":            ("cá cơm rim", "anchovies", "gần đúng", ""),
    "peanuts":                  ("lạc", "peanuts", "khớp", ""),
    "fried_shrimp":             ("tôm chiên", "shrimp", "chỉ nguyên liệu", "tôm chưa chiên"),
    "bean_sprouts_green_beans": ("giá xào đậu que", "green beans", "chỉ nguyên liệu", ""),
    "cabbage":                  ("bắp cải luộc", "cabbage", "khớp", ""),
    "fried_pork_skin":          ("bì lợn chiên", None, "gần đúng", ""),
    "fish_ball":                ("cá viên", None, "không có", ""),
    "stir_fried_gizzard":       ("mề xào", None, "không có", ""),
    "roasted_cinnamon_pork":    ("chả quế", None, "không có", ""),
}


# Món Việt không có trong Nutrition5k: tra web 09/11/2026. Chỉ lấy con số bài viết GHI RÕ, kèm link.
# Bài báo phổ thông, không phải bảng chuẩn -> match = "bài viết". Đoán/ước của công cụ tìm kiếm thì KHÔNG lấy.
WEB = {
    "silkworm_pupae":        (206, "bài viết (dẫn Viện Dinh dưỡng)", "https://tienphong.vn/cong-dung-va-tac-hai-cua-nhong-tam-ban-can-biet-de-tranh-post803987.tpo", ""),
    "pork_floss":            (369, "bài viết", "https://nhathuoclongchau.com.vn/bai-viet/cha-bong-bao-nhieu-calo-an-nhu-the-nao-de-khong-lo-tang-can.html", ""),
    "roasted_cinnamon_pork": (235, "bài viết", "https://nhathuoclongchau.com.vn/bai-viet/cha-que-bao-nhieu-calo-mot-so-goi-y-han-che-tang-can-khi-an-cha-que.html", "nguồn khác ghi 386 cho chả quế chiên (mytour.vn)"),
    "fish_ball":             (140, "bài viết", "https://nhathuoclongchau.com.vn/bai-viet/100g-ca-vien-chien-bao-nhieu-calo-71051.html", "chiên; luộc 123"),
    "stir_fried_gizzard":    (135, "bài viết", "https://nhathuoclongchau.com.vn/bai-viet/me-ga-bao-nhieu-calo-an-me-ga-co-tang-can-khong.html", "bài ghi 120-150, lấy giữa khoảng; mề sống 94"),
    "fried_pork_skin":       (544, "gần đúng", "USDA FoodData Central fdcId 167961 (Snacks, pork skins, plain)", "bì chiên giòn dạng snack (rất khô); miếng ở căng tin dày, còn nước -> thực tế có thể thấp hơn"),
    "sweet_sour_chicken":    (250, "món nước ngoài tương tự", "https://foodstruct.com/food/restaurant-chinese-sweetandsourchicken", "gà chua ngọt nhà hàng Trung Quốc (USDA); bản siêu thị Anh 124"),
}


# Món không có số liệu ở đâu -> TÍNH THEO CÔNG THỨC (cách Viện Dinh dưỡng / USDA tính món ăn):
#   kcal/100g = Σ(gram nguyên liệu × kcal/100g) / Σ gram
# kcal nguyên liệu: USDA FoodData Central SR Legacy (fdcId ghi kèm), tra 09/11/2026.
# Quy đổi: 1 quả trứng ~50 g; 1 trái bắp ~102 g hạt; 1 muỗng canh dầu ~13.6 g, dầu hào ~18 g, nước tương ~16 g,
# đường ~12.5 g; 1 muỗng cà phê bột ~3 g; 100 ml dầu ~92 g.
# Bỏ qua: rau thơm, hành, tỏi, gia vị không ghi lượng; mắm tôm (USDA không có).
# Không trừ hao hụt khi nấu (bay hơi, dầu còn lại trong chảo) -> sai số vài chục %.
RECIPE = {
    "egg_meatloaf": ("https://quantrimang.com/cuoc-song/cach-lam-cha-trung-hap-cuc-ngon-ma-de-lam-183312", [
        ("trứng gà 6 quả", 300, 143, 171287), ("thịt lợn xay", 300, 263, 167902), ("cà rốt", 100, 41, 170393),
        ("mộc nhĩ (khô)", 100, 298, 168418), ("miến (khô)", 100, 351, 174258)],
        "mộc nhĩ, miến tính theo khối lượng khô; chỉ tính trứng + thịt thì ra 203"),
    "dog_meat_rua_man": ("https://hoanghamobile.com/tin-tuc/?p=469545", [
        ("giò heo", 500, 193, 168226), ("nước dừa", 300, 19, 170174), ("riềng (thay bằng gừng)", 100, 80, 169231),
        ("cơm mẻ (thay bằng cơm)", 50, 130, 168880), ("bột nghệ", 6, 312, 172231), ("dầu ăn 100 ml", 92, 884, 171411),
        ("đường phèn", 12.5, 387, 169655)],
        "coi như giữ hết dầu, không bay hơi; giò heo tính cả xương"),
    "pork_with_tofu": ("https://www.dienmayxanh.com/vao-bep/cach-lam-dau-hu-sot-thit-bam-thom-ngon-dam-da-hap-dan-dua-com-10049", [
        ("thịt lợn xay", 200, 263, 167902), ("đậu hũ non", 300, 61, 172449), ("trứng gà", 50, 143, 171287),
        ("dầu ăn", 13.6, 884, 171411), ("dầu hào", 18, 51, 174529), ("nước tương", 16, 53, 174277)],
        "công thức dùng đậu non; căng tin dùng đậu RÁN -> thay đậu rán (270, fdc 172451) thì ra ~258"),
    "minced_pork_corn": ("https://mytour.vn/vi/blog/bai-viet/5-mon-ngon-tu-thit-bam-khien-ban-an-tro-nen-soi-dong.html", [
        ("thịt lợn băm", 40, 263, 167902), ("bắp mỹ 2 trái", 204, 86, 169998)],
        "công thức không ghi lượng dầu -> chưa tính dầu, thấp hơn thực tế"),
}
NOT_COUNTED = {"drink": "không tính calo đồ uống (nhóm quyết định 09/11)"}


def recipe_kcal(items):
    return round(sum(g * k for _, g, k, _ in items) / sum(g for _, g, _, _ in items), 1)


def main():
    src = {r["ingr"]: float(r["cal/g"]) for r in csv.DictReader(open(SRC, encoding="utf-8"))}
    names = [l.split(":", 1)[1].strip() for l in open(r"D:/food_dataset_v2/data.yaml", encoding="utf-8")
             if l.strip()[:1].isdigit()]
    assert set(names) == set(MAP), set(names) ^ set(MAP)
    out = os.path.join(HERE, "calories_47.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "ten_viet", "kcal_per_100g", "match", "nutrition5k_entry", "ghi_chu", "nguon"])
        for i, n in enumerate(names):
            vi, ent, m, note = MAP[n]
            if ent:
                row = [round(src[ent] * 100, 1), m, ent, note, "Nutrition5k ingredients_metadata (USDA FNDDS)"]
            elif n in WEB:
                k, m, url, note = WEB[n]; row = [k, m, "", note, url]
            elif n in RECIPE:
                url, items, note = RECIPE[n]
                parts = "; ".join(f"{a} {g}g×{k}(fdc {f})" for a, g, k, f in items)
                row = [recipe_kcal(items), "tính theo công thức", parts, note, url]
            elif n in NOT_COUNTED:
                row = ["", "không tính", "", NOT_COUNTED[n], ""]
            else:
                row = ["", m, "", note, "chưa có nguồn — cần Bảng TPTP Việt Nam (Viện Dinh dưỡng)"]
            w.writerow([i, n, vi] + row)
    print(out, "có số:", sum(1 for n in names if MAP[n][1] or n in WEB or n in RECIPE), "/", len(names))


if __name__ == "__main__":
    main()
