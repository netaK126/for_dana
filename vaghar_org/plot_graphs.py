import os
import matplotlib.pyplot as plt
import numpy as np

def create_stt_bar_chart(directory_path,c_tag):
    sub_labels = ["vaghar with PertruebedIntervals",
                  "vaghar",
                  "PertruebedIntervals",
                  "MipVerify",
                  "vaghar no hints"]
    
    # מבנה נתונים: { num2: { label: [times] } }
    data_map = {}
    
    # 1. איסוף ועיבוד הנתונים
    for filename in os.listdir(directory_path):
        if "cTag"+c_tag in filename:
            found_label = ""
            if "NoHints" in filename:
                found_label = "vaghar no hints"
            elif "HyperAttack_VagharDeps_PertruebedIntervals" in filename:
                found_label = "vaghar with PertruebedIntervals"
            elif "HyperAttack_VagharDeps" in filename:
                found_label = "vaghar"
            elif "PertruebedIntervals" in filename:
                found_label = "PertruebedIntervals"
            else:
                found_label="MipVerify"
            #next((s for s in sub_labels if s in filename), None)
            if not found_label:
                continue
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            num2 = int(float(parts[1]))
                            if num2>5:
                                continue
                            time_val = float(parts[4])
                            
                            if num2 not in data_map:
                                data_map[num2] = {label: [] for label in sub_labels}
                            # שומרים את ה-time תחת ה-label המתאים ל-num2 הזה
                            data_map[num2][found_label].append(time_val)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if not data_map:
        print("data not found.")
        return

    # 2. הכנת הנתונים לציור
    sorted_num2_keys = sorted(data_map.keys())
    x_indexes = np.arange(len(sorted_num2_keys))  # מיקומי הקבוצות על ציר X
    total_width = 0.64  # הרוחב הכולל של קבוצת עמודות מעל num2 אחד
    single_width = total_width / len(sub_labels)  # רוחב של עמודה בודדת

    plt.figure(figsize=(16, 8))
    # הגדרת צבעים קבועים לכל str
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#1613d3"]
    
    # 3. ציור העמודות לפי קבוצות (Grouped)
    for i, label in enumerate(sub_labels):
        # ממוצע של ה-time אם יש כמה מופעים לאותו num2 ו-str
        values = []
        for n2 in sorted_num2_keys:
            times = data_map[n2][label]
            values.append(sum(times) / len(times) if times else 0)
        
        # חישוב המיקום המדויק של העמודה הנוכחית בתוך הקבוצה
        # אנחנו מזיזים כל קבוצה כך שהמרכז יהיה מעל ה-tick
        pos = x_indexes - (total_width/2) + (i * single_width) + (single_width/2)
        
        bars = plt.bar(pos, values, single_width, label=label, color=colors[i], edgecolor='black', alpha=0.8)

        # # הוספת כיתוב ה-label מעל העמודה (רק אם יש ערך)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, str(round(height,2)),
                         ha='center', va='bottom', fontsize=8, rotation=45)

    # עיצוב הגרף
    plt.xlabel('c_target', fontweight='bold')
    plt.ylabel('Time', fontweight='bold')
    plt.title('Time per c_target, when c_tag='+c_tag, fontsize=14)
    plt.xticks(x_indexes, sorted_num2_keys) # הגדרת הערכים המקוריים של num2 על הציר
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = "cTag"+c_tag+".png"
    plt.savefig(save_path, dpi=300)
    print(f"saved plot in: {save_path}")
    plt.show()

# כאן יש להזין את הנתיב לתקייה שלך
c_tag_list = ["1","2","3"]
for c_tag in c_tag_list:
    path_to_files = r"/root/Downloads/vaghar_org/results_PerturbationInterval/"
    create_stt_bar_chart(path_to_files,c_tag)