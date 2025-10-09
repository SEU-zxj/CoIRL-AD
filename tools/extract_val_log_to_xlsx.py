import json
import xlsxwriter

# Specify the input and output file paths here
input_log_file = ""  # <-- Change to your log file path
output_excel_file = "" # <-- Change to your output Excel file path

# Keys to extract and order in Excel
wanted_keys_dict = {
    "epoch": "epoch",
    "plan_L2_1s": "L2 (1s)",
    "plan_L2_2s": "L2 (2s)",
    "plan_L2_3s": "L2 (3s)",
    "plan_obj_box_col_1s": "Col (1s)",
    "plan_obj_box_col_2s": "Col (2s)",
    "plan_obj_box_col_3s": "Col (3s)",

    "plan_L2_1s_rl_actor": "L2 (1s)",
    "plan_L2_2s_rl_actor": "L2 (2s)",
    "plan_L2_3s_rl_actor": "L2 (3s)",
    "plan_obj_box_col_1s_rl_actor": "Col (1s)",
    "plan_obj_box_col_2s_rl_actor": "Col (2s)",
    "plan_obj_box_col_3s_rl_actor": "Col (3s)"
}

def process_log(log_file):
    val_data = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("mode") == "val":
                row = []
                for k in wanted_keys_dict.keys():
                    v = entry.get(k, None)
                    # Multiply plan_obj_box_col_xs by 100
                    if k.startswith("plan_obj_box_col_") and v is not None:
                        v = float(v) * 100
                    row.append(v)
                val_data.append(row)
    return val_data

def write_to_excel(data, keys, out_path):
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet("val_metrics")
    # Write header
    for col, key in enumerate(keys):
        worksheet.write(0, col, key)
    # Write data rows
    for row_idx, row in enumerate(data, 1):
        for col_idx, value in enumerate(row):
            worksheet.write(row_idx, col_idx, value)
    workbook.close()

if __name__ == "__main__":
    val_rows = process_log(input_log_file)
    write_to_excel(val_rows, wanted_keys_dict.values(), output_excel_file)
    print(f"Extracted {len(val_rows)} rows to {output_excel_file}")