import math

def format_value(value, fmt: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return format(value, fmt)

def format_stage_table(row_labels: list[str], col_labels: list[str], cells: dict, fmt: str = ".4f", first_header: str = "지표") -> str:
    if not row_labels or not col_labels:
        raise ValueError("row_labels and col_labels must not be empty")

    lines = []
    
    # Header
    header = "| " + first_header + " | " + " | ".join(col_labels) + " |"
    lines.append(header)
    
    # Separator
    sep = "| --- |" + " --- |" * len(col_labels)
    lines.append(sep)
    
    # Rows
    for row in row_labels:
        if isinstance(fmt, dict):
            current_fmt = fmt.get(row, ".4f")
        else:
            current_fmt = fmt
            
        row_cells = []
        for col in col_labels:
            val = cells.get((row, col))
            row_cells.append(format_value(val, current_fmt))
        lines.append("| " + row + " | " + " | ".join(row_cells) + " |")
        
    return "\n".join(lines)
