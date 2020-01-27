# ===========================================================================
#   format.py ---------------------------------------------------------------
# ===========================================================================

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def print_data(data):
    if isinstance(data, dict):
        print_dict(data)
    if isinstance(data, list):
        if not isinstance(data[0], tuple):
            print_list(data)
        else:
            print_table(data)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def print_dict(data):
    data = [(str(item), str(value)) for item, value in data.items()]
    print_table(data)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def print_list(data):
    for item in data:
        print(item)

#   function ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def print_table(data):
    col_width = [max(len(x) for x in col) for col in zip(*data)]
    for line in data:
        print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")